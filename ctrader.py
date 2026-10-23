"""cTrader Open API execution (DEMO accounts only in this version).

Protocol: JSON messages over a WebSocket (wss://demo.ctraderapi.com:5036), as documented at
help.ctrader.com/open-api (proxies-endpoints, sending-receiving-json, account-authentication).
Every message is {"clientMsgId": str, "payloadType": int, "payload": {...}}.

What it guarantees (broker.py rules):
  * every entry carries a SERVER-SIDE stop-loss and take-profit (relative to the real fill price), so a
    position stays protected even if the bot, the VPS or the internet dies;
  * clientOrderId = the bot's client id; label = the bot's stream id -> positions are found again after
    a restart (reconcile), and a timeout is resolved by looking for the position instead of re-sending;
  * every rejection / timeout / ambiguity becomes Fill(ok=False) - nothing raises into the trade loop;
  * a position already closed on the server (its stop or target was hit) is reported as closed, so the
    bot never loops on an exit that cannot happen.

Configuration (environment of the container, never in the repository):
  EXECUTION=ctrader_demo           switch the bot from paper fills to the cTrader demo account
  CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET   from openapi.ctrader.com -> Applications -> Credentials
  CTRADER_ACCESS_TOKEN / CTRADER_REFRESH_TOKEN from Applications -> Sandbox/Playground -> Get token
  CTRADER_ACCOUNT_ID (optional)    ctidTraderAccountId; default = the first DEMO account of the token
  CTRADER_SYMBOL_MAP (optional)    e.g. "XAU/USD=XAUUSD,QQQ=US100" when the broker uses other names
  CTRADER_MAX_MIN_LOT_RISK=0.02    if the strategy's size is below the broker's minimum lot, trade the
                                   minimum lot only when that risks <= 2% of equity, otherwise skip
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import queue
import re
import threading
import time
from typing import Callable, Dict, Optional

from broker import Broker, Fill

log = logging.getLogger("ctrader")

DEMO_URL = "wss://demo.ctraderapi.com:5036"
TOKEN_FILE = "ctrader_tokens.json"

PT = {
    "HEARTBEAT": 51,
    "APP_AUTH_REQ": 2100, "APP_AUTH_RES": 2101,
    "ACCOUNT_AUTH_REQ": 2102, "ACCOUNT_AUTH_RES": 2103,
    "NEW_ORDER_REQ": 2106,
    "CLOSE_POSITION_REQ": 2111,
    "SYMBOLS_LIST_REQ": 2114, "SYMBOLS_LIST_RES": 2115,
    "SYMBOL_BY_ID_REQ": 2116, "SYMBOL_BY_ID_RES": 2117,
    "TRADER_REQ": 2121, "TRADER_RES": 2122,
    "RECONCILE_REQ": 2124, "RECONCILE_RES": 2125,
    "EXECUTION_EVENT": 2126,
    "ORDER_ERROR_EVENT": 2132,
    "ERROR_RES": 2142,
    "ACCOUNTS_BY_TOKEN_REQ": 2149, "ACCOUNTS_BY_TOKEN_RES": 2150,
    "REFRESH_TOKEN_REQ": 2173, "REFRESH_TOKEN_RES": 2174,
}
EXEC_FILLED, EXEC_REJECTED, EXEC_PARTIAL = 3, 7, 11
BUY, SELL = 1, 2
MARKET = 1
PRICE_SCALE = 100000          # relative SL/TP and spot prices are in 1/100000 of the price unit


class CTraderError(Exception):
    pass


def norm_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


class _WS:
    """Thin wrapper so tests can inject a fake connection."""

    def __init__(self, url: str, timeout: float = 10.0):
        import websocket                                    # websocket-client
        self.ws = websocket.create_connection(url, timeout=timeout)
        self.ws.settimeout(1.0)

    def send(self, text: str) -> None:
        self.ws.send(text)

    def recv(self) -> Optional[str]:
        import websocket
        try:
            return self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


class CTraderClient:
    """One authenticated connection: request/response by clientMsgId, events to a queue, heartbeat,
    automatic reconnect + re-authentication."""

    def __init__(self, client_id: str, client_secret: str, access_token: str, refresh_token: str = "",
                 account_id: int = 0, url: str = DEMO_URL, conn_factory: Callable[[str], object] = None,
                 token_file: str = TOKEN_FILE, clock=time.time):
        self.client_id, self.client_secret = client_id, client_secret
        self.access_token, self.refresh_token = access_token, refresh_token
        self.account_id = int(account_id or 0)
        self.url = url
        self.conn_factory = conn_factory or (lambda u: _WS(u))
        self.token_file, self.clock = token_file, clock
        self._ids = itertools.count(1)
        self._waiters: Dict[str, "queue.Queue"] = {}
        self.events: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self.conn = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.last_error = ""
        self._load_tokens()

    # ---------------------------------------------------------------- tokens
    def _load_tokens(self) -> None:
        """Refreshed tokens are saved to a file. They win over the env values only while the env values
        are the same ones they were refreshed from (new tokens pasted into the env always win)."""
        self.env_refresh = self.refresh_token
        try:
            with open(self.token_file, encoding="utf-8") as f:
                t = json.load(f)
            if t.get("env_refresh") == self.env_refresh and t.get("refresh_token"):
                self.access_token = t.get("access_token") or self.access_token
                self.refresh_token = t["refresh_token"]
        except (OSError, ValueError):
            pass

    def _save_tokens(self) -> None:
        try:
            tmp = self.token_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"access_token": self.access_token, "refresh_token": self.refresh_token,
                           "env_refresh": self.env_refresh, "saved_at": self.clock()}, f)
            os.replace(tmp, self.token_file)
            os.chmod(self.token_file, 0o600)
        except OSError as e:
            log.warning("could not save cTrader tokens: %s", e)

    # ---------------------------------------------------------------- low level
    def _send(self, payload_type: int, payload: dict, msg_id: Optional[str] = None) -> str:
        msg_id = msg_id or f"m{next(self._ids)}"
        text = json.dumps({"clientMsgId": msg_id, "payloadType": payload_type, "payload": payload})
        with self._send_lock:
            if self.conn is None:
                raise CTraderError("not connected")
            self.conn.send(text)
        return msg_id

    def request(self, payload_type: int, payload: dict, expect: int, timeout: float = 10.0) -> dict:
        msg_id = f"m{next(self._ids)}"
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._waiters[msg_id] = q
        try:
            self._send(payload_type, payload, msg_id)
            try:
                pt, body = q.get(timeout=timeout)
            except queue.Empty:
                raise CTraderError(f"timeout waiting for {expect}")
        finally:
            with self._lock:
                self._waiters.pop(msg_id, None)
        if pt == PT["ERROR_RES"]:
            raise CTraderError(f"{body.get('errorCode')}: {body.get('description', '')}")
        if pt != expect:
            raise CTraderError(f"unexpected reply {pt} (wanted {expect})")
        return body

    def _dispatch(self, text: str) -> None:
        try:
            m = json.loads(text)
        except ValueError:
            return
        pt, body, mid = m.get("payloadType"), m.get("payload") or {}, m.get("clientMsgId")
        if pt == PT["HEARTBEAT"]:
            return
        with self._lock:
            q = self._waiters.get(mid) if mid else None
        if q is not None:
            q.put((pt, body))
        if pt in (PT["EXECUTION_EVENT"], PT["ORDER_ERROR_EVENT"]) or (pt == PT["ERROR_RES"] and q is None):
            self.events.put((pt, body))

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        """Connect + application auth + account auth (blocking, raises CTraderError)."""
        self.ready.clear()
        if self.conn is not None:
            self.conn.close()
        self.conn = self.conn_factory(self.url)
        threading.Thread(target=self._reader, args=(self.conn,), name="ctrader-rx", daemon=True).start()
        self.request(PT["APP_AUTH_REQ"], {"clientId": self.client_id, "clientSecret": self.client_secret},
                     PT["APP_AUTH_RES"])
        if not self.account_id:
            res = self.request(PT["ACCOUNTS_BY_TOKEN_REQ"], {"accessToken": self.access_token},
                               PT["ACCOUNTS_BY_TOKEN_RES"])
            demo = [a for a in res.get("ctidTraderAccount", []) if not a.get("isLive")]
            if not demo:
                raise CTraderError("no DEMO account found for this access token (open one in the cTrader app)")
            self.account_id = int(demo[0]["ctidTraderAccountId"])
        try:
            self.request(PT["ACCOUNT_AUTH_REQ"], {"ctidTraderAccountId": self.account_id,
                                                   "accessToken": self.access_token}, PT["ACCOUNT_AUTH_RES"])
        except CTraderError as e:
            if not self.refresh_token or "TOKEN" not in str(e).upper():
                raise
            self.refresh()
            self.request(PT["ACCOUNT_AUTH_REQ"], {"ctidTraderAccountId": self.account_id,
                                                   "accessToken": self.access_token}, PT["ACCOUNT_AUTH_RES"])
        self.ready.set()
        log.info("cTrader connected: demo account %s", self.account_id)

    def refresh(self) -> None:
        res = self.request(PT["REFRESH_TOKEN_REQ"], {"refreshToken": self.refresh_token}, PT["REFRESH_TOKEN_RES"])
        self.access_token = res.get("accessToken") or self.access_token
        self.refresh_token = res.get("refreshToken") or self.refresh_token
        self._save_tokens()
        log.warning("cTrader access token refreshed")

    def _reader(self, conn) -> None:
        last_hb = 0.0
        while not self.stopping.is_set() and conn is self.conn:
            try:
                if time.time() - last_hb >= 10.0:                  # Open API: heartbeat every 10 s
                    with self._send_lock:
                        conn.send(json.dumps({"payloadType": PT["HEARTBEAT"], "payload": {}}))
                    last_hb = time.time()
                text = conn.recv()
                if text:
                    self._dispatch(text)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("cTrader connection lost: %s", self.last_error)
                self.ready.clear()
                break

    def ensure(self) -> bool:
        if self.ready.is_set():
            return True
        if self.clock() - getattr(self, "_last_try", 0.0) < 30.0:      # never hammer / never block the loop often
            return False
        self._last_try = self.clock()
        try:
            self.connect()
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("cTrader connect failed: %s", self.last_error)
            return False

    def close(self) -> None:
        self.stopping.set()
        if self.conn is not None:
            self.conn.close()


class CTraderBroker(Broker):
    name = "ctrader_demo"

    def __init__(self, client: CTraderClient, symbol_map: Optional[Dict[str, str]] = None,
                 max_min_lot_risk: float = 0.02, equity_fn: Callable[[], float] = lambda: 0.0,
                 fill_timeout: float = 12.0):
        self.c = client
        self.symbol_map = {k.upper(): v for k, v in (symbol_map or {}).items()}
        self.max_min_lot_risk = max_min_lot_risk
        self.equity_fn = equity_fn
        self.fill_timeout = fill_timeout
        self.symbols: Dict[str, dict] = {}           # normalised name -> light symbol
        self.details: Dict[int, dict] = {}            # symbolId -> full symbol
        self._sym_loaded = 0.0

    # ---------------------------------------------------------------- symbols
    def _load_symbols(self) -> None:
        if self.symbols and time.time() - self._sym_loaded < 6 * 3600:
            return
        res = self.c.request(PT["SYMBOLS_LIST_REQ"], {"ctidTraderAccountId": self.c.account_id},
                             PT["SYMBOLS_LIST_RES"])
        self.symbols = {norm_name(s.get("symbolName", "")): s for s in res.get("symbol", []) if s.get("enabled", True)}
        self._sym_loaded = time.time()

    def resolve(self, ticker: str) -> Optional[dict]:
        """Bot ticker (XAU/USD, EUR/USD, QQQ) -> broker symbol with digits / volume limits, or None."""
        self._load_symbols()
        want = self.symbol_map.get(ticker.upper(), ticker)
        light = self.symbols.get(norm_name(want))
        if light is None:
            return None
        sid = int(light["symbolId"])
        if sid not in self.details:
            res = self.c.request(PT["SYMBOL_BY_ID_REQ"], {"ctidTraderAccountId": self.c.account_id,
                                                          "symbolId": [sid]}, PT["SYMBOL_BY_ID_RES"])
            for s in res.get("symbol", []):
                self.details[int(s["symbolId"])] = s
        d = dict(self.details.get(sid, {}))
        d.setdefault("symbolId", sid)
        d["symbolName"] = light.get("symbolName")
        return d

    def can_trade(self, ticker: str) -> bool:
        try:
            return self.c.ensure() and self.resolve(ticker) is not None
        except Exception:
            return False

    # ---------------------------------------------------------------- sizing helpers
    def volume_for(self, sym: dict, qty: float, stop_dist: float, ref_price: float, conv: float) -> tuple:
        """Bot quantity (units) -> broker volume (units x 100), rounded DOWN to the step; returns (volume, note)."""
        step = int(sym.get("stepVolume") or 100)
        vmin = int(sym.get("minVolume") or step)
        vmax = int(sym.get("maxVolume") or 10 ** 12)
        vol = int(math.floor(qty * 100 / step) * step)
        note = ""
        if vol < vmin:
            eq = self.equity_fn() or 0.0
            risk = vmin / 100.0 * stop_dist * conv
            if eq > 0 and risk <= self.max_min_lot_risk * eq:
                note = f"min lot used (risk {risk:.2f}$ = {100 * risk / eq:.2f}% of equity)"
                vol = vmin
            else:
                return 0, (f"size {qty:.4f} is below the broker minimum ({vmin / 100:g} units); min lot would "
                           f"risk {risk:.2f}$ (> {100 * self.max_min_lot_risk:g}% of equity) - skipped")
        return min(vol, vmax), note

    @staticmethod
    def rel(dist: float, digits: int) -> int:
        stepu = 10 ** max(0, 5 - int(digits))
        return int(max(stepu, round(dist * PRICE_SCALE / stepu) * stepu))

    # ---------------------------------------------------------------- positions
    def _positions(self) -> list:
        res = self.c.request(PT["RECONCILE_REQ"], {"ctidTraderAccountId": self.c.account_id}, PT["RECONCILE_RES"])
        return res.get("position", []) or []

    def _find(self, label: str, comment: str = "") -> Optional[dict]:
        for p in self._positions():
            td = p.get("tradeData") or {}
            if td.get("label") == label and (not comment or td.get("comment") == comment):
                return p
        return None

    def sync_positions(self) -> Optional[Dict[str, float]]:
        if not self.c.ensure():
            return None
        try:
            out: Dict[str, float] = {}
            for p in self._positions():
                td = p.get("tradeData") or {}
                lab = td.get("label") or f"#{td.get('symbolId')}"
                sgn = 1 if int(td.get("tradeSide", BUY)) == BUY else -1
                out[lab] = out.get(lab, 0.0) + sgn * float(td.get("volume", 0)) / 100.0
            return out
        except Exception as e:
            log.warning("cTrader reconcile failed: %s", e)
            return None

    # ---------------------------------------------------------------- execution
    def _wait_fill(self, match: Callable[[dict], bool], timeout: float) -> tuple:
        end = time.time() + timeout
        while time.time() < end:
            try:
                pt, body = self.c.events.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                break
            if pt == PT["EXECUTION_EVENT"] and match(body):
                et = int(body.get("executionType", 0))
                if et in (EXEC_FILLED, EXEC_PARTIAL):
                    return "filled", body
                if et == EXEC_REJECTED:
                    return "rejected", body
            elif pt in (PT["ORDER_ERROR_EVENT"], PT["ERROR_RES"]) and match(body):
                return "rejected", body
        return "timeout", {}

    def market_entry(self, symbol, side, qty, ref_price, client_id, stop_dist: float = 0.0,
                     target_dist: float = 0.0, ticker: str = "", conv: float = 1.0) -> Fill:
        try:
            if not (stop_dist > 0 and target_dist > 0):
                return Fill(False, reason="refused: no stop/target distance (server-side stop is mandatory)")
            if not self.c.ensure():
                return Fill(False, reason=f"cTrader not connected: {self.c.last_error}")
            sym = self.resolve(ticker or symbol)
            if sym is None:
                return Fill(False, reason=f"{ticker or symbol} is not available at this broker (set CTRADER_SYMBOL_MAP)")
            vol, note = self.volume_for(sym, qty, stop_dist, ref_price, conv)
            if vol <= 0:
                return Fill(False, reason=note)
            digits = int(sym.get("digits", 5))
            sid = int(sym["symbolId"])
            while not self.c.events.empty():                  # drop stale events
                self.c.events.get_nowait()
            self.c._send(PT["NEW_ORDER_REQ"], {
                "ctidTraderAccountId": self.c.account_id, "symbolId": sid, "orderType": MARKET,
                "tradeSide": BUY if side == 1 else SELL, "volume": vol,
                "relativeStopLoss": self.rel(stop_dist, digits), "relativeTakeProfit": self.rel(target_dist, digits),
                "label": symbol[:100], "comment": client_id[:500], "clientOrderId": client_id[:50]})

            def mine(b):
                o = b.get("order") or {}
                return o.get("clientOrderId") == client_id[:50] or not o
            status, body = self._wait_fill(mine, self.fill_timeout)
            if status == "filled":
                deal = body.get("deal") or {}
                px = float(deal.get("executionPrice") or (body.get("position") or {}).get("price") or ref_price)
                filled = float(deal.get("filledVolume") or vol) / 100.0
                pid = str((body.get("position") or {}).get("positionId") or deal.get("positionId") or "")
                return Fill(True, price=px, qty=filled, order_id=pid, reason=note)
            if status == "rejected":
                return Fill(False, reason=f"rejected: {body.get('errorCode', '')} {body.get('description', '')}".strip())
            p = self._find(symbol[:100], client_id[:500])           # timeout: did it fill anyway?
            if p is not None:
                return Fill(True, price=float(p.get("price") or ref_price),
                            qty=float((p.get("tradeData") or {}).get("volume", vol)) / 100.0,
                            order_id=str(p.get("positionId", "")), reason="filled (confirmed by reconcile)")
            return Fill(False, reason="no fill confirmation within timeout (no position found)")
        except Exception as e:
            return Fill(False, reason=f"{type(e).__name__}: {e}")

    def market_exit(self, symbol, side, qty, ref_price, client_id, ticker: str = "", **kw) -> Fill:
        try:
            if not self.c.ensure():
                return Fill(False, reason=f"cTrader not connected: {self.c.last_error}")
            p = self._find(symbol[:100])
            if p is None:
                # already closed on the server by its stop-loss / take-profit
                return Fill(True, price=float(ref_price), qty=float(qty), order_id="",
                            reason="already closed on the server (SL/TP)")
            pid = int(p["positionId"])
            vol = int((p.get("tradeData") or {}).get("volume", 0))
            while not self.c.events.empty():
                self.c.events.get_nowait()
            self.c._send(PT["CLOSE_POSITION_REQ"], {"ctidTraderAccountId": self.c.account_id,
                                                    "positionId": pid, "volume": vol})

            def mine(b):
                pos = b.get("position") or {}
                return str(pos.get("positionId", "")) == str(pid) or str(b.get("positionId", "")) == str(pid)
            status, body = self._wait_fill(mine, self.fill_timeout)
            if status == "filled":
                deal = body.get("deal") or {}
                return Fill(True, price=float(deal.get("executionPrice") or ref_price), qty=vol / 100.0,
                            order_id=str(pid))
            if status == "rejected":
                return Fill(False, reason=f"close rejected: {body.get('errorCode', '')} {body.get('description', '')}")
            if self._find(symbol[:100]) is None:
                return Fill(True, price=float(ref_price), qty=vol / 100.0, order_id=str(pid),
                            reason="closed (confirmed by reconcile)")
            return Fill(False, reason="close not confirmed within timeout - will retry")
        except Exception as e:
            return Fill(False, reason=f"{type(e).__name__}: {e}")


class RoutingBroker(Broker):
    """Symbols the cTrader account can trade go to cTrader; everything else stays paper (clearly logged).

    A ticker that was ever traded on cTrader is remembered (routes file): its exits ALWAYS go to cTrader
    (a disconnect makes the exit fail and retry - it never gets "closed" on paper while still open at the
    broker), and new entries on it are refused while cTrader is unreachable."""

    def __init__(self, live: CTraderBroker, paper: Broker, routes_file: str = "ctrader_routes.json"):
        self.live, self.paper = live, paper
        self.name = f"{live.name}+paper"
        self.routes_file = routes_file
        try:
            with open(routes_file, encoding="utf-8") as f:
                self.live_tickers = set(json.load(f))
        except (OSError, ValueError):
            self.live_tickers = set()

    def _remember(self, ticker: str) -> None:
        if ticker in self.live_tickers:
            return
        self.live_tickers.add(ticker)
        try:
            with open(self.routes_file, "w", encoding="utf-8") as f:
                json.dump(sorted(self.live_tickers), f)
        except OSError:
            pass

    def market_entry(self, symbol, side, qty, ref_price, client_id, **kw):
        t = kw.get("ticker") or symbol
        if self.live.can_trade(t):
            self._remember(t)
            return self.live.market_entry(symbol, side, qty, ref_price, client_id, **kw)
        if t in self.live_tickers:
            return Fill(False, reason=f"cTrader unreachable - entry on {t} skipped ({self.live.c.last_error})")
        return self.paper.market_entry(symbol, side, qty, ref_price, client_id, **kw)

    def market_exit(self, symbol, side, qty, ref_price, client_id, **kw):
        t = kw.get("ticker") or symbol
        if t in self.live_tickers:
            return self.live.market_exit(symbol, side, qty, ref_price, client_id, **kw)
        return self.paper.market_exit(symbol, side, qty, ref_price, client_id, **kw)

    def sync_positions(self):
        return self.live.sync_positions()

    def describe(self) -> dict:
        return {"mode": self.name, "connected": self.live.c.ready.is_set(), "account": self.live.c.account_id,
                "live_symbols": sorted(self.live_tickers), "error": self.live.c.last_error[:160]}


def from_env(equity_fn: Callable[[], float]) -> Optional[RoutingBroker]:
    if os.getenv("EXECUTION", "paper").strip().lower() != "ctrader_demo":
        return None
    cid, sec, tok = (os.getenv(k, "").strip() for k in ("CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET",
                                                         "CTRADER_ACCESS_TOKEN"))
    if not (cid and sec and tok):
        raise SystemExit("EXECUTION=ctrader_demo needs CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN")
    smap = {}
    for part in os.getenv("CTRADER_SYMBOL_MAP", "").split(","):
        if "=" in part:
            a, b = part.split("=", 1)
            smap[a.strip()] = b.strip()
    client = CTraderClient(cid, sec, tok, os.getenv("CTRADER_REFRESH_TOKEN", "").strip(),
                           int(os.getenv("CTRADER_ACCOUNT_ID", "0") or 0))
    from broker import PaperBroker
    live = CTraderBroker(client, smap, float(os.getenv("CTRADER_MAX_MIN_LOT_RISK", "0.02") or 0.02), equity_fn)
    return RoutingBroker(live, PaperBroker())
