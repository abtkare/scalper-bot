"""cTrader adapter tests against a simulated cTrader Open API server (no network).
Run: python test_ctrader.py
"""
import json
import logging
import os
import queue
import tempfile
import threading
import time

import bot
import test_robust as tr
from broker import PaperBroker
from config import SymbolCfg
from ctrader import PT, CTraderBroker, CTraderClient, RoutingBroker

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


class FakeServer:
    """Speaks the JSON Open API protocol: auth, symbols, market orders with relative SL/TP, reconcile, close."""

    def __init__(self, prices=None, reject=False, silent_fill=False, token_expired=False):
        self.out: "queue.Queue" = queue.Queue()
        self.sent = []
        self.positions = {}
        self.pid = 1000
        self.prices = prices or {1: 4000.0, 2: 1.1000}
        self.reject, self.silent_fill, self.token_expired = reject, silent_fill, token_expired
        self.connected = True

    # connection interface used by CTraderClient
    def send(self, text):
        if not self.connected:
            raise ConnectionError("down")
        m = json.loads(text)
        self.sent.append(m)
        pt, p, mid = m["payloadType"], m.get("payload", {}), m.get("clientMsgId")
        r = lambda t, body: self.out.put(json.dumps({"clientMsgId": mid, "payloadType": t, "payload": body}))
        ev = lambda body: self.out.put(json.dumps({"payloadType": PT["EXECUTION_EVENT"], "payload": body}))
        if pt == PT["APP_AUTH_REQ"]:
            r(PT["APP_AUTH_RES"], {})
        elif pt == PT["ACCOUNTS_BY_TOKEN_REQ"]:
            r(PT["ACCOUNTS_BY_TOKEN_RES"], {"ctidTraderAccount": [{"ctidTraderAccountId": 9, "isLive": True},
                                                                  {"ctidTraderAccountId": 42, "isLive": False}]})
        elif pt == PT["ACCOUNT_AUTH_REQ"]:
            if self.token_expired and p["accessToken"] == "old":
                r(PT["ERROR_RES"], {"errorCode": "CH_ACCESS_TOKEN_INVALID", "description": "expired"})
            else:
                r(PT["ACCOUNT_AUTH_RES"], {"ctidTraderAccountId": p["ctidTraderAccountId"]})
        elif pt == PT["REFRESH_TOKEN_REQ"]:
            r(PT["REFRESH_TOKEN_RES"], {"accessToken": "new", "refreshToken": "r2"})
        elif pt == PT["SYMBOLS_LIST_REQ"]:
            r(PT["SYMBOLS_LIST_RES"], {"symbol": [{"symbolId": 1, "symbolName": "XAUUSD", "enabled": True},
                                                  {"symbolId": 2, "symbolName": "EURUSD", "enabled": True}]})
        elif pt == PT["SYMBOL_BY_ID_REQ"]:
            det = {1: {"symbolId": 1, "digits": 2, "minVolume": 100, "stepVolume": 100, "maxVolume": 10 ** 9},
                   2: {"symbolId": 2, "digits": 5, "minVolume": 100000, "stepVolume": 100000, "maxVolume": 10 ** 12}}
            r(PT["SYMBOL_BY_ID_RES"], {"symbol": [det[i] for i in p["symbolId"]]})
        elif pt == PT["NEW_ORDER_REQ"]:
            if self.reject:
                ev({"executionType": 7, "order": {"clientOrderId": p["clientOrderId"]}, "errorCode": "NOT_ENOUGH_MONEY"})
                return
            self.pid += 1
            px = self.prices[p["symbolId"]]
            pos = {"positionId": self.pid, "price": px, "tradeData": {
                "symbolId": p["symbolId"], "volume": p["volume"], "tradeSide": p["tradeSide"],
                "label": p["label"], "comment": p["comment"]}, "relSL": p["relativeStopLoss"],
                "relTP": p["relativeTakeProfit"]}
            self.positions[self.pid] = pos
            if not self.silent_fill:
                ev({"executionType": 2, "order": {"clientOrderId": p["clientOrderId"]}})
                ev({"executionType": 3, "order": {"clientOrderId": p["clientOrderId"]}, "position": pos,
                    "deal": {"executionPrice": px, "filledVolume": p["volume"], "positionId": self.pid}})
        elif pt == PT["RECONCILE_REQ"]:
            r(PT["RECONCILE_RES"], {"position": list(self.positions.values())})
        elif pt == PT["CLOSE_POSITION_REQ"]:
            pos = self.positions.pop(p["positionId"], None)
            if pos:
                ev({"executionType": 3, "position": {"positionId": p["positionId"]},
                    "deal": {"executionPrice": self.prices[pos["tradeData"]["symbolId"]] + 1, "positionId": p["positionId"]}})

    def recv(self):
        try:
            return self.out.get(timeout=0.05)
        except queue.Empty:
            return None

    def close(self):
        pass


def make(server=None, equity=1000.0, **kw):
    server = server or FakeServer(**kw)
    d = tempfile.mkdtemp()
    c = CTraderClient("cid", "sec", kw.get("token", "tok"), kw.get("refresh", ""), conn_factory=lambda u: server,
                      token_file=os.path.join(d, "t.json"))
    b = CTraderBroker(c, equity_fn=lambda: equity, fill_timeout=1.0)
    return server, c, b


def test_connect_picks_demo():
    s, c, b = make()
    assert c.ensure() and c.account_id == 42, "must authenticate on the DEMO account, never the live one"
    ok("connect: app auth -> picks the DEMO account (42), not the live one (9) -> account auth")


def test_entry_bracket():
    s, c, b = make()
    f = b.market_entry("XAU/USD", 1, 2.0, 4000.0, "XAU-1", stop_dist=12.3456, target_dist=18.5, ticker="XAU/USD")
    assert f.ok and f.price == 4000.0 and f.qty == 2.0, f
    order = [m for m in s.sent if m["payloadType"] == PT["NEW_ORDER_REQ"]][0]["payload"]
    assert order["volume"] == 200 and order["tradeSide"] == 1 and order["orderType"] == 1
    assert order["relativeStopLoss"] == 1235000 and order["relativeTakeProfit"] == 1850000, order
    assert order["label"] == "XAU/USD" and order["clientOrderId"] == "XAU-1"
    ok("entry: market order with SERVER-SIDE stop & target (relative, rounded to the symbol's digits), "
       "clientOrderId + label set")


def test_min_lot_rules():
    s, c, b = make(equity=1000.0)
    f = b.market_entry("XAU/USD", 1, 0.4, 4000.0, "g1", stop_dist=12.0, target_dist=18.0, ticker="XAU/USD")
    assert f.ok and f.qty == 1.0 and "min lot" in f.reason, f          # 1 oz x 12$ = 1.2% <= 2%
    s2, c2, b2 = make(equity=300.0)
    f2 = b2.market_entry("XAU/USD", 1, 0.4, 4000.0, "g2", stop_dist=12.0, target_dist=18.0, ticker="XAU/USD")
    assert not f2.ok and "minimum" in f2.reason, f2                    # 12$ = 4% of 300 -> skipped
    ok("size below the broker's minimum lot: min lot only if risk <= 2% of equity, otherwise skipped")


def test_refuses_without_stop():
    s, c, b = make()
    f = b.market_entry("XAU/USD", 1, 1.0, 4000.0, "x", ticker="XAU/USD")
    assert not f.ok and "stop" in f.reason and not [m for m in s.sent if m["payloadType"] == PT["NEW_ORDER_REQ"]]
    ok("an order without a stop-loss is never sent")


def test_reject_and_timeout():
    s, c, b = make(reject=True)
    f = b.market_entry("XAU/USD", 1, 1.0, 4000.0, "r1", stop_dist=10, target_dist=15, ticker="XAU/USD")
    assert not f.ok and "rejected" in f.reason
    s2, c2, b2 = make(silent_fill=True)
    f2 = b2.market_entry("XAU/USD", -1, 1.0, 4000.0, "t1", stop_dist=10, target_dist=15, ticker="XAU/USD")
    assert f2.ok and "reconcile" in f2.reason, f2
    news = [m for m in s2.sent if m["payloadType"] == PT["NEW_ORDER_REQ"]]
    assert len(news) == 1, "a timeout must never re-send the order (no duplicate position)"
    ok("rejection -> Fill(ok=False); lost confirmation -> found by reconcile, order NOT re-sent")


def test_exit_paths():
    s, c, b = make()
    b.market_entry("XAU/USD", 1, 1.0, 4000.0, "e1", stop_dist=10, target_dist=15, ticker="XAU/USD")
    f = b.market_exit("XAU/USD", 1, 1.0, 4005.0, "x1", ticker="XAU/USD")
    assert f.ok and f.price == 4001.0 and not s.positions, f
    f2 = b.market_exit("XAU/USD", 1, 1.0, 3990.0, "x2", ticker="XAU/USD")
    assert f2.ok and "already closed" in f2.reason                       # server SL/TP closed it
    ok("exit: closes the right position (by label) at the broker's price; already closed by server SL/TP -> ok")


def test_routing():
    s, c, b = make()
    d = tempfile.mkdtemp()
    rb = RoutingBroker(b, PaperBroker(), routes_file=os.path.join(d, "r.json"))
    f = rb.market_entry("QQQ", 1, 3.0, 600.0, "q1", stop_dist=1, target_dist=2, ticker="QQQ")
    assert f.ok and f.order_id == "q1", "symbol not at the broker -> paper"
    f = rb.market_entry("XAU/USD", 1, 1.0, 4000.0, "g1", stop_dist=10, target_dist=15, ticker="XAU/USD")
    assert f.ok and f.order_id.isdigit(), "broker symbol -> cTrader"
    s.connected = False
    c.ready.clear()
    c._last_try = time.time()
    f = rb.market_entry("XAU/USD", 1, 1.0, 4000.0, "g2", stop_dist=10, target_dist=15, ticker="XAU/USD")
    assert not f.ok and "unreachable" in f.reason, "a cTrader symbol must never fall back to paper"
    f = rb.market_exit("XAU/USD", 1, 1.0, 4000.0, "gx", ticker="XAU/USD")
    assert not f.ok, "exit of a live position while disconnected must FAIL (bot retries), never close on paper"
    rb2 = RoutingBroker(b, PaperBroker(), routes_file=os.path.join(d, "r.json"))
    assert "XAU/USD" in rb2.live_tickers, "live routes must survive a restart"
    ok("routing: broker symbols -> cTrader, others -> paper; a cTrader symbol never falls back to paper "
       "(entries skipped, exits retried while disconnected); routes persisted")


def test_token_refresh():
    s = FakeServer(token_expired=True)
    d = tempfile.mkdtemp()
    tf = os.path.join(d, "t.json")
    c = CTraderClient("cid", "sec", "old", "r1", conn_factory=lambda u: s, token_file=tf)
    assert c.ensure() and c.access_token == "new"
    saved = json.load(open(tf))
    assert saved["refresh_token"] == "r2" and saved["env_refresh"] == "r1"
    c2 = CTraderClient("cid", "sec", "old", "r1", conn_factory=lambda u: s, token_file=tf)
    assert c2.access_token == "new", "refreshed token reused after restart"
    c3 = CTraderClient("cid", "sec", "pasted", "r9", conn_factory=lambda u: s, token_file=tf)
    assert c3.access_token == "pasted", "tokens newly pasted into the env win over the saved file"
    ok("expired access token -> refreshed automatically and saved; newly pasted tokens win")


def test_bot_passes_bracket_and_real_qty():
    calls = []

    class Spy(PaperBroker):
        def market_entry(self, symbol, side, qty, ref_price, client_id, **kw):
            calls.append(kw)
            from broker import Fill
            return Fill(True, price=ref_price, qty=round(qty) or 1.0, order_id="77", reason="min lot used")
    cfg = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", strategy="scalp",
                    allow_short=True, stop_atr=2.0, target_atr=0.75)
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.0), cfg=cfg, sig_side=1, broker=Spy())
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    kw = calls[0]
    assert abs(kw["stop_dist"] - 2.0) < 1e-9 and abs(kw["target_dist"] - 0.75) < 1e-9 and kw["ticker"] == "TEST/USD"
    pos = e.ss.eng.pos
    assert pos.qty == round(pos.qty) and abs(pos.risk_amt - pos.qty * 2.0) < 1e-9, "engine uses the broker's real qty"
    ok("bot sends stop/target distances + ticker to the broker and books the broker's real (rounded) quantity")


if __name__ == "__main__":
    real = bot.signals
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            bot.signals = real
            fn()
    bot.signals = real
    print(f"\nall {len(PASSED)} ctrader tests passed")
