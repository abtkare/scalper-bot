"""Remote link over ntfy.sh (HTTPS pub/sub) - lets Claude (or you) monitor and control the bot from anywhere,
without opening ports, SSH or a domain.

Topics (NTFY_TOPIC is a long random secret - knowing it is the first key, MONITOR_TOKEN the second):
  <topic>-s   status snapshot every 60 s (compact JSON)             -> read by the monitor
  <topic>-a   alerts: entries, exits, guardian interventions, errors -> subscribe in the ntfy phone app
  <topic>-c   commands: "<cmd> <MONITOR_TOKEN>"                      -> stop|start|flatten|halt|resume|restart|ping

"restart" saves state and exits; Docker restarts the container, which downloads the latest code from GitHub
(see docker-entrypoint.sh) - that is how fixes reach the server without anyone touching it.
The link can only do what the dashboard buttons do. It can never open a trade.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Callable

import requests

from data import redact

log = logging.getLogger("remote")
BASE = os.getenv("NTFY_BASE", "https://ntfy.sh")


class _AlertHandler(logging.Handler):
    """Forwards the bot's notifications (logger 'infra' + 'guardian') to the alert topic."""

    def __init__(self, q: deque):
        super().__init__(level=logging.INFO)
        self.q = q

    def emit(self, record):
        try:
            msg = record.getMessage()
            if record.name == "infra" or record.levelno >= logging.WARNING:
                if "remote link" in msg:                 # never echo our own errors (loop)
                    return
                self.q.append((record.levelno, redact(msg)[:900]))
        except Exception:
            pass


class RemoteLink:
    def __init__(self, topic: str, token: str, status_fn: Callable[[], dict], command_fn: Callable[[str], str],
                 restart_fn: Callable[[], None]):
        self.topic, self.token = topic, token
        self.status_fn, self.command_fn, self.restart_fn = status_fn, command_fn, restart_fn
        self.alerts: deque = deque(maxlen=200)
        self.seen: deque = deque(maxlen=500)              # message ids already executed (exactly-once)
        self.boot = time.time()                            # commands sent before this process started are ignored
        self.sess = requests.Session()

    @property
    def enabled(self) -> bool:
        return len(self.topic) >= 16 and len(self.token) >= 16

    def start(self) -> None:
        if not self.enabled:
            log.info("remote link disabled (NTFY_TOPIC / MONITOR_TOKEN not set)")
            return
        h = _AlertHandler(self.alerts)
        logging.getLogger("infra").addHandler(h)
        logging.getLogger("guardian").addHandler(h)
        threading.Thread(target=self._loop, name="remote", daemon=True).start()
        log.info("remote link on (ntfy)")

    # ------------------------------------------------------------ pieces (testable)
    def compact_status(self) -> dict:
        st = self.status_fn()
        g = st.get("guardian") or {}
        pos = {k: v.get("position") for k, v in (st.get("symbols") or {}).items() if v.get("position")}
        return {"t": st.get("time_utc"), "eq": st.get("equity"), "day0": st.get("day_start_equity"),
                "n": st.get("trades_today"), "cl": st.get("consec_losses"),
                "stop": st.get("stop_file"), "flat": st.get("flatten_file"),
                "halt": g.get("halted"), "pause": g.get("paused"), "gp": g.get("global_pause_min"),
                "mkt": (st.get("market") or {}).get("open"),
                "px": {k: v.get("last_price") for k, v in (st.get("symbols") or {}).items()},
                "err": {k: v.get("errors") for k, v in (st.get("symbols") or {}).items()},
                "pos": pos, "ev": (g.get("last_events") or [])[-3:],
                "last": [{k: t.get(k) for k in ("symbol", "side", "reason", "pnl", "R", "exit_ts")}
                         for t in (st.get("recent_trades") or [])[-3:]],
                # per stream: auto-trading?, qualification, forward paper results [n, wins, sumR, pnl]
                "tr": [k for k, v in (st.get("symbols") or {}).items() if v.get("trading")],
                "q": {k: (v.get("qual") or {}).get("status") for k, v in (st.get("symbols") or {}).items()},
                "fw": {k: [f.get("n"), f.get("w"), f.get("sumR"), f.get("pnl")]
                       for k, v in (st.get("symbols") or {}).items() for f in [v.get("fw") or {}] if f.get("n")},
                "off": {k: v.get("auto_off") for k, v in (st.get("symbols") or {}).items() if v.get("auto_off")},
                "cr": {k: v for k, v in (st.get("credits") or {}).items() if k in ("used", "cap", "health")} or None}

    def handle(self, text: str) -> str:
        parts = (text or "").strip().split()
        if len(parts) != 2 or parts[1] != self.token:
            return ""                                              # wrong/missing token: ignore silently
        cmd = parts[0].lower().lstrip("/")
        if cmd == "ping":
            return "pong"
        if cmd == "restart":
            threading.Timer(2.0, self.restart_fn).start()
            return "restarting (will load the latest code from GitHub)"
        if cmd in ("stop", "start", "flatten", "halt", "resume", "status"):
            return self.command_fn("/" + cmd)
        if cmd == "diagnose":
            try:
                return self._build_diagnose_report()
            except Exception as e:
                return f"diagnose failed: {type(e).__name__}: {e}"
        return ""

    # ------------------------------------------------------------ diagnose (READ-ONLY)
    DIAG_MAX_CHARS = 3500
    DIAG_MAX_BYTES = 3900          # ntfy.sh message limit is 4096 bytes (Arabic = 2 bytes/char)
    # Patterns match the bot's REAL error lines (bot._on_err: "<SYM>: <kind> error #N -> <kind>: <msg>").
    # A Twelve Data 429 is logged as "error #N -> rate: <provider text>" - the text does not contain "429",
    # so a bare "429" only ever matched digits inside timestamps/prices.
    _DIAG_ERR = {"rate_limit_429": r"error #\d+ -> rate:", "credits": r"\bcredits\b", "quota": r"\bquota\b",
                 "Forbidden": r"\bForbidden\b", "Unauthorized": r"\bUnauthorized\b",
                 "auth": r"DATA API AUTH ERROR", "error #": r"\berror #\d+"}
    _DIAG_LATE = ("signal skipped", "too late")

    @staticmethod
    def _log_path() -> str:
        """Path of the file the bot really logs to (infra.setup_logging -> /data/bot.log in the container)."""
        from logging.handlers import RotatingFileHandler
        for h in logging.getLogger().handlers:
            if isinstance(h, RotatingFileHandler):
                return h.baseFilename
        return os.path.abspath("bot.log")

    def _build_diagnose_report(self) -> str:
        """Read-only snapshot for remote debugging. Never writes, never changes state, never shows secrets."""
        import hashlib
        import subprocess
        from datetime import datetime, timedelta, timezone
        na = "غير متاح: "
        out = [f"[DIAG {datetime.now(timezone.utc).isoformat(timespec='seconds')}]"]

        # 1) COMMIT
        code_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=code_dir, capture_output=True,
                               text=True, timeout=5)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or "not a git checkout").strip()[:80])
            out.append(f"1) COMMIT: {r.stdout.strip()}")
        except Exception as e:
            fp = []
            for f in ("bot.py", "remote.py", "providers.py", "strategy.py"):
                try:
                    with open(os.path.join(code_dir, f), "rb") as fh:
                        fp.append(f"{f}={hashlib.sha256(fh.read()).hexdigest()[:10]}")
                except OSError:
                    fp.append(f"{f}=?")
            out.append(f"1) COMMIT: {na}{type(e).__name__}: {e} | code={code_dir} | sha256 {' '.join(fp)}")

        # 2) MODE (values of non-secret settings; secrets only as present/absent)
        def env(k: str) -> str:
            v = os.getenv(k)
            return "غير موجود" if v is None else (v.strip() or "(فارغ)")
        sec = {k: ("موجود" if os.getenv(k) else "غير موجود")
               for k in ("TWELVE_DATA_API_KEY", "DASHBOARD_PASSWORD", "CTRADER_CLIENT_SECRET")}
        out.append(f"2) MODE: MODE={env('MODE')} PAPER_TRADING={env('PAPER_TRADING')} "
                   f"TRADING_MODE={env('TRADING_MODE')} EXECUTION={env('EXECUTION')} EVENT_BRAKE={env('EVENT_BRAKE')} "
                   f"| " + " ".join(f"{k}:{v}" for k, v in sec.items()))

        # 3) WATCHLIST
        try:
            st = self.status_fn() or {}
            syms = st.get("symbols") or {}
            if not syms:
                raise RuntimeError("status has no symbols")
            tr = [k for k, v in syms.items() if v.get("trading")]
            dia = any((v.get("ticker") or k).split("-")[0] == "DIA" for k, v in syms.items())
            out.append(f"3) WATCHLIST: {len(syms)} lines, DIA={'نعم' if dia else 'لا'}, tradable={','.join(tr) or '-'}")
            out.append(f"   order: {','.join(syms)}")
            q = st.get("bar_queue")
            out.append(f"   queue_order (bar requests, top of the hour): {','.join(q) if q else 'غير متاح: old bot.py'}")
            import watchlist as _wl
            src = os.path.abspath(_wl.WATCHLIST_FILE)
            out.append(f"   source: {'file ' + src if os.path.exists(src) else 'default list in watchlist.py (no file)'}")
        except Exception as e:
            st = {}
            out.append(f"3) WATCHLIST: {na}{type(e).__name__}: {e}")

        # 4) ERRORS + 5) LATE + 7) TAIL - one pass over the current log file
        path = self._log_path()
        import re
        pats = {k: re.compile(v) for k, v in self._DIAG_ERR.items()}
        lines, late, log_err, last_hb = [], [], "", ""
        counts = {k: 0 for k in pats}
        seen_at = {k: ["", ""] for k in pats}               # first / last timestamp per error type
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    ln = ln.rstrip("\n")
                    ts = ln[:19] if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", ln) else "?"
                    for k, rx in pats.items():
                        if rx.search(ln):
                            counts[k] += 1
                            seen_at[k][0] = seen_at[k][0] or ts
                            seen_at[k][1] = ts
                    if any(k in ln for k in self._DIAG_LATE):
                        late.append(ln)
                    if " bot: hb " in ln:                    # heartbeat: keep only the latest, shown in full
                        last_hb = ln
                        continue
                    lines.append(ln)
                    if len(lines) > 400:
                        del lines[:200]
        except Exception as e:
            log_err = f"{type(e).__name__}: {e}"
        if log_err:
            out.append(f"4) ERRORS: {na}{log_err}")
            out.append(f"5) LATE: {na}{log_err}")
        else:
            out.append(f"4) ERRORS ({path}): " + " ".join(f"{k}={v}" for k, v in counts.items()))
            out.extend(f"   {k}: first {seen_at[k][0]} | last {seen_at[k][1]}" for k in counts if counts[k])
            out.append("5) LATE: " + (f"{len(late)} total" if late else "0"))
            out.extend("   " + redact(x)[:200] for x in late[-10:])

        # 6) QUOTA
        cr = (st or {}).get("credits") or {}                 # providers.RateBudget.quota_status() via bot._credits
        nxt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        used, cap = cr.get("used", "غير متاح"), cr.get("cap", "-")
        out.append(f"6) QUOTA: provider={cr.get('provider') or 'twelvedata'}, count={used}, soft={cr.get('soft', '-')}, "
                   f"hard={cr.get('hard', '-')}, max={cap}, health={cr.get('health', '-')}, "
                   f"source={cr.get('source', 'memory')}, file={cr.get('file') or '-'}, "
                   f"reset_at={cr.get('reset_at') or nxt.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                   f"({used}/{cap}, journal={cr.get('journal') or '-'}"
                   + (f", error={cr['error']}" if cr.get("error") else "") + ")")

        # 7) TAIL - drop the OLDEST lines first until the message fits
        head = "\n".join(out)
        if log_err:
            return self._fit(head + f"\n7) TAIL: {na}{log_err}")
        tail = [redact(x)[:240] for x in lines[-40:]]
        hb = "\n8) LAST HB (full):\n" + (redact(last_hb) if last_hb else "غير متاح: no heartbeat line in the log")
        while True:
            msg = head + f"\n7) TAIL (last {len(tail)}, heartbeats excluded):\n" + "\n".join(tail) + hb
            if (len(msg) <= self.DIAG_MAX_CHARS and len(msg.encode("utf-8")) <= self.DIAG_MAX_BYTES) or not tail:
                return self._fit(msg)
            tail.pop(0)

    def _fit(self, msg: str) -> str:
        while len(msg) > self.DIAG_MAX_CHARS or len(msg.encode("utf-8")) > self.DIAG_MAX_BYTES:
            msg = msg[:-50]
        return msg

    # ------------------------------------------------------------ network
    def _pub(self, suffix: str, msg: str, title: str = "", prio: int = 3, tags: str = "") -> None:
        hdr = {"Priority": str(prio)}
        if title:
            hdr["Title"] = title.encode("utf-8").decode("latin-1", "ignore") or "bot"
        if tags:
            hdr["Tags"] = tags
        self.sess.post(f"{BASE}/{self.topic}-{suffix}", data=msg.encode("utf-8"), headers=hdr, timeout=(5, 10))

    def _poll_commands(self) -> None:
        r = self.sess.get(f"{BASE}/{self.topic}-c/json", params={"poll": "1", "since": "3m"}, timeout=(5, 15))
        for line in r.text.splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("event") != "message":
                continue
            mid = m.get("id")
            if mid in self.seen:
                continue                                       # ntfy may re-deliver: run each command once
            self.seen.append(mid)
            sent = float(m.get("time", 0))
            if time.time() - sent > 120 or sent < self.boot - 1:
                continue                                       # stale, or already handled by the previous process
            reply = self.handle(m.get("message", ""))
            if reply:
                cmd = m.get("message", "").split()[0]
                log.warning("remote command %s", cmd)
                self._pub("a", f"{cmd}: {reply}", title="command", prio=4, tags="robot")

    def _loop(self) -> None:
        last_status = 0.0
        while True:
            try:
                self._poll_commands()
                while self.alerts:
                    lvl, msg = self.alerts.popleft()
                    prio = 5 if lvl >= logging.CRITICAL else 4 if lvl >= logging.ERROR else 3
                    self._pub("a", msg, prio=prio, tags="warning" if lvl >= logging.WARNING else "chart_with_upwards_trend")
                if time.time() - last_status >= 60:
                    self._pub("s", json.dumps(self.compact_status(), default=str, ensure_ascii=False), prio=1)
                    last_status = time.time()
            except Exception as e:
                log.debug("remote link error: %s", redact(f"{type(e).__name__}: {e}"))
            time.sleep(10)
