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
                         for t in (st.get("recent_trades") or [])[-3:]]}

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
        return ""

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
