"""Remote link tests (no network):  python test_remote.py"""
import json, time
import remote
from remote import RemoteLink

calls = []
rl = RemoteLink("t" * 20, "k" * 20, lambda: {"equity": 1, "guardian": {}, "symbols": {}}, lambda c: calls.append(c) or "ok",
                lambda: calls.append("RESTART"))
pubs = []
rl._pub = lambda *a, **k: pubs.append(a)


class R:
    def __init__(self, msgs): self.text = "\n".join(json.dumps(m) for m in msgs)


now = time.time()
msgs = [{"event": "message", "id": "a1", "time": now, "message": "stop " + "k" * 20},
        {"event": "message", "id": "a2", "time": now, "message": "flatten WRONG"},
        {"event": "message", "id": "a3", "time": now - 600, "message": "halt " + "k" * 20},
        {"event": "keepalive", "id": "x"}]
rl.sess.get = lambda *a, **k: R(msgs)
rl._poll_commands(); rl._poll_commands()            # second poll re-delivers the same messages
assert calls == ["/stop"], calls
print("  ok  each command runs exactly once; wrong token and stale commands ignored")
assert rl.handle("restart " + "k" * 20).startswith("restarting"); time.sleep(2.2)
assert "RESTART" in calls
print("  ok  restart command triggers a clean stop (container restarts with the latest code)")
c = rl.compact_status(); assert c["eq"] == 1 and len(json.dumps(c)) < 4000
rl2 = RemoteLink("t" * 20, "k" * 20, lambda: {}, lambda c: calls.append("AFTER-RESTART:" + c) or "ok", lambda: None)
rl2._pub = lambda *a, **k: None
rl2.boot = now + 5                                  # process started AFTER the command was sent
rl2.sess.get = lambda *a, **k: R([{"event": "message", "id": "r1", "time": now, "message": "restart " + "k" * 20},
                                  {"event": "message", "id": "r2", "time": now, "message": "stop " + "k" * 20}])
rl2._poll_commands()
assert not any(c.startswith("AFTER-RESTART") for c in calls), calls
print("  ok  a restarted bot never re-runs commands sent before it started (no restart loop)")

# diagnose: read-only report from the real log file, secrets redacted, fits one ntfy message
import logging, os, tempfile
from logging.handlers import RotatingFileHandler
d = tempfile.mkdtemp()
lp = os.path.join(d, "bot.log")
with open(lp, "w", encoding="utf-8") as f:
    for i in range(300):
        f.write(f"2026-10-23 04:19:39,429 INFO bot: hb equity=1000.00 line {i} " + "x" * 300 + "\n")
        f.write(f"2026-10-23 04:19:40,111 INFO bot: other line {i} price=1429.5\n")
    f.write("2026-10-23 05:00:01,000 WARNING infra: XAU/USD: signal skipped - too late (31s after the bar open)\n")
    f.write("2026-10-23 05:00:02,000 WARNING infra: QQQ: bar error #1 -> rate: You have run out of API credits apikey=SECRETKEY123456\n")
    f.write("2026-10-23 05:01:00,429 INFO bot: hb equity=1000.00 LAST " + "y" * 600 + "\n")
fh = RotatingFileHandler(lp); logging.getLogger().addHandler(fh)
before = open(lp, encoding="utf-8").read()
os.environ["TWELVE_DATA_API_KEY"] = "SECRETKEY123456"
st = {"credits": {"used": 8, "cap": 800}, "symbols": {"QQQ": {"ticker": "QQQ", "trading": True},
                                                       "DIA-SCALP": {"ticker": "DIA", "trading": False}},
      "bar_queue": ["XAU/USD", "DIA-SCALP"]}
rl3 = RemoteLink("t" * 20, "k" * 20, lambda: st, lambda c: "ok", lambda: calls.append("RESTART3"))
rep = rl3.handle("diagnose " + "k" * 20)
logging.getLogger().removeHandler(fh); fh.close()
assert rep.startswith("[DIAG "), rep[:80]
assert "SECRETKEY123456" not in rep, "secret leaked"
assert "DIA=نعم" in rep and "8/800" in rep and "too late" in rep, rep
assert "rate_limit_429=1" in rep and "error #=1" in rep, rep          # timestamps/prices with 429 are NOT counted
assert "rate_limit_429: first 2026-10-23 05:00:02 | last 2026-10-23 05:00:02" in rep, rep
assert "heartbeats excluded" in rep and "line 299 x" not in rep, rep  # old heartbeats left out of the tail
assert rep.rstrip().endswith("y" * 600), rep[-200:]                   # latest heartbeat shown in full, at the end
assert "reset_at=" in rep and "source: " in rep, rep
assert "queue_order (bar requests, top of the hour): XAU/USD,DIA-SCALP" in rep, rep
assert len(rep) <= 3500 and len(("diagnose: " + rep).encode("utf-8")) <= 4096
assert open(lp, encoding="utf-8").read() == before and "RESTART3" not in calls
assert rl3.handle("diagnose WRONG") == ""
print("  ok  diagnose: read-only, no secrets, fits one ntfy message, wrong token ignored")
print("\nall remote tests passed")
