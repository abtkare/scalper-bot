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
print("\nall remote tests passed")
