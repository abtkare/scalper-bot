"""ntfy publish budget (ntfy.sh anonymous: ~250 messages/day per IP):  python test_ntfy.py   (no network)"""
import json
import logging
import time

import remote
from remote import RemoteLink


class Resp:
    def __init__(self, code):
        self.status_code = code


def link():
    rl = RemoteLink("t" * 20, "k" * 20, lambda: {"equity": 1, "guardian": {}, "symbols": {}}, lambda c: "ok", lambda: None)
    rl.sent = []
    rl.code = 200
    rl.sess.post = lambda url, data, headers, timeout: (rl.sent.append((url[-1], data.decode())) or Resp(rl.code))
    return rl


# 1) 429 -> 15 min pause, the alert is kept (not lost), nothing counted
rl = link()
rl.code = 429
rl.alerts.append((logging.INFO, "ENTRY XAU/USD SHORT qty=1"))
rl._flush_alerts()
assert len(rl.alerts) == 1 and rl.sent_today == 0 and rl.block_until > time.time() + 890
n = len(rl.sent)
rl._flush_alerts()
assert len(rl.sent) == n, "published again during the pause"
rl.code, rl.block_until = 200, 0
rl._flush_alerts()
assert not rl.alerts and rl.sent_today == 1 and rl.sent[-1] == ("a", "ENTRY XAU/USD SHORT qty=1")
print("  ok  429: publishing paused 15 min, the alert stays queued and goes out afterwards")

# 2) daily tiers: status <= 60%, news <= 80%, trades/errors up to the cap
rl = link()
rl._allow("status")
rl.sent_today = int(remote.DAILY_CAP * 0.6)
assert not rl._allow("status") and rl._allow("info") and rl._allow("alert")
rl.alerts.extend([(logging.INFO, "📰 news"), (logging.INFO, "EXIT XAU/USD target")])
rl.sent_today = int(remote.DAILY_CAP * 0.8)
rl._flush_alerts()
assert [m for _, m in rl.sent] == ["EXIT XAU/USD target"] and rl.dropped_today == 1
rl.sent_today = remote.DAILY_CAP
rl.alerts.append((logging.ERROR, "EXIT REJECTED"))
rl._flush_alerts()
assert rl.dropped_today == 2 and len(rl.sent) == 1
rl.pub_day = "2000-01-01"                                     # next UTC day -> fresh budget
assert rl._allow("status") and rl.sent_today == 0
print(f"  ok  budget: status stops at {int(remote.DAILY_CAP * .6)}, news at {int(remote.DAILY_CAP * .8)}, "
      f"trades at {remote.DAILY_CAP}; resets at 00:00 UTC")

# 3) command replies jump the queue and keep the "command" title
rl = link()
rl.alerts.append((logging.INFO, "📰 old news"))


class R:
    def __init__(self, msgs):
        self.text = "\n".join(json.dumps(m) for m in msgs)


rl.sess.get = lambda *a, **k: R([{"event": "message", "id": "p1", "time": time.time(), "message": "ping " + "k" * 20}])
rl._poll_commands()
rl._flush_alerts()
assert rl.sent[0] == ("a", "ping: pong"), rl.sent
print("  ok  command replies are sent first")

# 4) a whole day fits: 144 status (every 10 min) + 60 alerts <= cap, with the 60-second interval it did not
assert remote.STATUS_EVERY == 600 and 86400 / remote.STATUS_EVERY + 60 <= remote.DAILY_CAP < 250
assert 86400 / 60 > 250
print("  ok  10-min status: 144/day + 60 alerts fit under 250 (60-s status = 1440/day did not)")
print("\nall ntfy tests passed")
