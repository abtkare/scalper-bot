"""Persistent Twelve Data quota (providers.QuotaStore + RateBudget(store=...)).
Run: python test_quota.py      (no network, no API key; temp SQLite files only)
"""
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

import bot
from config import Params, SymbolCfg
from providers import Binance, QuotaStore, RateBudget, TwelveData, make_providers, utc_day
from test_robust import T0, Env, Rec, ramp

logging.getLogger().setLevel(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))
PASSED = []
DAY = 86400.0


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def tmpdb():
    return os.path.join(tempfile.mkdtemp(prefix="quota_"), "quota.db")


class Clk:
    def __init__(self, t):
        self.now = float(t)

    def __call__(self):
        return self.now


def budget(path, clk, per_day=800, provider="twelvedata"):
    return RateBudget(1000, clock=clk, per_day=per_day, store=QuotaStore(path, provider, clock=clk))


# ------------------------------------------------------------------ 1) atomicity
def test_1_atomic_concurrent():
    path, day = tmpdb(), "2026-01-05"
    QuotaStore(path).close()
    stores = [QuotaStore(path) for _ in range(10)]          # 10 separate connections = worst case
    errs = []

    def worker(s):
        for _ in range(20):
            if s.add(day, 1) is None:
                errs.append(s.last_error)
    th = [threading.Thread(target=worker, args=(s,)) for s in stores]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs, errs
    assert QuotaStore(path).get(day) == 200, QuotaStore(path).get(day)
    clk = Clk(T0)
    b = budget(tmpdb(), clk)                                # one budget shared by 10 threads
    res = []
    th = [threading.Thread(target=lambda: res.append(b.try_acquire(1.0, "quote"))) for _ in range(10)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert res.count(True) == 10 and b.store.get(utc_day(T0)) == 10 == b.used_day
    assert b.store.journal == "wal", b.store.journal
    ok("atomic: 10 connections x 20 increments = 200, 10 threads on one budget = 10, journal=WAL")


# ------------------------------------------------------------------ 2) UTC midnight reset + same-day restart
def test_2_midnight_reset():
    path = tmpdb()
    QuotaStore(path).add(utc_day(T0 - DAY), 500)            # yesterday's row
    clk = Clk(T0)
    b = budget(path, clk)
    q = b.quota_status()
    assert q["used"] == 0 and q["day"] == utc_day(T0) and q["health"] == "ok", q
    assert q["reset_at"] == time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(T0 + DAY)), q
    for _ in range(5):
        assert b.try_acquire(1.0, "bars")
        clk.now += 1
    b2 = budget(path, clk)                                  # restart, same UTC day -> NOT zeroed
    assert b2.quota_status()["used"] == 5
    clk.now = (int(T0) // 86400 + 1) * 86400 - 1            # 23:59:59 UTC
    assert b2.quota_status()["used"] == 5
    clk.now += 2                                            # 00:00:01 UTC next day
    assert b2.quota_status()["used"] == 0 and b2.try_acquire(1.0, "scan") and b2.used_day == 1
    assert QuotaStore(path).get(utc_day(T0)) == 5           # yesterday kept (history), today starts at 0
    ok("midnight: yesterday's row ignored, restart keeps 5, 00:00 UTC starts from 0")


# ------------------------------------------------------------------ 3) limits
def test_3_limits():
    path, clk = tmpdb(), Clk(T0)
    QuotaStore(path).add(utc_day(T0), 799)
    b = budget(path, clk)
    assert not b.try_acquire(2.0, "quote") and b.store.get(utc_day(T0)) == 799   # 799+2 > 800: refused, not counted
    assert b.quota_status()["health"] == "critical"
    clk.now += 1
    assert b.try_acquire(1.0, "quote") and b.quota_status()["health"] == "blocked"
    clk.now += 1
    assert not b.try_acquire(1.0, "quote") and b.store.get(utc_day(T0)) == 800
    q = b.quota_status()
    assert (q["soft"], q["hard"], q["cap"], q["remaining"]) == (560, 740, 800, 0), q
    # tiers unchanged with the store: scan stops at 560, bars at 740 (same numbers as before)
    b3 = budget(tmpdb(), clk)
    n = {"scan": 0, "bars": 0, "quote": 0}
    for tier in n:
        while b3.try_acquire(1.0, tier):
            n[tier] += 1
            clk.now += 0.1
    assert n == {"scan": 560, "bars": 180, "quote": 60}, n
    hl = [(c, RateBudget(1, per_day=800).per_day) for c in (719, 720, 759, 760, 799, 800)]
    from providers import quota_health
    assert [quota_health(c, cap) for c, cap in hl] == ["ok", "warn", "warn", "critical", "critical", "blocked"]
    ok("limits: 799 + 2 refused, 800 = blocked, tiers still 560/740/800, health 720/760/800")


# ------------------------------------------------------------------ 4) migration from RAM
def test_4_migrate():
    path, day = tmpdb(), utc_day(T0)
    s = QuotaStore(path)
    mem = RateBudget(1000, clock=Clk(T0), per_day=800)       # old behaviour: RAM only
    for _ in range(100):
        mem.try_acquire(1.0, "quote")
        mem.clock.now += 1
    assert mem.used_day == 100 and s.get(day) == 0
    assert s.migrate_from_memory(day, mem.used_day) == 100 and s.get(day) == 100
    assert s.migrate_from_memory(day, 50) == 100            # never lowers a saved count
    assert budget(path, Clk(T0)).used_day == 100
    ok("migrate: RAM count 100 -> SQLite 100; a lower RAM value never lowers it")


# ------------------------------------------------------------------ 5) crash safety
CHILD = r"""
import sys; sys.path.insert(0, sys.argv[1])
from providers import QuotaStore
s = QuotaStore(sys.argv[2])
if sys.argv[3] == "open-tx":
    s.add("2026-01-05", 1)
    s.conn.execute("BEGIN IMMEDIATE")
    s.conn.execute("UPDATE api_quota SET count = count + 1000")
    print("IN-TX", flush=True)
    import time; time.sleep(60)
else:
    while True:
        print(s.add("2026-01-05", 1), flush=True)
"""


def test_5_crash_safety():
    path = tmpdb()
    p = subprocess.Popen([sys.executable, "-c", CHILD, HERE, path, "open-tx"], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "IN-TX"
    p.send_signal(signal.SIGKILL)                           # killed in the middle of a write
    p.wait()
    s = QuotaStore(path)
    assert s.get("2026-01-05") == 1, s.get("2026-01-05")    # the half-done +1000 is gone
    assert s.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    p = subprocess.Popen([sys.executable, "-c", CHILD, HERE, path, "loop"], stdout=subprocess.PIPE, text=True)
    last = None
    for _ in range(300):
        last = int(p.stdout.readline())
    p.send_signal(signal.SIGKILL)                           # killed while hammering commits
    p.wait()
    s2 = QuotaStore(path)
    got = s2.get("2026-01-05")
    assert last <= got <= last + 1, (last, got)             # every confirmed commit survived, nothing corrupt
    assert s2.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert s2.add("2026-01-05", 1) == got + 1
    ok(f"crash: SIGKILL mid-transaction -> rolled back; SIGKILL mid-loop -> {got} (>= {last} confirmed), integrity ok")


# ------------------------------------------------------------------ 6) integration
class _R:
    status_code, headers = 200, {}


def test_6_integration():
    path = tmpdb()
    cfg = SymbolCfg("QQQ", "stock", "America/New_York", provider="twelvedata")
    td = make_providers(Params(), [cfg], quota_path=path)["twelvedata"]
    vals = [{"datetime": "2026-01-05 10:00:00", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "9"}]
    sent = []
    td._request = lambda url, params: (sent.append(params) or _R(), {"values": vals, "price": "1.5"})
    clk = Clk(time.time())
    td.budget.clock, td.budget.t = clk, clk.now
    for _ in range(3):                                      # the bot's pattern: reserve, then call
        if td.budget.try_acquire(1.0, "bars"):
            td.bars(cfg, 5)
        clk.now += 10                                       # real plan rate (6/min) stays in force
    assert len(sent) == 3 and QuotaStore(path).get(td.budget.day) == 3
    assert not any("apikey" in r for r in QuotaStore(path).conn.execute("SELECT * FROM api_quota").fetchall())
    assert TwelveData(Params()).budget.store is None        # tests / no path -> RAM only, no file created
    # the real bot loop: every request it sends is counted in SQLite
    e = Env(ramp(T0 + 450, 0.1))
    e.mkt.budget = budget(tmpdb(), e.clock)
    assert e.run(900, until=lambda: bool(e.trades()))
    calls = e.mkt.bar_calls + e.mkt.quote_calls
    assert calls > 3 and e.mkt.budget.store.get(utc_day(T0)) == calls == e.mkt.budget.used_day
    ok(f"integration: 3 Twelve Data calls -> count=3; bot loop {calls} requests -> count={calls}; no key stored")


# ------------------------------------------------------------------ 7) blocked: no requests, open trade kept
def test_7_blocked_keeps_positions():
    path = tmpdb()
    e = Env(ramp(T0 + 450, 0.1))
    e.mkt.budget = budget(path, e.clock)
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    QuotaStore(path).add(utc_day(T0), 0, floor=800)         # quota exhausted (e.g. by an earlier process)
    e.mkt.budget = budget(path, e.clock)                    # restart -> loads 800 from SQLite
    calls = e.mkt.bar_calls + e.mkt.quote_calls
    e.run(120)
    assert e.mkt.bar_calls + e.mkt.quote_calls == calls, "requests sent while blocked"
    assert e.ss.eng.pos is not None and not e.trades(), "position closed because of the quota"
    assert e.mkt.budget.quota_status()["health"] == "blocked"
    assert Binance(Params()).budget is None                 # crypto streams never touch this counter
    e.rt._quota_alerts()
    e.rt._quota_alerts()
    assert sum("blocked: 800/800, no new requests until 00:00 UTC" in m for m in e.n.msgs) == 1, e.n.msgs
    e.clock.now = (int(T0) // 86400 + 1) * 86400 + 5        # 00:00 UTC -> requests resume, trade still managed
    assert e.run(900, until=lambda: bool(e.trades()))
    assert e.trades()[0]["reason"] == "target"
    ok("blocked: 0 requests, open position kept and managed after 00:00 UTC, Binance unaffected, 1 alert")


# ------------------------------------------------------------------ 8) 7-day cleanup
def test_8_cleanup():
    path = tmpdb()
    s = QuotaStore(path)
    for d in range(10):
        s.add(utc_day(T0 - d * DAY), d + 1)
        s.mark_alert(utc_day(T0 - d * DAY), "warn")
    budget(path, Clk(T0))                                   # day start runs the cleanup
    days = [r[0] for r in s.conn.execute("SELECT day_utc FROM api_quota ORDER BY day_utc").fetchall()]
    assert days == [utc_day(T0 - d * DAY) for d in range(6, -1, -1)], days
    assert s.conn.execute("SELECT COUNT(*) FROM api_quota_alerts").fetchone()[0] == 7
    ok(f"cleanup: 10 days -> 7 kept ({days[0]} .. {days[-1]})")


# ------------------------------------------------------------------ extra: degraded -> retry after 60 s
class _Broken:
    def execute(self, *a):
        raise sqlite3.OperationalError("disk I/O error")

    def close(self):
        pass


def test_9_degrade_and_retry():
    path, clk = tmpdb(), Clk(T0)
    b = budget(path, clk)
    assert b.try_acquire(1.0, "quote")
    b.store.conn = _Broken()                                # SQLite fails in the middle of work
    clk.now += 1
    assert b.try_acquire(1.0, "quote") and b.used_day == 2  # trading data keeps flowing (RAM)
    q = b.quota_status()
    assert q["source"] == "memory" and "disk I/O" in q["error"], q
    tries = []
    real = b.store._connect
    b.store._connect = lambda: tries.append(clk.now) or real()
    for _ in range(5):
        clk.now += 10
        assert b.try_acquire(1.0, "quote")
    assert tries == [] and b.used_day == 7                  # no reconnect storm inside the 60 s window
    clk.now += 15                                           # > 60 s after the failure
    assert b.try_acquire(1.0, "quote") and len(tries) == 1
    assert b.quota_status()["source"] == "sqlite" and QuotaStore(path).get(utc_day(T0)) == 8   # file caught up
    e = Env(ramp(T0 + 450, 0.1))
    e.mkt.budget = b
    b.store.conn = _Broken()
    b.store.add(utc_day(T0), 0)
    for _ in range(3):
        e.rt._quota_alerts()
    assert sum("degraded to memory" in m for m in e.n.msgs) == 1, e.n.msgs
    ok("degraded: RAM keeps counting, one retry after 60 s, file catches up, one ntfy only")


# ------------------------------------------------------------------ extra: alerts once per day, across restarts
def test_10_alerts_once_across_restarts():
    path, clk = tmpdb(), Clk(T0)
    QuotaStore(path).add(utc_day(T0), 0, floor=725)
    msgs = []
    for _ in range(3):                                      # 3 restarts on the same day
        e = Env(ramp(T0 + 450, 0.1))
        e.mkt.budget = budget(path, e.clock)
        e.rt._quota_alerts()
        msgs += e.n.msgs
    assert sum("quota warning: 725/800 (watch-only scanning already paused at 560)" in m for m in msgs) == 1, msgs
    assert not any("near limit" in m for m in msgs)
    QuotaStore(path).add(utc_day(T0), 0, floor=765)
    e.mkt.budget = budget(path, e.clock)
    e.rt._quota_alerts()
    assert sum("near limit: 765/800" in m for m in e.n.msgs) == 1 and not any("quota warning" in m for m in e.n.msgs)
    ok("alerts: 720 warning sent once in 3 restarts, 760 sent once, no repeats")


# ------------------------------------------------------------------ extra: diagnose line
def test_11_diagnose_line():
    from remote import RemoteLink
    path, clk = tmpdb(), Clk(time.time())
    QuotaStore(path).add(utc_day(clk.now), 0, floor=312)
    b = budget(path, clk)
    st = {"credits": dict(b.quota_status(), provider="twelvedata"), "symbols": {"QQQ": {"ticker": "QQQ", "trading": True}},
          "bar_queue": ["QQQ"]}
    rl = RemoteLink("t" * 20, "k" * 20, lambda: st, lambda c: "ok", lambda: None)
    rep = rl.handle("diagnose " + "k" * 20)
    line = next(x for x in rep.splitlines() if x.startswith("6) QUOTA"))
    want = (f"6) QUOTA: provider=twelvedata, count=312, soft=560, hard=740, max=800, health=ok, source=sqlite, "
            f"file={path}, reset_at={time.strftime('%Y-%m-%dT00:00:00Z', time.gmtime(clk.now + DAY))}")
    assert line.startswith(want), line
    assert len(line) < 300 and "apikey" not in rep
    assert rl.compact_status()["cr"] == {"used": 312, "cap": 800, "health": "ok"}
    ok("diagnose: " + line[:110] + " ...")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items(), key=lambda kv: int(kv[0].split("_")[1]) if kv[0].startswith("test_") else 0)
             if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nall {len(PASSED)} quota tests passed")
