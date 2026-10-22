"""Guardian tests:  python test_guardian.py"""
import os
import tempfile

from guardian import GuardCfg, Guardian


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def mk(**kw):
    d = tempfile.mkdtemp(prefix="g_")
    msgs = []
    clk = Clock()
    g = Guardian(GuardCfg(**kw), lambda m, lvl=0: msgs.append(m), clock=clk, halt_file=os.path.join(d, "HALT"))
    return g, clk, msgs


def test_drawdown_halt_and_manual_resume():
    g, clk, msgs = mk(max_drawdown_pct=5)
    g.on_equity(1000)
    g.on_trade("BTC/USD", -1, 970)
    assert g.entry_block("ETH/USD") is None
    g.on_trade("BTC/USD", -1, 949)                       # -5.1% from peak
    assert "halted" in (g.entry_block("ETH/USD") or "")
    assert os.path.exists(g.halt_file)
    os.remove(g.halt_file)                               # human resets
    assert g.entry_block("ETH/USD") is None
    print("  ok  drawdown from peak -> HALT, deleting HALT file resumes")


def test_strategy_decay_pause():
    g, clk, _ = mk(decay_window=5, decay_min_avg_R=-0.3, decay_pause_h=2, streak_losses=99, sym_daily_loss_R=99)
    for r in (-1, 0.5, -1, -1, 0.4):
        g.on_trade("SOL/USD", r, 1000)
        clk.t += 60
    assert "decay" in (g.entry_block("SOL/USD") or "")
    assert g.entry_block("BTC/USD") is None
    clk.t += 2 * 3600 + 1
    assert g.entry_block("SOL/USD") is None
    print("  ok  strategy decay pauses only that symbol, expires on time")


def test_shock_exits_position_against_move():
    g, clk, _ = mk(shock_atr=3, shock_window_sec=120)
    assert not g.on_price("BTC/USD", 100.0, 1.0, pos_side=1)
    clk.t += 10
    assert not g.on_price("BTC/USD", 99.0, 1.0, pos_side=1)
    clk.t += 10
    assert g.on_price("BTC/USD", 96.5, 1.0, pos_side=1)        # -3.5 ATR against a long
    assert "shock" in (g.entry_block("BTC/USD") or "")
    print("  ok  violent move against a position -> immediate exit + cool-down")


def test_shock_in_favour_does_not_exit():
    g, clk, _ = mk(shock_atr=3)
    g.on_price("ETH/USD", 100.0, 1.0, pos_side=1)
    clk.t += 5
    assert not g.on_price("ETH/USD", 104.0, 1.0, pos_side=1)   # +4 ATR in our favour
    print("  ok  spike in our favour does not force an exit")


def test_symbol_daily_loss_and_streak():
    g, clk, _ = mk(sym_daily_loss_R=2.5, streak_losses=3, streak_window_h=6, streak_cooldown_h=1, decay_window=99)
    g.on_trade("QQQ", -1.2, 1000); clk.t += 60
    g.on_trade("QQQ", -1.4, 990); clk.t += 60
    assert "daily loss" in (g.entry_block("QQQ") or "")
    g.on_trade("BTC/USD", -1, 980); clk.t += 60
    assert "cool-down" in (g.entry_block("ETH/USD") or "")
    print("  ok  per-symbol daily loss cap + account-wide loss-streak cool-down")


def test_persistence():
    g, clk, _ = mk(max_drawdown_pct=3)
    g.on_equity(1000); g.on_trade("BTC/USD", -2, 960)
    snap = g.snapshot()
    g2, _, _ = mk()
    g2.halt_file = g.halt_file
    g2.restore(snap)
    assert g2.entry_block("BTC/USD") is not None
    print("  ok  halt survives a restart (state.json)")


def test_manual_halt_file_and_validation():
    g, clk, _ = mk()
    open(g.halt_file, "w").write("telegram /halt\n")
    assert "telegram /halt" in (g.entry_block("QQQ") or "")
    os.remove(g.halt_file)
    assert g.entry_block("QQQ") is None
    try:
        GuardCfg(decay_window=0).validate()
        raise AssertionError("decay_window=0 accepted")
    except ValueError:
        pass
    print("  ok  hand-written HALT file halts; invalid GUARD_* values rejected at startup")


def test_first_trade_counts_for_drawdown():
    g, clk, _ = mk(max_drawdown_pct=5)
    g.on_equity(1000)                      # seeded at startup by the bot
    g.on_trade("QQQ", -3, 940)             # the very first trade: -6%
    assert g.entry_block("QQQ") is not None
    print("  ok  drawdown measured from the starting equity (first loss counts)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nall guardian tests passed")
