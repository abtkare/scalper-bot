"""Central configuration. Edit values here, or override via .env / environment.

Every numeric field of `Params` can be overridden with an environment variable of the
same name in UPPER_CASE (e.g. RISK_PCT=0.003, TD_CREDITS_PER_MIN=55).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from typing import List, Tuple
from zoneinfo import ZoneInfo


class ConfigError(ValueError):
    """Raised at startup for an invalid configuration (the bot refuses to run)."""


def load_env(path: str = ".env") -> None:
    """Tiny .env loader (no extra dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Params:
    # --- account / risk ---
    initial_equity: float = 1000.0
    risk_pct: float = 0.005          # risk per trade as fraction of equity (0.5%)
    max_leverage: float = 5.0        # cap on notional / equity
    daily_loss_cap: float = 0.03     # stop trading for the day after -3%
    max_consec_losses: int = 3       # stop for the day after N losses in a row
    max_trades_per_day: int = 10     # per account, per day

    # --- strategy (5-minute bars) ---
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_min: float = 40.0            # long needs RSI in [rsi_min, rsi_max]
    rsi_max: float = 65.0            # shorts use the mirrored band
    atr_len: int = 14
    stop_atr: float = 1.0            # stop distance = stop_atr * ATR
    target_atr: float = 1.5          # target distance = target_atr * ATR
    # Skip setups whose target is not at least N x the round-trip cost.
    min_target_cost_multiple: float = 3.0
    # v11: adverse slippage (bps) added to every MARKET fill in the backtest and the qualification gate,
    # on top of cost_bps. Live fills record their real slippage (trades_live.csv slip_in_bps / slip_out_bps).
    bt_slip_bps: float = 0.25

    # --- speed (live loop) ---
    loop_sec: float = 0.2            # main loop period; exits are evaluated on every quote
    bar_settle_sec: float = 1.0      # wait this long after a bar closes before trusting it
    bar_retry_sec: float = 1.5       # re-poll interval while waiting for the new bar
    quote_poll_sec: float = 2.0      # live price poll while a position is open
    entry_poll_sec: float = 0.5      # live price poll while an entry is pending
    max_entry_delay_sec: float = 20.0  # skip the entry if we cannot fill within N s of the bar open
    max_entry_slip_atr: float = 0.5  # skip if the fill is > N*ATR away from the signal bar close

    # --- safety ---
    max_quote_age_sec: float = 30.0  # position open + no fresh price for N s -> alert + wick backstop
    max_tick_jump_pct: float = 3.0   # a single tick moving more than this must be confirmed
    max_error_streak: int = 8        # N consecutive data errors -> block NEW entries
    td_credits_per_min: int = 8      # your Twelve Data plan's credits/minute (free plan = 8)
    td_credits_per_day: int = 800    # daily cap of your plan (free = 800; 0 = no daily cap)
    http_connect_timeout: float = 3.0
    http_read_timeout: float = 5.0


@dataclass
class SymbolCfg:
    symbol: str                      # bot id = Twelve Data symbol unless provider_symbol is set ("QQQ", "XAU/USD")
    kind: str                        # "stock" | "crypto" | "fx" (spot gold/silver + currency pairs)
    tz: str                          # timezone of the bar timestamps
    interval: str = "5min"
    interval_minutes: int = 5
    cost_bps: float = 1.0            # per SIDE: commission + spread + slippage (ASSUMPTION - replace with real numbers)
    allow_short: bool = False
    flatten_eod: bool = False        # close everything at the end of the session
    eod_time: str = "15:55"          # bar-start time at/after which we flatten (exchange local)
    windows: List[Tuple[str, str]] = field(default_factory=list)  # entry windows, exchange local; empty = always
    provider: str = "twelvedata"     # bars (+ quotes) source: "twelvedata" | "binance"
    quote_provider: str = ""         # optional different source for live prices ("" = same as provider)
    provider_symbol: str = ""        # override symbol name at the provider (binance: BTCUSDT)
    poll_sec: float = 0.0            # per-symbol live-price poll interval (0 = Params.quote_poll_sec)
    strategy: str = "pullback"       # pullback | breakout (Donchian+trend) | scalp (VWAP reversion) | rsi2 (RSI(2) reversion)
    donchian_n: int = 24             # breakout: close above the highest high of the previous N bars
    vwap_k: float = 1.0              # scalp: enter when close is more than k*ATR away from the session VWAP
    history_bars: int = 300          # bars kept/downloaded for indicators (scalp needs ~600 for EMA200)
    htf_mult: int = 0                # >0: skip entries AGAINST the higher-timeframe market structure (HTF = interval*N)
    rsi2_th: float = 10.0            # rsi2: long when RSI(2) < th (above EMA200), short when RSI(2) > 100-th (below)
    exit_only: bool = False          # set automatically for a leftover position of a mode that is switched off
    stop_atr: float = 0.0            # per-symbol overrides (0 = use Params.stop_atr / target_atr)
    target_atr: float = 0.0
    # v11 exits from market structure (RESEARCH.md section 15). trail="swing": after each CLOSED bar the stop
    # moves behind the last confirmed swing low (long) / high (short), never wider, once trail_after_r R ahead.
    trail: str = ""
    trail_after_r: float = 0.5
    trail_buf_atr: float = 0.1
    trail_no_target: bool = False    # True: no fixed target, the trailing stop takes the profit
    swing_k: int = 3                 # bars on each side that confirm a swing high/low

    @property
    def file_stub(self) -> str:
        return self.symbol.replace("/", "-")

    @property
    def quote_src(self) -> str:
        return self.quote_provider or self.provider

    @property
    def metal(self) -> bool:
        return self.kind == "fx" and self.ticker.split("/")[0] in ("XAU", "XAG", "XPT", "XPD")

    def usd_per_quote(self, price: float) -> float:
        """P&L of FX/metals is in the QUOTE currency. XXX/USD -> 1;  USD/JPY -> 1/price (JPY -> USD)."""
        if self.kind == "fx" and self.ticker.startswith("USD/") and price > 0:
            return 1.0 / price
        return 1.0

    @property
    def ticker(self) -> str:
        """Real market symbol (the `symbol` field is the bot's unique id, e.g. "QQQ-SCALP")."""
        return self.provider_symbol or self.symbol


# v3 research (1 year of real data, see RESEARCH.md):
#   * the original 5-min pullback LOST money on every market after costs (QQQ -15%, BTC -39%/yr)
#   * ~25,000 configurations tested; the only candidate that survived in- and out-of-sample checks and a
#     random-entry placebo is a 15-min breakout on QQQ (long only). It FAILED on SPY/IWM/DIA and breaks
#     even at 2 bps cost -> treat it as an EXPERIMENT to be forward-tested in paper mode, not as a proven edge.
#   * BTC/ETH/SOL: no configuration profitable at realistic crypto costs -> not traded (kept below, commented).
BREAKOUT_QQQ = SymbolCfg(
    symbol="QQQ", kind="stock", tz="America/New_York",
    interval="15min", interval_minutes=15, strategy="breakout", donchian_n=24, htf_mult=4,
    stop_atr=0.75, target_atr=2.0,
    cost_bps=1.0, flatten_eod=True, eod_time="15:45",
    windows=[("09:45", "15:30")],
)

# v4 scalping mode (RESEARCH.md sections 7-8): 15-min VWAP mean reversion on QQQ, long AND short.
#   Long : close > EMA200 and close < session VWAP - 1.0*ATR  -> buy next bar open
#   Short: close < EMA200 and close > session VWAP + 1.0*ATR  -> sell next bar open
#   Small target (0.75 ATR) + wide stop (2 ATR) -> high win rate, small edge per trade.
#   backtest.py --mode scalp, 1 year QQQ: 193 trades, win rate 75.1%, +0.061R/trade, PF 1.35, max DD -3.0%
#   (in-sample 75.0% / +0.041R, out-of-sample 75.3% / +0.091R).
#   It LOST on SPY / IWM / DIA with the same rules -> EXPERIMENT, paper only.
# Its id is "QQQ-SCALP" (provider_symbol="QQQ") so it can run side by side with the breakout on QQQ.
SCALP_QQQ = SymbolCfg(
    symbol="QQQ-SCALP", provider_symbol="QQQ", kind="stock", tz="America/New_York",
    interval="15min", interval_minutes=15, strategy="scalp", vwap_k=1.0, history_bars=600,
    stop_atr=2.0, target_atr=0.75, allow_short=True,
    cost_bps=1.0, flatten_eod=True, eod_time="15:45",
    windows=[("09:45", "15:30")],
)

# v5 (RESEARCH.md section 9): gold + major FX, 1-hour RSI(2) mean reversion in the New York morning.
#   Long : close > EMA200 and RSI(2) < 10   |   Short: close < EMA200 and RSI(2) > 90
#   Entries only on bars opening 08:00-12:00 New York (15:00-19:00 Baghdad), stop 1 ATR, target 1.5 ATR,
#   everything closed at 16:45 New York (never held overnight / over the weekend).
#   7 months of real 15-min data (Mar-Oct 2026), 4 instruments pooled: 169 trades, win rate 52.7%,
#   +0.18R/trade, PF 1.42, max DD -6.5%; random-entry placebo -0.095R. Only 7 months -> EXPERIMENT (paper).
#   cost_bps = assumed per-side cost of an ECN / cTrader account (spread + commission). Replace with yours.
def _fx(sym: str, cost: float) -> SymbolCfg:
    return SymbolCfg(
        symbol=sym, kind="fx", tz="America/New_York", interval="1h", interval_minutes=60,
        strategy="rsi2", rsi2_th=10.0, history_bars=600, stop_atr=1.0, target_atr=1.5, allow_short=True,
        cost_bps=cost, flatten_eod=True, eod_time="16:45", windows=[("08:00", "12:00")], poll_sec=30.0,
    )


GOLD = replace(_fx("XAU/USD", 0.5), trail="swing", trail_after_r=0.5, trail_no_target=True)   # v11
EURUSD = replace(_fx("EUR/USD", 0.5), trail="swing", trail_after_r=0.5)                         # v11
GBPUSD = _fx("GBP/USD", 0.6)
USDJPY = _fx("USD/JPY", 0.5)
FX_SET = [GOLD, EURUSD, GBPUSD, USDJPY]

# all   : QQQ breakout + gold + 3 FX pairs (default)
# fx    : gold + 3 FX pairs only
# qqq   : QQQ breakout only ("breakout" is the old name of this mode)
# scalp : QQQ VWAP scalping only (v4 experiment)    both : QQQ breakout + QQQ scalping
TRADING_MODES = ("all", "fx", "qqq", "scalp", "both")
_MODE_ALIASES = {"breakout": "qqq"}
MODE_FILE = "trading_mode.txt"      # written by /mode (Telegram / dashboard); wins over TRADING_MODE env


def read_trading_mode(path: str = MODE_FILE) -> str:
    """all (default) | fx | qqq | scalp | both.  File (set from the dashboard) > env TRADING_MODE > default."""
    m = ""
    try:
        with open(path, encoding="utf-8") as f:
            m = f.read().strip().lower()
    except OSError:
        pass
    m = _MODE_ALIASES.get(m, m)
    if m not in TRADING_MODES:
        m = os.getenv("TRADING_MODE", "all").strip().lower()
        m = _MODE_ALIASES.get(m, m)
    if m not in TRADING_MODES:
        raise ConfigError(f"TRADING_MODE must be one of {TRADING_MODES}, got {m!r}")
    return m


def symbols_for_mode(mode: str) -> List[SymbolCfg]:
    mode = _MODE_ALIASES.get(mode, mode)
    if mode == "all":
        return [BREAKOUT_QQQ] + FX_SET
    if mode == "fx":
        return list(FX_SET)
    if mode == "qqq":
        return [BREAKOUT_QQQ]
    if mode == "scalp":
        return [SCALP_QQQ]
    if mode == "both":
        return [BREAKOUT_QQQ, SCALP_QQQ]
    raise ConfigError(f"unknown trading mode {mode!r}")


# Default list (used by backtest.py --all and the tests). The live bot uses symbols_for_mode(read_trading_mode()).
SYMBOLS: List[SymbolCfg] = [BREAKOUT_QQQ, SCALP_QQQ] + FX_SET
# Original v2 setting - lost 39% in the 1-year backtest:
# SymbolCfg(symbol="BTC/USD", kind="crypto", tz="UTC", cost_bps=6.0)

PROVIDERS = ("twelvedata", "binance")


def _env_key(symbol: str) -> str:
    return re.sub(r"\W", "_", symbol).upper()


def apply_env_overrides(symbols: List[SymbolCfg]) -> None:
    """PROVIDER_BTC_USD=binance  /  QUOTE_PROVIDER_BTC_USD=binance  /  POLL_SEC_BTC_USD=1"""
    for s in symbols:
        k = _env_key(s.symbol)
        s.provider = os.getenv(f"PROVIDER_{k}", s.provider).strip().lower()
        s.quote_provider = os.getenv(f"QUOTE_PROVIDER_{k}", s.quote_provider).strip().lower()
        s.poll_sec = float(os.getenv(f"POLL_SEC_{k}", s.poll_sec) or 0.0)


def get_params() -> Params:
    p = Params()
    for f in fields(p):
        raw = os.getenv(f.name.upper())
        if raw is None or raw.strip() == "":
            continue
        cur = getattr(p, f.name)
        try:
            setattr(p, f.name, int(float(raw)) if isinstance(cur, int) else float(raw))
        except ValueError:
            raise ConfigError(f"{f.name.upper()}={raw!r} is not a valid number")
    return p


_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def validate_config(p: Params, symbols: List[SymbolCfg]) -> None:
    """Refuse to start on values that would silently cause losses or crashes later."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    need(p.initial_equity > 0, "INITIAL_EQUITY must be > 0")
    need(0 < p.risk_pct <= 0.05, "RISK_PCT must be in (0, 0.05]")
    need(0 < p.max_leverage <= 20, "MAX_LEVERAGE must be in (0, 20]")
    need(0 < p.daily_loss_cap <= 0.2, "DAILY_LOSS_CAP must be in (0, 0.2]")
    need(p.max_consec_losses >= 1 and p.max_trades_per_day >= 1, "daily limits must be >= 1")
    need(p.stop_atr > 0 and p.target_atr > 0, "stop_atr / target_atr must be > 0")
    need(0.05 <= p.loop_sec <= 2.0, "LOOP_SEC must be in [0.05, 2]")
    need(p.bar_settle_sec >= 0 and p.bar_retry_sec >= 0.2, "bar_settle_sec >= 0 and bar_retry_sec >= 0.2")
    need(p.quote_poll_sec >= 0.2 and p.entry_poll_sec >= 0.2, "poll intervals must be >= 0.2 s")
    need(p.max_entry_delay_sec > 0 and p.max_entry_slip_atr > 0, "entry guards must be > 0")
    need(p.max_tick_jump_pct > 0, "MAX_TICK_JUMP_PCT must be > 0")
    need(p.td_credits_per_min >= 1, "TD_CREDITS_PER_MIN must be >= 1")
    need(p.http_connect_timeout > 0 and p.http_read_timeout > 0, "HTTP timeouts must be > 0")
    need(len(symbols) > 0, "SYMBOLS is empty")

    seen = set()
    for s in symbols:
        need(s.symbol not in seen, f"duplicate symbol {s.symbol}")
        seen.add(s.symbol)
        need(s.kind in ("stock", "crypto", "fx"), f"{s.symbol}: kind must be 'stock', 'crypto' or 'fx'")
        try:
            ZoneInfo(s.tz)
        except Exception:
            raise ConfigError(f"{s.symbol}: unknown timezone {s.tz!r}")
        need(s.interval_minutes >= 1, f"{s.symbol}: interval_minutes must be >= 1")
        need(s.cost_bps >= 0, f"{s.symbol}: cost_bps must be >= 0")
        need(s.provider in PROVIDERS, f"{s.symbol}: provider must be one of {PROVIDERS}")
        need(s.quote_src in PROVIDERS, f"{s.symbol}: quote_provider must be one of {PROVIDERS}")
        need(s.poll_sec >= 0, f"{s.symbol}: poll_sec must be >= 0")
        need(s.strategy in ("pullback", "breakout", "scalp", "rsi2"),
             f"{s.symbol}: strategy must be 'pullback', 'breakout', 'scalp' or 'rsi2'")
        need(0 < s.rsi2_th < 50, f"{s.symbol}: rsi2_th must be in (0, 50)")
        if s.kind == "fx":
            need(s.tz == "America/New_York", f"{s.symbol}: fx symbols need tz='America/New_York' (trading hours)")
        need(s.vwap_k > 0, f"{s.symbol}: vwap_k must be > 0")
        need(150 <= s.history_bars <= 5000, f"{s.symbol}: history_bars must be in [150, 5000]")
        need(s.donchian_n >= 2, f"{s.symbol}: donchian_n must be >= 2")
        need(s.stop_atr >= 0 and s.target_atr >= 0, f"{s.symbol}: stop_atr/target_atr must be >= 0")
        need(s.trail in ("", "swing"), f"{s.symbol}: trail must be '' or 'swing'")
        need(2 <= s.swing_k <= 6 and s.trail_after_r >= 0 and s.trail_buf_atr >= 0, f"{s.symbol}: bad trail settings")
        need(_HHMM.match(s.eod_time) is not None, f"{s.symbol}: eod_time must be HH:MM")
        for a, b in s.windows:
            need(_HHMM.match(a) and _HHMM.match(b) and a <= b, f"{s.symbol}: bad window {(a, b)}")
        if "binance" in (s.provider, s.quote_src):
            need(s.kind == "crypto" and s.tz == "UTC", f"{s.symbol}: binance is only for crypto with tz='UTC'")
        if s.kind == "stock" and s.flatten_eod:
            need(s.tz == "America/New_York", f"{s.symbol}: the built-in NYSE calendar needs tz='America/New_York'")
    if any("twelvedata" in (s.provider, s.quote_src) for s in symbols):
        need(bool(os.getenv("TWELVE_DATA_API_KEY")), "TWELVE_DATA_API_KEY is not set (see .env.example)")
