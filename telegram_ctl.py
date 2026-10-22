"""Telegram remote control (from your phone). Runs in a background thread; never blocks the trade loop.

Only messages from TELEGRAM_CHAT_ID are obeyed - anyone else is ignored.

  /status   equity, open positions, guardian state, last trades
  /stop     no new entries (creates STOP)            /start   remove STOP
  /flatten  close everything now + no entries         /resume  remove STOP, FLATTEN and HALT
  /halt     same as the guardian's HALT (manual reset needed)
  /trades   last 10 closed trades
  /help
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Optional

import requests

from data import redact

log = logging.getLogger("telegram")

HELP = ("/status  الحالة والصفقات المفتوحة\n"
        "/stop  إيقاف الدخول الجديد\n/start  إلغاء الإيقاف\n"
        "/flatten  إغلاق كل الصفقات فوراً + منع الدخول\n"
        "/resume  استئناف كامل (يحذف STOP و FLATTEN و HALT)\n"
        "/halt  تجميد طارئ (يحتاج /resume)\n/trades  آخر الصفقات")


class TelegramControl:
    def __init__(self, status_fn: Callable[[], dict], files: dict, token: Optional[str] = None,
                 chat: Optional[str] = None):
        self.token = token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat = str(chat if chat is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.status_fn, self.files = status_fn, files
        self.owner = str(os.getenv("TELEGRAM_OWNER_ID", "") or "").strip()   # optional: your user id (groups)
        self.offset = 0
        self.started = time.time()
        self.stopping = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat)

    def start(self) -> None:
        if self.enabled:
            threading.Thread(target=self._loop, name="tg-control", daemon=True).start()

    # ------------------------------------------------------------ commands (pure: testable)
    def _touch(self, key: str, text: str) -> None:
        with open(self.files[key], "w", encoding="utf-8") as f:
            f.write(text + "\n")

    def _rm(self, key: str) -> None:
        try:
            os.remove(self.files[key])
        except FileNotFoundError:
            pass

    def handle(self, text: str) -> str:
        cmd = (text or "").strip().split()[0].split("@")[0].lower() if text and text.strip() else ""
        if cmd == "/stop":
            self._touch("stop", "telegram /stop")
            return "تم: لا دخول جديد. الصفقات المفتوحة ما زالت مُدارة."
        if cmd == "/start":
            self._rm("stop")
            return "تم: الدخول الجديد مسموح."
        if cmd == "/flatten":
            self._touch("flatten", "telegram /flatten")
            return "تم: إغلاق كل الصفقات عند أول سعر + منع الدخول. أرسل /resume للاستئناف."
        if cmd == "/halt":
            self._touch("halt", "telegram /halt\nDelete this file to resume trading.")
            self._touch("stop", "telegram /halt")
            return "تم التجميد. أرسل /resume للاستئناف."
        if cmd == "/resume":
            for k in ("stop", "flatten", "halt"):
                self._rm(k)
            return "تم الاستئناف الكامل."
        if cmd in ("/status", "/trades"):
            try:
                st = self.status_fn()
            except Exception as e:
                return f"تعذّر قراءة الحالة: {type(e).__name__}"
            return self._fmt_trades(st) if cmd == "/trades" else self._fmt_status(st)
        return HELP

    @staticmethod
    def _fmt_status(st: dict) -> str:
        g = st.get("guardian") or {}
        lines = [f"الرصيد: {st.get('equity')}  (بداية اليوم {st.get('day_start_equity')})",
                 f"صفقات اليوم: {st.get('trades_today')}  خسائر متتالية: {st.get('consec_losses')}"]
        flags = [n for n, on in (("STOP", st.get("stop_file")), ("FLATTEN", st.get("flatten_file"))) if on]
        if g.get("halted"):
            flags.append(f"HALT: {g['halted']}")
        if g.get("global_pause_min"):
            flags.append(f"تهدئة عامة {g['global_pause_min']} د")
        if flags:
            lines.append("⚠️ " + " | ".join(flags))
        for sym, s in (st.get("symbols") or {}).items():
            p = s.get("position")
            px = s.get("last_price")
            if p:
                lines.append(f"• {sym} {p['side']} {p['qty']} @ {p['entry']:.4f}  وقف {p['stop']:.4f}  "
                             f"هدف {p['target']:.4f}  غير محقق {p.get('unrealized')}")
            else:
                lines.append(f"• {sym} بلا صفقة" + (f"  السعر {px:.4f}" if px else ""))
        if g.get("paused"):
            lines.append("موقوفة مؤقتاً: " + ", ".join(f"{k} {v}د" for k, v in g["paused"].items()))
        return "\n".join(lines)

    @staticmethod
    def _fmt_trades(st: dict) -> str:
        tr = st.get("recent_trades") or []
        if not tr:
            return "لا صفقات مغلقة بعد."
        out = []
        for t in tr[-10:]:
            try:
                out.append(f"{t['symbol']} {t['side']} {t['reason']}  {float(t['pnl']):+.2f}  ({float(t['R']):+.2f}R)")
            except Exception:
                out.append(json.dumps(t)[:120])
        return "\n".join(out)

    # ------------------------------------------------------------ network
    def _api(self, method: str, **params):
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", data=params, timeout=(5, 40))
        return r.json()

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                js = self._api("getUpdates", offset=self.offset, timeout=30, allowed_updates='["message"]')
                for u in js.get("result", []):
                    self.offset = max(self.offset, int(u["update_id"]) + 1)
                    msg = u.get("message") or {}
                    if str((msg.get("chat") or {}).get("id", "")) != self.chat:
                        continue                                   # not the owner -> ignore silently
                    if self.owner and str((msg.get("from") or {}).get("id", "")) != self.owner:
                        continue                                   # group chat: only the owner may command
                    if float(msg.get("date", 0)) < self.started - 5:
                        continue                                   # sent while the bot was down -> never replay
                    try:
                        reply = self.handle(msg.get("text", ""))
                    except Exception as e:
                        reply = f"فشل تنفيذ الأمر: {type(e).__name__}: {e}"
                    log.warning("telegram command %r", msg.get("text", "")[:40])
                    self._api("sendMessage", chat_id=self.chat, text=reply[:3500])
            except Exception as e:
                log.warning("telegram control error: %s", redact(f"{type(e).__name__}: {e}"))
                time.sleep(10)
