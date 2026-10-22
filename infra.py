"""Plumbing that keeps the bot alive and honest: crash-safe state, single-instance lock,
non-blocking alerts, systemd watchdog, logging."""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

import requests

from data import redact

log = logging.getLogger("infra")


# ------------------------------------------------------------------ logging
def setup_logging(path: str = "bot.log") -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------------------------------------------------------------ state
class StateStore:
    """Atomic + fsynced JSON with a rolling .bak. A crash or power loss in the middle of a save can
    never leave a half-written file; a corrupt file is quarantined and the backup is used."""

    def __init__(self, path: str = "state.json"):
        self.path = path
        self.bak = path + ".bak"
        self.notes: list = []            # human-readable recovery notes (shown at startup)

    def save(self, obj: dict) -> None:
        tmp = self.path + ".tmp"
        data = json.dumps(obj, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(self.path):
            try:
                shutil.copyfile(self.path, self.bak)
            except OSError as e:
                log.warning("could not refresh state backup: %s", e)
        os.replace(tmp, self.path)
        try:                                                     # persist the rename itself (POSIX)
            fd = os.open(os.path.dirname(os.path.abspath(self.path)), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def load(self) -> Optional[dict]:
        for p in (self.path, self.bak):
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict) or "equity" not in obj:
                    raise ValueError("missing 'equity'")
                if p == self.bak:
                    self.notes.append(f"{self.path} was unusable - restored from backup")
                return obj
            except Exception as e:
                q = f"{p}.corrupt-{int(time.time())}"
                try:
                    os.replace(p, q)
                except OSError:
                    q = "(could not rename)"
                self.notes.append(f"{p} unreadable ({type(e).__name__}); moved to {q}")
        return None


# ------------------------------------------------------------------ single instance
class SingleInstance:
    """Two bot processes on the same folder would double every trade. The OS releases this lock
    automatically if the process dies."""

    def __init__(self, path: str = "bot.lock"):
        self.path = path
        self.fh = None

    def acquire(self) -> bool:
        self.fh = open(self.path, "a+")
        try:
            try:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:                                  # Windows
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return True


# ------------------------------------------------------------------ systemd watchdog
def sd_notify(msg: str) -> None:
    """Minimal sd_notify (READY=1 / WATCHDOG=1). No-op when not started by systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        if addr.startswith("@"):
            addr = "\0" + addr[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
    except Exception:
        pass


# ------------------------------------------------------------------ alerts
class Notifier:
    """Telegram alerts sent from a background thread: a slow/blocked Telegram can never delay an exit.
    Repeated messages with the same `key` are rate-limited."""

    def __init__(self, token: Optional[str] = None, chat: Optional[str] = None):
        self.token = token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat = chat if chat is not None else os.getenv("TELEGRAM_CHAT_ID")
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=200)
        self.last: dict = {}
        self.sent = 0
        if self.token and self.chat:
            threading.Thread(target=self._worker, name="notifier", daemon=True).start()

    def send(self, msg: str, key: Optional[str] = None, every: float = 0.0, level: int = logging.INFO) -> None:
        now = time.time()
        if key and every and now - self.last.get(key, 0.0) < every:
            return
        if key:
            self.last[key] = now
        msg = redact(msg)
        log.log(level, msg)
        if self.token and self.chat:
            try:
                self.q.put_nowait(msg)
            except queue.Full:
                pass

    def _worker(self) -> None:
        while True:
            msg = self.q.get()
            for attempt in range(3):
                try:
                    r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                      data={"chat_id": self.chat, "text": msg[:3500]}, timeout=(3, 8))
                    if r.status_code < 500 and r.status_code != 429:
                        break
                except Exception as e:
                    log.warning("telegram failed: %s", redact(f"{type(e).__name__}: {e}"))
                time.sleep(2 * (attempt + 1))
            self.sent += 1
