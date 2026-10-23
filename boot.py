"""Bootstrap for a plain python:3.12-slim container (no image build needed - works in any Docker UI).

compose:  image: python:3.12-slim
          command: python -c "import urllib.request as u;exec(u.urlopen('https://raw.githubusercontent.com/abtkare/scalper-bot/main/boot.py',timeout=30).read())"

On every (re)start: download the latest code from GitHub -> install requirements -> run the self-test ->
start the bot in /data (persistent volume). If GitHub is unreachable or the new code fails its test, the last
good copy saved in /data/app_good is used. A remote "restart" command therefore = update.
"""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

TGZ = os.getenv("REPO_TGZ", "https://codeload.github.com/abtkare/scalper-bot/tar.gz/refs/heads/main")
DATA, NEW, GOOD = "/data", "/tmp/app_new", "/data/app_good"
os.makedirs(DATA, exist_ok=True)


def say(m):
    print("boot:", m, flush=True)


def pip_ok(code_dir):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "--root-user-action=ignore", "--cache-dir", f"{DATA}/.pipcache",
                        "-r", os.path.join(code_dir, "requirements.txt")])
    return r.returncode == 0


def test_ok(code_dir):
    return subprocess.run([sys.executable, "test_engine.py"], cwd=code_dir,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


code = None
try:
    shutil.rmtree(NEW, ignore_errors=True)
    data = urllib.request.urlopen(TGZ, timeout=30).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(NEW, filter="data")
    top = os.path.join(NEW, os.listdir(NEW)[0])
    if pip_ok(top) and test_ok(top):
        shutil.rmtree(GOOD, ignore_errors=True)
        shutil.copytree(top, GOOD, ignore=shutil.ignore_patterns("data", "research", "*.csv"))
        code = GOOD
        say("running the latest code from GitHub")
    else:
        say("latest code failed install/self-test")
except Exception as e:
    say(f"could not fetch the latest code ({type(e).__name__}: {e})")

if code is None:
    if os.path.exists(os.path.join(GOOD, "bot.py")) and pip_ok(GOOD):
        code = GOOD
        say("using the last good copy")
    else:
        say("no usable code yet - retrying in 30 s")
        import time
        time.sleep(30)
        sys.exit(1)                       # Docker restarts the container -> try again

os.chdir(DATA)
os.execv(sys.executable, [sys.executable, "-u", os.path.join(code, "bot.py")])
