#!/bin/sh
# Use the newest code from GitHub if it downloads and passes the self-test; otherwise the baked-in copy.
REPO_TGZ="${REPO_TGZ:-https://codeload.github.com/abtkare/scalper-bot/tar.gz/refs/heads/main}"
CODE=/app
if [ "${AUTO_UPDATE:-1}" = "1" ]; then
  rm -rf /tmp/new && mkdir -p /tmp/new
  if python - "$REPO_TGZ" <<'PY'
import io, sys, tarfile, urllib.request
data = urllib.request.urlopen(sys.argv[1], timeout=20).read()
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    t.extractall("/tmp/new", filter="data")
PY
  then
    NEW=$(find /tmp/new -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -n "$NEW" ] && pip install -q -r "$NEW/requirements.txt" && (cd "$NEW" && python test_engine.py >/dev/null 2>&1); then
      CODE="$NEW"; echo "entrypoint: running latest code from GitHub"
    else
      echo "entrypoint: latest code failed its self-test - using the built-in copy"
    fi
  else
    echo "entrypoint: GitHub not reachable - using the built-in copy"
  fi
fi
cd /data
exec python -u "$CODE/bot.py"
