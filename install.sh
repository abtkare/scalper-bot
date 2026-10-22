#!/usr/bin/env bash
# One-command installer for Ubuntu/Debian VPS (Hostinger KVM etc.). Run as root:
#   curl -fsSL https://raw.githubusercontent.com/abtkare/scalper-bot/main/install.sh | TD_KEY=xxxx bash
# Optional: DASH_PW=...  (dashboard password; generated if empty)   BRANCH=main
# Safe to run again: it updates the code and keeps state/trades/.env.
set -euo pipefail

REPO="${REPO:-https://github.com/abtkare/scalper-bot.git}"
BRANCH="${BRANCH:-main}"
APP_USER="scalper"
APP_DIR="/opt/scalper-bot"
PORT_INTERNAL=8088

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
[ "$(id -u)" = 0 ] || { echo "Run as root (use: sudo -i)"; exit 1; }
command -v apt-get >/dev/null || { echo "This installer supports Ubuntu/Debian only."; exit 1; }

say "1/8 System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv python3-pip curl ca-certificates chrony >/dev/null
systemctl enable --now chrony >/dev/null 2>&1 || true

say "2/8 Service user + code"
id "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "/home/$APP_USER" --shell /usr/sbin/nologin "$APP_USER"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch -q origin "$BRANCH" && git -C "$APP_DIR" reset -q --hard "origin/$BRANCH"
else
  git clone -q --branch "$BRANCH" "$REPO" "$APP_DIR"
fi
git config --global --add safe.directory "$APP_DIR" || true

say "3/8 Python environment"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
"$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

say "4/8 Self-tests"
( cd "$APP_DIR" && "$APP_DIR/venv/bin/python" test_engine.py && "$APP_DIR/venv/bin/python" test_guardian.py | tail -1 )

say "5/8 Configuration (.env)"
ENV="$APP_DIR/.env"
touch "$ENV"
setkv() {  # setkv KEY VALUE  -> replace or append
  if grep -q "^$1=" "$ENV"; then sed -i "s|^$1=.*|$1=$2|" "$ENV"; else echo "$1=$2" >> "$ENV"; fi
}
getkv() { grep "^$1=" "$ENV" | head -1 | cut -d= -f2- || true; }
[ -n "${TD_KEY:-}" ] && setkv TWELVE_DATA_API_KEY "$TD_KEY"
[ -n "$(getkv TWELVE_DATA_API_KEY)" ] || { echo "TD_KEY is required the first time (TD_KEY=xxxx bash)"; exit 1; }
DASH_PW="${DASH_PW:-$(getkv DASHBOARD_PASSWORD)}"; DASH_PW="${DASH_PW:-$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 14)}"
MON="$(getkv MONITOR_TOKEN)"; MON="${MON:-$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)}"
setkv MODE paper
setkv DASHBOARD_PASSWORD "$DASH_PW"
setkv MONITOR_TOKEN "$MON"
setkv PORT "$PORT_INTERNAL"
setkv QUOTE_POLL_SEC "$(getkv QUOTE_POLL_SEC | grep . || echo 20)"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$ENV"

say "6/8 Web access (HTTPS)"
IP="$(curl -fsS4 --max-time 8 https://api.ipify.org || hostname -I | awk '{print $1}')"
HOST="scalper-${IP//./-}.sslip.io"
busy() { ss -ltnH "( sport = :$1 )" 2>/dev/null | grep -q . ; }
URL=""
if command -v caddy >/dev/null || { ! busy 80 && ! busy 443; }; then
  setkv DASHBOARD_HOST 127.0.0.1
  if ! command -v caddy >/dev/null; then
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https gnupg >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq && apt-get install -y -qq caddy >/dev/null
  fi
  mkdir -p /etc/caddy
  touch /etc/caddy/Caddyfile
  # drop the default ":80" welcome site on a fresh install, keep any site the owner already has
  if grep -q '^:80' /etc/caddy/Caddyfile && ! grep -q 'reverse_proxy\|root \*' /etc/caddy/Caddyfile; then : > /etc/caddy/Caddyfile; fi
  sed -i "/# scalper-bot begin/,/# scalper-bot end/d" /etc/caddy/Caddyfile
  cat >> /etc/caddy/Caddyfile <<EOF
# scalper-bot begin
$HOST {
	reverse_proxy 127.0.0.1:$PORT_INTERNAL
}
# scalper-bot end
EOF
  systemctl enable caddy >/dev/null 2>&1; systemctl reload caddy 2>/dev/null || systemctl restart caddy
  URL="https://$HOST"
else
  warn "Ports 80/443 are used by another web server (nginx/apache). Dashboard will be on plain HTTP port $PORT_INTERNAL."
  setkv DASHBOARD_HOST 0.0.0.0
  URL="http://$IP:$PORT_INTERNAL"
fi
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
  [[ "$URL" == http://* ]] && ufw allow "$PORT_INTERNAL/tcp" >/dev/null
fi

say "7/8 Bot service (auto-restart, watchdog)"
cat > /etc/systemd/system/scalper.service <<EOF
[Unit]
Description=Scalper paper-trading bot
After=network-online.target time-sync.target
Wants=network-online.target time-sync.target
StartLimitIntervalSec=0

[Service]
Type=notify
NotifyAccess=main
WatchdogSec=60
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python -u bot.py
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5
TimeoutStopSec=20
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR
MemoryMax=600M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

say "8/8 Auto-update from GitHub (every 10 min, only when there is a new commit)"
cat > /usr/local/bin/scalper-update <<EOF
#!/usr/bin/env bash
set -e
cd $APP_DIR
old=\$(git rev-parse HEAD)
git fetch -q origin $BRANCH
new=\$(git rev-parse origin/$BRANCH)
[ "\$old" = "\$new" ] && exit 0
git reset -q --hard origin/$BRANCH
$APP_DIR/venv/bin/pip install -q -r requirements.txt
if $APP_DIR/venv/bin/python test_engine.py >/dev/null 2>&1; then
  chown -R $APP_USER:$APP_USER $APP_DIR
  systemctl restart scalper
  echo "updated \$old -> \$new"
else
  git reset -q --hard "\$old"; chown -R $APP_USER:$APP_USER $APP_DIR
  echo "new commit \$new FAILED tests - kept \$old"
fi
EOF
chmod +x /usr/local/bin/scalper-update
cat > /etc/systemd/system/scalper-update.service <<EOF
[Unit]
Description=Update scalper bot from GitHub
[Service]
Type=oneshot
ExecStart=/usr/local/bin/scalper-update
EOF
cat > /etc/systemd/system/scalper-update.timer <<EOF
[Unit]
Description=Check GitHub for scalper bot updates
[Timer]
OnBootSec=3min
OnUnitActiveSec=10min
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now scalper >/dev/null
systemctl restart scalper
systemctl enable --now scalper-update.timer >/dev/null

sleep 8
ok="NOT RUNNING"
systemctl is-active --quiet scalper && ok="RUNNING"
H="$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT_INTERNAL/healthz" || echo 'no answer')"

cat <<EOF

=====================================================================
 Bot status : $ok   (health: $H)
 Dashboard  : $URL
 Login      : any user name  +  password: $DASH_PW
 Logs       : journalctl -u scalper -f
---------------------------------------------------------------------
 SEND THIS LINE TO CLAUDE (for monitoring):
 MONITOR $URL $MON
=====================================================================
EOF
