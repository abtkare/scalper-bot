"""Web dashboard + control API (runs in a background thread inside the bot process).

  GET  /                 dashboard page (HTTP Basic auth: any user name + DASHBOARD_PASSWORD)
  GET  /api/status       JSON status          (Basic auth, or ?token=MONITOR_TOKEN)
  POST /api/control      {"cmd": "stop|start|flatten|halt|resume|restart|mode_all|mode_fx|mode_qqq|mode_scalp|mode_both"}
  GET  /api/control?cmd=flatten&token=MONITOR_TOKEN                  (for the remote monitor)
  GET  /healthz          200 while the trade loop is alive, 503 if it stalled (> 90 s) - no auth

The dashboard can only do what the STOP / FLATTEN / HALT files do. It can never open a trade.
"""
from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("dashboard")
COMMANDS = ("stop", "start", "flatten", "halt", "resume", "restart",
            "mode_all", "mode_fx", "mode_qqq", "mode_scalp", "mode_both")


class LogTail(logging.Handler):
    """Keeps the last N log lines in memory for the dashboard."""

    def __init__(self, n: int = 150):
        super().__init__(level=logging.INFO)
        self.lines: deque = deque(maxlen=n)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%m-%d %H:%M:%S"))

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass


class Dashboard:
    def __init__(self, status_fn: Callable[[], dict], command_fn: Callable[[str], str],
                 alive_fn: Callable[[], float], password: str, token: str, tail: Optional[LogTail] = None,
                 port: int = 8000, restart_fn: Optional[Callable[[], None]] = None):
        self.status_fn, self.command_fn, self.alive_fn = status_fn, command_fn, alive_fn
        self.password, self.token, self.tail, self.port = password, token, tail, port
        self.started = time.time()
        self.restart_fn = restart_fn

    # ------------------------------------------------------------ auth
    def _basic_ok(self, header: Optional[str]) -> bool:
        if not self.password or not header or not header.startswith("Basic "):
            return False
        try:
            raw = base64.b64decode(header[6:]).decode("utf-8", "replace")
        except Exception:
            return False
        pw = raw.split(":", 1)[1] if ":" in raw else raw
        return hmac.compare_digest(pw.encode(), self.password.encode())

    def _token_ok(self, q: dict) -> bool:
        t = (q.get("token") or [""])[0]
        return bool(self.token) and hmac.compare_digest(t.encode(), self.token.encode())

    # ------------------------------------------------------------ payloads
    def status(self) -> dict:
        st = self.status_fn()
        st["loop_age_s"] = round(time.time() - self.alive_fn(), 1)
        st["uptime_min"] = round((time.time() - self.started) / 60)
        if self.tail is not None:
            st["log"] = list(self.tail.lines)[-60:]
        return st

    def start(self) -> None:
        dash = self

        class H(BaseHTTPRequestHandler):
            server_version = "scalper"

            def log_message(self, *a):                  # keep request noise out of bot.log
                return

            def _send(self, code: int, body, ctype="application/json; charset=utf-8", extra=None):
                data = body if isinstance(body, bytes) else (
                    json.dumps(body, ensure_ascii=False, default=str).encode() if not isinstance(body, str)
                    else body.encode())
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _deny(self):
                self._send(401, {"error": "auth required"},
                           extra={"WWW-Authenticate": 'Basic realm="scalper-bot", charset="UTF-8"'})

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/healthz":
                    age = time.time() - dash.alive_fn()
                    return self._send(200 if age < 90 else 503, {"ok": age < 90, "loop_age_s": round(age, 1)})
                basic = dash._basic_ok(self.headers.get("Authorization"))
                tok = dash._token_ok(q)
                if u.path == "/api/status":
                    if not (basic or tok):
                        return self._deny()
                    return self._send(200, dash.status())
                if u.path == "/api/control":
                    if not tok:                          # GET control only with the monitor token
                        return self._deny()
                    c = (q.get("cmd") or [""])[0]
                    if c.strip().lower().startswith("mode_"):    # mode changes only from the owner's dashboard
                        return self._send(403, {"error": "mode change needs the dashboard password"})
                    return self._control(c, "monitor")
                if u.path in ("/", "/index.html"):
                    if not basic:
                        return self._deny()
                    return self._send(200, PAGE, "text/html; charset=utf-8")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                u = urlparse(self.path)
                if u.path != "/api/control":
                    return self._send(404, {"error": "not found"})
                if not dash._basic_ok(self.headers.get("Authorization")):
                    return self._deny()
                try:
                    n = min(int(self.headers.get("Content-Length") or 0), 1000)
                    cmd = json.loads(self.rfile.read(n) or b"{}").get("cmd", "")
                except Exception:
                    cmd = ""
                return self._control(cmd, "dashboard")

            def _control(self, cmd: str, who: str):
                cmd = (cmd or "").strip().lower()
                if cmd not in COMMANDS:
                    return self._send(400, {"error": f"cmd must be one of {COMMANDS}"})
                if cmd == "restart":
                    if dash.restart_fn is None:
                        return self._send(400, {"error": "restart not available"})
                    threading.Timer(2.0, dash.restart_fn).start()
                    msg = "جارٍ التحديث: البوت يحفظ حالته ويعيد التشغيل بأحدث كود من GitHub خلال دقيقة."
                elif cmd.startswith("mode_"):
                    msg = dash.command_fn("/mode " + cmd[5:])
                else:
                    msg = dash.command_fn("/" + cmd)
                log.warning("control %s from %s", cmd, who)
                return self._send(200, {"ok": True, "cmd": cmd, "message": msg})

        srv = ThreadingHTTPServer((os.getenv("DASHBOARD_HOST", "0.0.0.0"), self.port), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, name="dashboard", daemon=True).start()
        log.info("dashboard listening on port %d", self.port)


PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>لوحة البوت</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--tx:#e8eaf0;--mut:#8a93a6;--g:#2fbf71;--r:#ef4e4e;--y:#e7b416;--b:#4c8dff}
@media (prefers-color-scheme: light){:root{--bg:#f4f5f8;--card:#fff;--line:#e3e6ec;--tx:#141821;--mut:#5d6677}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif}
.w{max-width:900px;margin:0 auto;padding:16px}
h1{font-size:19px;margin:4px 0 2px}.mut{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:14px 0}
@media(min-width:640px){.grid{grid-template-columns:repeat(4,1fr)}}
.c{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.k{color:var(--mut);font-size:12px}.v{font-size:21px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.pos{color:var(--g)}.neg{color:var(--r)}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:600;margin:2px 0 2px 6px}
.ok{background:rgba(47,191,113,.15);color:var(--g)}.warn{background:rgba(231,180,22,.16);color:var(--y)}.bad{background:rgba(239,78,78,.15);color:var(--r)}
.btns{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:12px 0}
@media(min-width:640px){.btns{grid-template-columns:repeat(4,1fr)}}
button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:13px 8px;color:#fff;cursor:pointer}
.bs{background:#5b6475}.bu{background:var(--b)}.bf{background:var(--r)}.bh{background:#9b3df0}.br{background:var(--g)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:7px 4px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--mut);font-weight:500}
.scroll{overflow-x:auto}
pre{white-space:pre-wrap;word-break:break-word;font-size:11.5px;max-height:320px;overflow:auto;margin:0;color:var(--mut)}
h2{font-size:15px;margin:18px 0 8px}#msg{min-height:20px;font-size:13px}
</style></head><body><div class="w">
<h1>بوت التداول — Paper</h1>
<div class="mut" id="sub">جارٍ التحميل…</div>
<div id="flags" style="margin-top:8px"></div>
<div class="grid">
 <div class="c"><div class="k">الرصيد</div><div class="v" id="eq">–</div></div>
 <div class="c"><div class="k">ربح اليوم</div><div class="v" id="day">–</div></div>
 <div class="c"><div class="k">صفقات اليوم</div><div class="v" id="nt">–</div></div>
 <div class="c"><div class="k">خسائر متتالية</div><div class="v" id="cl">–</div></div>
</div>
<div class="c"><div class="k" style="margin-bottom:6px">التحكم الطارئ</div>
 <div class="btns">
  <button class="bs" onclick="cmd('stop','إيقاف الدخول الجديد؟')">⏸ إيقاف الدخول</button>
  <button class="bf" onclick="cmd('flatten','إغلاق كل الصفقات الآن؟')">⛔ إغلاق الكل</button>
  <button class="bh" onclick="cmd('halt','تجميد كامل؟')">🧊 تجميد</button>
  <button class="br" onclick="cmd('resume','استئناف التداول؟')">▶ استئناف</button>
 </div>
 <button class="bu" style="width:100%;margin-top:2px" onclick="cmd('restart','تحديث البوت بأحدث كود من GitHub؟ (الحالة والصفقات تبقى محفوظة)')">🔄 تحديث البوت من GitHub</button>
 <div id="msg" class="mut"></div></div>
<h2>ماذا يتداول البوت؟</h2><div class="c"><div class="k">الوضع الحالي: <b id="mode">–</b> · التغيير مرفوض أثناء وجود صفقة مفتوحة، ويعيد تشغيل البوت تلقائياً</div>
 <div class="btns">
  <button class="bs" id="m_all" onclick="cmd('mode_all','تداول الكل: QQQ + الذهب + EURUSD + GBPUSD + USDJPY؟')">🌍 الكل</button>
  <button class="bs" id="m_fx" onclick="cmd('mode_fx','الذهب والعملات فقط (بدون QQQ)؟')">🥇 ذهب وعملات</button>
  <button class="bs" id="m_qqq" onclick="cmd('mode_qqq','QQQ فقط (اختراق)؟')">📈 QQQ فقط</button>
  <button class="bs" id="m_scalp" onclick="cmd('mode_scalp','سكالبينج QQQ فقط؟ (تجريبي - نسبة ربح ~75% وربح صغير لكل صفقة)')">⚡ سكالبينج QQQ</button>
 </div></div>
<h2>الرموز والصفقات المفتوحة</h2><div class="c scroll"><table id="syms"></table></div>
<h2>الحارس</h2><div class="c" id="guard">–</div>
<h2>آخر الصفقات</h2><div class="c scroll"><table id="trades"></table></div>
<h2>السجل</h2><div class="c"><pre id="log" dir="ltr" style="text-align:left"></pre></div>
<p class="mut">تتحدث تلقائياً كل 5 ثوانٍ. الأزرار تقلل المخاطرة فقط ولا تفتح أي صفقة.</p>
</div>
<script>
const $=id=>document.getElementById(id);
const f=(x,d=2)=>x==null||isNaN(x)?'–':Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){
 try{
  const r=await fetch('api/status',{cache:'no-store'}); if(!r.ok) throw new Error(r.status);
  const s=await r.json();
  $('sub').textContent='آخر تحديث '+new Date().toLocaleTimeString('ar-IQ')+' · عمر الحلقة '+s.loop_age_s+' ث · تشغيل '+s.uptime_min+' د';
  $('eq').textContent=f(s.equity);
  const d=s.equity-s.day_start_equity; $('day').textContent=(d>=0?'+':'')+f(d); $('day').className='v '+(d>=0?'pos':'neg');
  $('nt').textContent=s.trades_today; $('cl').textContent=s.consec_losses;
  const g=s.guardian||{}; let fl=[];
  if(s.loop_age_s>60) fl.push('<span class="badge bad">الحلقة متوقفة!</span>');
  if(g.halted) fl.push('<span class="badge bad">مجمّد: '+esc(g.halted)+'</span>');
  if(s.flatten_file) fl.push('<span class="badge bad">FLATTEN</span>');
  if(s.stop_file) fl.push('<span class="badge warn">الدخول موقوف</span>');
  if(g.global_pause_min) fl.push('<span class="badge warn">تهدئة '+g.global_pause_min+' د</span>');
  if(!fl.length) fl.push('<span class="badge ok">يعمل طبيعياً</span>');
  if(s.market) fl.push('<span class="badge '+(s.market.open?'ok':'warn')+'">'+esc(s.market.text)+'</span>');
  if(s.fx_market) fl.push('<span class="badge '+(s.fx_market.open?'ok':'warn')+'">'+esc(s.fx_market.text)+'</span>');
  $('flags').innerHTML=fl.join('');
  const MN={all:'الكل: QQQ + ذهب + عملات',fx:'الذهب والعملات',qqq:'QQQ فقط',scalp:'سكالبينج QQQ',both:'QQQ اختراق + سكالبينج'};
  $('mode').textContent=MN[s.mode]||s.mode||'–';
  for(const m of ['all','fx','qqq','scalp']){const b=$('m_'+m); if(b) b.className=(s.mode==m?'bu':'bs');}
  let h='<tr><th>الرمز</th><th>الاستراتيجية</th><th>الحالة</th><th>السعر</th><th>دخول</th><th>وقف</th><th>هدف</th><th>غير محقق</th><th>عمر السعر</th></tr>';
  for(const [k,v] of Object.entries(s.symbols||{})){const p=v.position;
   h+='<tr><td>'+esc(k)+'</td><td>'+esc({breakout:'اختراق',scalp:'سكالبينج',pullback:'سحب',rsi2:'ارتداد RSI2'}[v.strategy]||v.strategy||'')+'</td><td>'+(p?(p.side=='long'?'شراء ':'بيع ')+p.qty:'بلا صفقة')+'</td><td>'+f(v.last_price)+'</td><td>'+(p?f(p.entry):'–')+'</td><td>'+(p?f(p.stop):'–')+'</td><td>'+(p?f(p.target):'–')+'</td><td class="'+(p&&p.unrealized<0?'neg':'pos')+'">'+(p?f(p.unrealized):'–')+'</td><td>'+(v.quote_age_s==null?'–':v.quote_age_s+' ث')+'</td></tr>';}
  $('syms').innerHTML=h;
  let gh=[]; if(g.peak) gh.push('قمة الرصيد: '+f(g.peak));
  for(const [k,v] of Object.entries(g.paused||{})) gh.push('موقوف '+esc(k)+' لمدة '+v+' د');
  for(const [k,v] of Object.entries(g.recent_avg_R||{})) gh.push('متوسط آخر الصفقات '+esc(k)+': '+v+'R');
  for(const e of (g.last_events||[])) gh.push('• '+esc(e));
  $('guard').innerHTML=gh.length?gh.join('<br>'):'لا تدخلات حتى الآن';
  let t='<tr><th>الرمز</th><th>الاتجاه</th><th>السبب</th><th>الدخول</th><th>الخروج</th><th>الربح</th><th>R</th><th>وقت الخروج</th></tr>';
  for(const x of (s.recent_trades||[]).slice().reverse()){const p=parseFloat(x.pnl);
   t+='<tr><td>'+esc(x.symbol)+'</td><td>'+esc(x.side)+'</td><td>'+esc(x.reason)+'</td><td>'+f(+x.entry)+'</td><td>'+f(+x.exit)+'</td><td class="'+(p<0?'neg':'pos')+'">'+f(p)+'</td><td>'+f(+x.R)+'</td><td>'+esc(String(x.exit_ts).slice(5,16))+'</td></tr>';}
  $('trades').innerHTML=(s.recent_trades||[]).length?t:'<tr><td>لا صفقات مغلقة بعد</td></tr>';
  $('log').textContent=(s.log||[]).slice().reverse().join('\n');
 }catch(e){$('sub').textContent='تعذّر الاتصال بالبوت ('+e.message+')';}
}
async function cmd(c,q){ if(!confirm(q)) return;
 const r=await fetch('api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cmd:c})});
 const j=await r.json().catch(()=>({})); $('msg').textContent=j.message||j.error||r.status; load(); }
load(); setInterval(load,5000);
</script></body></html>"""
