"""Web dashboard + control API (runs in a background thread inside the bot process).

  GET  /                 dashboard page (HTTP Basic auth: any user name + DASHBOARD_PASSWORD)
  GET  /api/status       JSON status          (Basic auth, or ?token=MONITOR_TOKEN)
  POST /api/control      {"cmd": "stop|start|flatten|halt|resume|restart"}               (Basic auth)
  POST /api/watchlist    {"action": "add|remove|on|off", "symbol", "class", "strategies", "id"}  (Basic auth)
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
COMMANDS = ("stop", "start", "flatten", "halt", "resume", "restart")
WATCH_ACTIONS = ("add", "remove", "on", "off")


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
                 port: int = 8000, restart_fn: Optional[Callable[[], None]] = None,
                 watch_fn: Optional[Callable[[str, dict], str]] = None,
                 feeds: Optional[dict] = None):
        self.watch_fn = watch_fn
        self.feeds = feeds or {}                     # extra read-only JSON endpoints (ticker, news)
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
                if u.path in dash.feeds:
                    if not (basic or tok):
                        return self._deny()
                    try:
                        return self._send(200, dash.feeds[u.path]())
                    except Exception as e:
                        return self._send(500, {"error": type(e).__name__})
                if u.path == "/api/control":
                    if not tok:                          # GET control only with the monitor token
                        return self._deny()
                    return self._control((q.get("cmd") or [""])[0], "monitor")
                if u.path in ("/", "/index.html"):
                    if not basic:
                        return self._deny()
                    return self._send(200, PAGE, "text/html; charset=utf-8")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                u = urlparse(self.path)
                if u.path not in ("/api/control", "/api/watchlist"):
                    return self._send(404, {"error": "not found"})
                if not dash._basic_ok(self.headers.get("Authorization")):
                    return self._deny()
                if u.path == "/api/watchlist":
                    try:
                        n = min(int(self.headers.get("Content-Length") or 0), 2000)
                        body = json.loads(self.rfile.read(n) or b"{}")
                        if not isinstance(body, dict):
                            raise ValueError
                    except Exception:
                        return self._send(400, {"error": "bad json"})
                    action = str(body.get("action", "")).lower()
                    if action not in WATCH_ACTIONS or dash.watch_fn is None:
                        return self._send(400, {"error": f"action must be one of {WATCH_ACTIONS}"})
                    msg = dash.watch_fn(action, body)
                    log.warning("watchlist %s %s from dashboard", action, body.get("symbol") or body.get("id"))
                    return self._send(200, {"ok": True, "message": msg})
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
.tk{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);display:flex;gap:8px;overflow-x:auto;padding:8px 10px;scrollbar-width:none}
.tk::-webkit-scrollbar{display:none}
.tc{flex:0 0 auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:5px 9px;min-width:104px;transition:background .6s}
.tc .n{font-size:11px;color:var(--mut);white-space:nowrap}.tc .p{font-weight:700;font-variant-numeric:tabular-nums;font-size:15px}
.tc .d{font-size:11px;font-variant-numeric:tabular-nums}.fu{background:rgba(47,191,113,.28)!important}.fd{background:rgba(239,78,78,.28)!important}
.chip{display:inline-block;padding:5px 11px;border-radius:999px;border:1px solid var(--line);font-size:12px;margin:3px 0 3px 4px;cursor:pointer}
.chip.on{background:var(--b);color:#fff;border-color:var(--b)}
.ni{border-top:1px solid var(--line);margin-top:8px;padding-top:8px}.ni .en{font-size:12px;color:var(--mut);direction:ltr;text-align:left}
.ni .ar{font-size:14px;margin-top:3px}
.chw{margin-top:8px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--bg)}
.chw svg{display:block;width:100%;height:auto}
.an{font-size:12.5px;margin-top:6px}.an summary{cursor:pointer;color:var(--b);font-weight:600}
.an .row{display:flex;gap:6px;margin-top:4px}.an .row b{white-space:nowrap;min-width:84px;color:var(--mut);font-weight:500}
.vd{margin-top:6px;padding:6px 9px;border-radius:8px;background:rgba(76,141,255,.12);font-size:13px}.imp2{border-right:3px solid var(--r);padding-right:7px}.imp1{border-right:3px solid var(--y);padding-right:7px}
</style></head><body><div class="tk" id="tk"><div class="mut">⏳ تحميل الأسعار…</div></div><div class="w">
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
<h2>قائمة المراقبة وخطط الدخول</h2>
<div class="mut" id="wlsub">يتداول البوت آلياً فقط ما نجح في اختبار تاريخه (✅). الباقي مراقبة مع خطة جاهزة.</div>
<div id="wl"></div>
<h2>📅 الأجندة الاقتصادية (أحداث هذا الأسبوع المؤثرة)</h2>
<div id="cal"></div>
<h2>📰 نافذة الأخبار — مصادر القرار المالي</h2>
<div id="nchips"></div>
<div id="news"></div>
<div class="c" style="margin-top:10px"><div class="k" style="margin-bottom:6px">➕ إضافة رمز (مؤشر، زوج عملات، معدن، عملة رقمية)</div>
 <input id="nsym" placeholder="مثال: XAU/USD أو EUR/USD أو SPY أو BTC/USD" dir="ltr" style="width:100%;font:inherit;padding:11px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--tx)">
 <select id="ncls" style="width:100%;font:inherit;padding:10px;margin-top:8px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--tx)">
  <option value="">النوع: تلقائي</option><option value="etf">مؤشر / صندوق (SPY, DIA, QQQ)</option><option value="fx">زوج عملات</option><option value="metal">معادن (ذهب، فضة)</option><option value="crypto">عملة رقمية</option></select>
 <div style="margin:8px 0;display:flex;gap:14px;flex-wrap:wrap">
  <label><input type="checkbox" id="st_scalp" checked> سكالبينج (15 د)</label>
  <label><input type="checkbox" id="st_rsi2"> ارتداد RSI2 (ساعة)</label>
  <label><input type="checkbox" id="st_breakout"> اختراق</label></div>
 <button class="bu" style="width:100%" onclick="addSym()">➕ إضافة إلى المراقبة</button>
 <div class="mut" style="margin-top:6px">داو جونز = DIA · ناسداك = QQQ · S&P 500 = SPY. الفضة والنفط تحتاجان خطة Twelve Data مدفوعة.</div>
 <div id="wmsg" class="mut"></div></div>
<h2>الصفقات المفتوحة</h2><div class="c scroll"><table id="syms"></table></div>
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
  LAST=s; renderWL(s);
  let h='<tr><th>الرمز</th><th>الاستراتيجية</th><th>الحالة</th><th>السعر</th><th>دخول</th><th>وقف</th><th>هدف</th><th>غير محقق</th><th>عمر السعر</th></tr>';
  for(const [k,v] of Object.entries(s.symbols||{})){const p=v.position; if(!p) continue;
   h+='<tr><td>'+esc(k)+'</td><td>'+esc({breakout:'اختراق',scalp:'سكالبينج',pullback:'سحب',rsi2:'ارتداد RSI2'}[v.strategy]||v.strategy||'')+'</td><td>'+(p?(p.side=='long'?'شراء ':'بيع ')+p.qty:'بلا صفقة')+'</td><td>'+f(v.last_price)+'</td><td>'+(p?f(p.entry):'–')+'</td><td>'+(p?f(p.stop)+(p.trail_moves?' <span class="mut">🪜'+p.trail_moves+'</span>':''):'–')+'</td><td>'+(p?(p.target==null?'<span class="mut">وقف متحرك</span>':f(p.target)):'–')+'</td><td class="'+(p&&p.unrealized<0?'neg':'pos')+'">'+(p?f(p.unrealized):'–')+'</td><td>'+(v.quote_age_s==null?'–':v.quote_age_s+' ث')+'</td></tr>';}
  $('syms').innerHTML=Object.values(s.symbols||{}).some(v=>v.position)?h:'<tr><td>لا صفقات مفتوحة</td></tr>';
  let gh=[]; if(g.peak) gh.push('قمة الرصيد: '+f(g.peak));
  for(const [k,v] of Object.entries(g.paused||{})) gh.push('موقوف '+esc(k)+' لمدة '+v+' د');
  for(const [k,v] of Object.entries(g.recent_avg_R||{})) gh.push('متوسط آخر الصفقات '+esc(k)+': '+v+'R');
  for(const e of (g.last_events||[])) gh.push('• '+esc(e));
  $('guard').innerHTML=gh.length?gh.join('<br>'):'لا تدخلات حتى الآن';
  let t='<tr><th>الرمز</th><th>الاتجاه</th><th>السبب</th><th>الدخول</th><th>الخروج</th><th>الربح</th><th>R</th><th>انزلاق</th><th>وقت الخروج</th></tr>';
  for(const x of (s.recent_trades||[]).slice().reverse()){const p=parseFloat(x.pnl);
   t+='<tr><td>'+esc(x.symbol)+'</td><td>'+esc(x.side)+'</td><td>'+esc(x.reason)+'</td><td>'+f(+x.entry)+'</td><td>'+f(+x.exit)+'</td><td class="'+(p<0?'neg':'pos')+'">'+f(p)+'</td><td>'+f(+x.R)+'</td><td class="mut" dir="ltr">'+(x.slip_in_bps!=null?f(+x.slip_in_bps,1)+' / '+f(+x.slip_out_bps,1):'–')+'</td><td>'+esc(String(x.exit_ts).slice(5,16))+'</td></tr>';}
  $('trades').innerHTML=(s.recent_trades||[]).length?t:'<tr><td>لا صفقات مغلقة بعد</td></tr>';
  $('log').textContent=(s.log||[]).slice().reverse().join('\n');
 }catch(e){$('sub').textContent='تعذّر الاتصال بالبوت ('+e.message+')';}
}
async function cmd(c,q){ if(!confirm(q)) return;
 const r=await fetch('api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cmd:c})});
 const j=await r.json().catch(()=>({})); $('msg').textContent=j.message||j.error||r.status; load(); }
const SN={scalp:'سكالبينج',rsi2:'ارتداد RSI2',breakout:'اختراق',pullback:'سحب'};
const CN={etf:'مؤشر',fx:'عملات',metal:'معادن',crypto:'رقمية',stock:'مؤشر'};
function dp(x){if(x==null)return'–';const a=Math.abs(x);return Number(x).toLocaleString('en-US',{maximumFractionDigits:a>=1000?1:a>=10?2:a>=1?4:6});}
function sideTxt(pl,buy){ if(!pl) return '<span class="mut">'+(buy?'لا شراء الآن':'لا بيع الآن')+'</span>';
 const w=buy?'شراء':'بيع', op=pl.op=='<'?'تحت':'فوق';
 return '<b class="'+(buy?'pos':'neg')+'">'+w+'</b> إذا أغلقت الشمعة '+op+' <b>'+dp(pl.trigger)+'</b> <span class="mut">('+(pl.dist_pct>0?'+':'')+pl.dist_pct+'%)</span>'+(pl.bound!=null?' <span class="mut">و'+(buy?'فوق':'تحت')+' '+dp(pl.bound)+'</span>':'')+'<br><span class="mut">وقف</span> '+dp(pl.stop)+' · <span class="mut">هدف</span> '+dp(pl.target);}
function stTxt(v){ if(v.disabled) return '<span class="badge warn">⏸ موقوف يدوياً</span>';
 if(v.auto_off) return '<span class="badge bad">🛑 أوقفه البوت: نتائجه الفعلية سيئة</span>';
 if(v.trading) return '<span class="badge ok">✅ تداول آلي</span>';
 const q=v.qual; if(!q) return '<span class="badge warn">⏳ قيد الاختبار</span>';
 if(q.status=='error') return '<span class="badge bad">⚠️ '+esc(q.reason||'خطأ')+'</span>';
 return '<span class="badge warn">👁 مراقبة فقط</span>';}
function fwTxt(f){ if(!f||!f.n) return ''; return '<div style="font-size:12px">النتائج الفعلية: '+f.n+' صفقة · ربح '+Math.round(100*f.w/f.n)+'% · '+(f.sumR>=0?'+':'')+(f.sumR/f.n).toFixed(2)+'R · '+(f.pnl>=0?'+':'')+f.pnl.toFixed(2)+'$</div>';}
function qTxt(q){ if(!q||q.n==null) return ''; let t='اختبار تاريخه: '+q.n+' صفقة';
 if(q.wr!=null) t+=' · ربح '+q.wr+'% · '+(q.avgR>0?'+':'')+q.avgR+'R · معامل '+q.pf;
 return '<div class="mut" style="font-size:12px">'+esc(t)+(q.status!='ok'&&q.reason?' — '+esc(q.reason):'')+'</div>';}
function stxt(x){ if(!x) return ''; const H={1:'صاعد',"-1":'هابط',0:'متذبذب'};
 return '<div style="font-size:12px;margin-top:2px">🧭 '+esc(x.text)+(x.htf!=null?' · الإطار الأكبر: '+H[x.htf]:'')+(x.swing_hi!=null?'<span class="mut" dir="ltr"> · قمة '+dp(x.swing_hi)+' / قاع '+dp(x.swing_lo)+'</span>':'')+'</div>';}
const TKP={}; let NF='all', NEWS=null;
let CH={}, LAST=null; const SEL={}, OPEN={};
function slipTxt(x){ if(!x||!x.n) return ''; return '<div class="mut" style="font-size:12px">الانزلاق المقاس ('+x.n+' صفقة): دخول '+(x.in>=0?'+':'')+x.in+' · خروج '+(x.out>=0?'+':'')+x.out+' نقطة أساس'+(x.fill?' · تنفيذ الوسيط '+(x.fill>=0?'+':'')+x.fill:'')+'</div>';}
function anTxt(id,v){ const a=v.analysis; if(!a) return '';
 let h='<div class="vd">🎯 '+esc(a.verdict)+'</div><details class="an" '+(OPEN[id]?'open':'')+' ontoggle="OPEN[\''+esc(id)+'\']=this.open"><summary>🔎 تحليل الاستراتيجية لهذا الرمز</summary><div class="mut" style="margin-top:4px">'+esc(a.desc)+'</div>';
 for(const [k,x] of (a.lines||[])) h+='<div class="row"><b>'+esc(k)+'</b><span>'+esc(x)+'</span></div>';
 return h+'</details>';}
function pickSel(t,rows){ const ids=rows.map(r=>r[0]).filter(i=>CH[i]); if(!ids.length) return null;
 if(SEL[t]&&ids.includes(SEL[t])) return SEL[t]; const op=ids.find(i=>CH[i].pos); return op||ids[0];}
function setSel(t,id){SEL[t]=id; if(LAST) renderWL(LAST);}
function chartBlock(t,rows){ const sid=pickSel(t,rows); if(!sid) return '<div class="mut" style="font-size:12px;margin-top:6px">📊 الشموع تظهر بعد أول شمعة مغلقة…</div>';
 let tabs=''; const ids=rows.map(r=>r[0]).filter(i=>CH[i]);
 if(ids.length>1) for(const i of ids){ const v=(LAST.symbols||{})[i]||{}; tabs+='<span class="chip '+(i==sid?'on':'')+'" onclick="setSel(\''+esc(t)+'\',\''+esc(i)+'\')">'+esc((SN[v.strategy]||v.strategy)+' · '+CH[i].iv)+(CH[i].pos?' ●':'')+'</span>';}
 return '<div style="margin-top:6px">'+tabs+'</div><div class="chw" dir="ltr">'+svgChart(CH[sid],(LAST.symbols||{})[sid])+'</div>'
  +'<div class="mut" style="font-size:11px;margin-top:3px">🟩🟥 الشموع · ▲▼ قيعان وقمم مؤكدة · <span style="color:var(--b)">━ دخول</span> · <span style="color:var(--r)">━ وقف</span> · <span style="color:var(--g)">━ هدف</span> · ┅ مستوى إشارة الشراء/البيع</div>';}
function hm(ts){ return new Date(ts*1000).toLocaleString('en-GB',{timeZone:'Asia/Baghdad',weekday:'short',hour:'2-digit',minute:'2-digit',hour12:false});}
function svgChart(c,v){ const W=360,H=210,L=4,R=52,T=8,B=20, n=c.t.length; if(!n) return '';
 const px=v&&v.last_price; let lo=Math.min(...c.l), hi=Math.max(...c.h);
 const lv=[]; const P=c.pos; if(P){lv.push(P.entry,P.stop); if(P.target!=null) lv.push(P.target); if(P.stop0!=P.stop) lv.push(P.stop0);}
 for(const k in (c.trig||{})) lv.push(c.trig[k]); if(px) lv.push(px);
 const span0=hi-lo||1; for(const x of lv){ if(x>=lo-span0*0.6&&x<=hi+span0*0.6){lo=Math.min(lo,x);hi=Math.max(hi,x);} }
 const pad=(hi-lo)*0.06||1; lo-=pad; hi+=pad;
 const cw=(W-L-R)/n, y=x=>T+(hi-x)/(hi-lo)*(H-T-B), xs=i=>L+cw*(i+0.5);
 let s='<svg viewBox="0 0 '+W+' '+H+'" xmlns="http://www.w3.org/2000/svg" font-size="9" font-family="system-ui,sans-serif">';
 for(let g=0;g<=4;g++){ const yy=T+g*(H-T-B)/4, val=hi-g*(hi-lo)/4; s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+yy+'" y2="'+yy+'" style="stroke:var(--line)" stroke-width="0.5"/><text x="'+(W-R+3)+'" y="'+(yy+3)+'" style="fill:var(--mut)">'+dp(val)+'</text>';}
 for(let i=0;i<n;i+=Math.ceil(n/4)) s+='<text x="'+xs(i)+'" y="'+(H-6)+'" text-anchor="middle" style="fill:var(--mut)">'+hm(c.t[i])+'</text>';
 const LC={VWAP:'var(--y)',EMA200:'#9b3df0',EMA20:'var(--y)',EMA50:'#9b3df0'};
 for(const [nm,arr] of Object.entries(c.lines||{})){ let d=''; arr.forEach((x,i)=>{ if(x==null) return; d+=(d?'L':'M')+xs(i).toFixed(1)+' '+y(x).toFixed(1);}); s+='<path d="'+d+'" fill="none" style="stroke:'+(LC[nm]||'var(--mut)')+'" stroke-width="1" opacity="0.8"/>';}
 const bw=Math.max(1,cw*0.62);
 for(let i=0;i<n;i++){ const up=c.c[i]>=c.o[i], col=up?'var(--g)':'var(--r)', x=xs(i);
  s+='<line x1="'+x+'" x2="'+x+'" y1="'+y(c.h[i])+'" y2="'+y(c.l[i])+'" style="stroke:'+col+'" stroke-width="0.8"/>';
  const yt=y(Math.max(c.o[i],c.c[i])), yb=y(Math.min(c.o[i],c.c[i]));
  s+='<rect x="'+(x-bw/2)+'" y="'+yt+'" width="'+bw+'" height="'+Math.max(0.8,yb-yt)+'" style="fill:'+col+'"/>';}
 for(const [i,k,val] of (c.piv||[])){ const x=xs(i); s+=k=='H'?'<path d="M'+(x-3)+' '+(y(val)-7)+'h6l-3 4z" style="fill:var(--r)"/>':'<path d="M'+(x-3)+' '+(y(val)+7)+'h6l-3 -4z" style="fill:var(--g)"/>';}
 const used=[]; const hl=(val,col,lab,dash,w)=>{ if(val==null||val<lo||val>hi) return ''; const yy=y(val); let ly=yy;
  for(let k=0;k<6&&used.some(u=>Math.abs(u-ly)<12);k++) ly+=(ly>H/2?-12:12); used.push(ly);
  return '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+yy+'" y2="'+yy+'" style="stroke:'+col+'" stroke-width="'+(w||1)+'"'+(dash?' stroke-dasharray="4 3"':'')+'/><rect x="'+(W-R+1)+'" y="'+(ly-6)+'" width="'+(R-2)+'" height="12" rx="2" style="fill:'+col+'"/><text x="'+(W-R+3)+'" y="'+(ly+3)+'" fill="#fff" font-weight="600">'+lab+' '+dp(val)+'</text>';};
 if(c.trig){ if(c.trig.long!=null) s+=hl(c.trig.long,'var(--g)','B',true,0.8); if(c.trig.short!=null) s+=hl(c.trig.short,'var(--r)','S',true,0.8);}
 if(P){ if(P.stop0!=P.stop) s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(P.stop0)+'" y2="'+y(P.stop0)+'" style="stroke:var(--r)" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.6"/>';
  s+=hl(P.entry,'var(--b)','E',false,1.2)+hl(P.stop,'var(--r)','SL',false,1.2)+(P.target!=null?hl(P.target,'var(--g)','TP',false,1.2):'');
  let ei=c.t.findIndex(t=>t>=P.t); if(ei<0) ei=n-1; const ex=xs(ei), ey=y(P.entry);
  s+=P.side==1?'<path d="M'+ex+' '+(ey+4)+'l-5 9h10z" style="fill:var(--b)"/>':'<path d="M'+ex+' '+(ey-4)+'l-5 -9h10z" style="fill:var(--b)"/>';}
 if(px) s+=hl(px,'#5b6475','',false,0.6);
 return s+'</svg>';}

function age(s){return s<120?'لحظي':s<3600?'متأخر '+Math.round(s/60)+' د':'السوق مغلق';}
async function loadTicker(){ try{ const r=await fetch('api/ticker',{cache:'no-store'}); if(!r.ok) return; const t=await r.json();
 let h=''; for(const i of (t.items||[])){ const old=TKP[i.key]; const fl=old==null||old===i.price?'':(i.price>old?'fu':'fd'); TKP[i.key]=i.price;
  const c=i.chg_pct==null?'':'<div class="d '+(i.chg_pct>=0?'pos':'neg')+'" dir="ltr">'+(i.chg_pct>=0?'▲ +':'▼ ')+i.chg_pct.toFixed(2)+'%</div>';
  h+='<div class="tc '+fl+'" id="tc-'+i.key+'"><div class="n">'+esc(i.name)+'</div><div class="p" dir="ltr">'+Number(i.price).toLocaleString('en-US',{minimumFractionDigits:i.dec,maximumFractionDigits:i.dec})+'</div>'+c+'<div class="n">'+(i.src=='live'?'🟢 ':'🟡 ')+age(i.age_s)+'</div></div>';}
 $('tk').innerHTML=h||'<div class="mut">لا أسعار بعد</div>';
 setTimeout(()=>document.querySelectorAll('.tc.fu,.tc.fd').forEach(e=>e.classList.remove('fu','fd')),700);}catch(e){} }
function bgT(ts){ try{return new Date(ts).toLocaleString('ar-IQ',{timeZone:'Asia/Baghdad',weekday:'short',hour:'2-digit',minute:'2-digit',hour12:false});}catch(e){return ts;} }
async function loadNews(){ try{ const r=await fetch('api/news',{cache:'no-store'}); if(!r.ok) return; NEWS=await r.json(); renderNews(); }catch(e){} }
function setNF(g){NF=g;renderNews();}
function renderNews(){ const n=NEWS; if(!n){$('news').innerHTML='<div class="c mut">النافذة متوقفة (NEWS_WATCH=0)</div>';return;}
 const g=n.groups||{}; let ch='<span class="chip '+(NF=='all'?'on':'')+'" onclick="setNF(\'all\')">الكل</span><span class="chip '+(NF=='imp'?'on':'')+'" onclick="setNF(\'imp\')">🔴 المهم فقط</span>';
 for(const [k,v] of Object.entries(g)) ch+='<span class="chip '+(NF==k?'on':'')+'" onclick="setNF(\''+k+'\')">'+esc(v)+'</span>';
 $('nchips').innerHTML=ch; const h=n.health||{};
 let x='<div class="c"><div class="mut" style="font-size:12px">'+(h.ok===false?'⚠ مصادر لا تستجيب: '+esc(Object.keys(h.bad||{}).join('، ')):h.ok?'✅ '+h.sources+' مصدر تعمل · جديد اليوم: '+(h.new_today||0):'⏳ أول فحص…')+' · للمعلومة — لا تفتح ولا تغلق صفقات</div>';
 const its=(n.items||[]).filter(i=>NF=='all'||(NF=='imp'?i.imp>=2:i.group==NF)).slice(0,40);
 for(const i of its){ x+='<div class="ni '+(i.imp>=2?'imp2':i.imp==1?'imp1':'')+'"><div class="mut" style="font-size:12px"><b>'+esc(i.src_ar)+'</b> · '+esc(bgT(i.ts))+(i.cat&&i.cat!='other'?' · '+esc(i.label):'')+(i.url?' · <a href="'+esc(i.url)+'" target="_blank" rel="noopener">المصدر ↗</a>':'')+'</div>'
  +'<div class="ar">'+(i.ar?esc(i.ar):'<span class="mut">… جارٍ الترجمة</span>')+'</div><div class="en">'+esc(i.text.slice(0,400))+'</div></div>';}
 $('news').innerHTML=x+(its.length?'':'<div class="mut">لا أخبار في هذا التصنيف</div>')+'</div>';
 const cal=n.calendar||[]; let c='<div class="c">';
 for(const e of cal){ c+='<div class="ni '+(e.impact=='High'?'imp2':'imp1')+'" style="'+(e.past?'opacity:.55':'')+'"><div style="display:flex;justify-content:space-between;gap:6px"><b>'+esc(e.ar||e.title)+'</b><span class="badge '+(e.impact=='High'?'bad':'warn')+'">'+esc(e.ccy)+' · '+(e.impact=='High'?'تأثير عالٍ':'متوسط')+'</span></div>'
  +'<div class="mut" style="font-size:12px">'+esc(bgT(e.ts))+' بغداد · المتوقع '+esc(e.forecast||'–')+' · السابق '+esc(e.previous||'–')+(e.actual?' · <b>الفعلي '+esc(e.actual)+'</b>':'')+'</div>'+(e.ar?'<div class="en">'+esc(e.title)+'</div>':'')+'</div>';}
 $('cal').innerHTML=c+(cal.length?'':'<div class="mut">لا أحداث مؤثرة قريبة</div>')+'</div>';}
function renderWL(s){ const by={}; for(const [id,v] of Object.entries(s.symbols||{})){(by[v.ticker||id]=by[v.ticker||id]||[]).push([id,v]);}
 const near=a=>Math.min(...a.map(([i,v])=>{const p=v.plan||{};return Math.min(p.long?Math.abs(p.long.dist_pct):99,p.short?Math.abs(p.short.dist_pct):99);}));
 const keys=Object.keys(by).sort((a,b)=>near(by[a])-near(by[b]));
 let h='';
 for(const t of keys){ const rows=by[t]; const v0=rows[0][1]; const px=(v0.plan&&v0.plan.price)||v0.last_price;
  h+='<div class="c" style="margin-top:8px"><div style="display:flex;justify-content:space-between;align-items:center"><div><b dir="ltr">'+esc(t)+'</b> <span class="mut">'+esc(CN[v0.class]||v0.class||'')+' · '+dp(px)+'</span></div>'
   +'<button class="bs" style="padding:6px 10px;font-size:12px" onclick="wl(\'remove\',{symbol:\''+esc(t)+'\'},\'حذف '+esc(t)+' من المراقبة؟\')">✕ حذف</button></div>'
   +chartBlock(t,rows);
  for(const [id,v] of rows){ const p=v.plan||{};
   h+='<div style="border-top:1px solid var(--line);margin-top:8px;padding-top:8px"><div style="display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap"><div><b>'+esc(SN[v.strategy]||v.strategy)+'</b> <span class="mut">'+esc(v.interval||'')+(v.win_bg?' · الدخول '+esc(v.win_bg)+' بغداد':'')+'</span></div><div>'+stTxt(v)+'</div></div>'
    +'<div style="margin-top:4px">'+sideTxt(p.long,true)+'</div>'+(v.allow_short?'<div style="margin-top:4px">'+sideTxt(p.short,false)+'</div>':'')
    +(p.note?'<div class="mut" style="font-size:12px">'+esc(p.note)+'</div>':'')+stxt(p.structure)+(p.bar?'<div class="mut" style="font-size:11px">آخر شمعة: '+esc(String(p.bar).slice(5,16))+'</div>':'<div class="mut" style="font-size:12px">بانتظار أول بيانات…</div>')
    +anTxt(id,v)+slipTxt(v.slip)+fwTxt(v.fw)+(v.auto_off?'<div class="mut" style="font-size:12px">'+esc(v.auto_off)+'</div>':'')+qTxt(v.qual)
    +'<button class="'+(v.disabled||v.auto_off?'br':'bs')+'" style="padding:6px 10px;font-size:12px;margin-top:6px" onclick="wl(\''+(v.disabled||v.auto_off?'on':'off')+'\',{id:\''+esc(id)+'\'},null)">'+(v.disabled||v.auto_off?'▶ تفعيل التداول الآلي':'⏸ إيقاف التداول الآلي لهذا السطر')+'</button></div>';}
  h+='</div>';}
 $('wl').innerHTML=h||'<div class="c">القائمة فارغة</div>';
 const cr=s.credits; $('wlsub').textContent='يتداول البوت آلياً فقط ما نجح في اختبار تاريخه (✅). الباقي مراقبة مع خطة جاهزة.'+(cr?' · رصيد بيانات اليوم: '+cr.used+' / '+cr.cap:'');}
async function wl(action,body,q){ if(q&&!confirm(q)) return; body.action=action;
 const r=await fetch('api/watchlist',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json().catch(()=>({})); $('wmsg').textContent=j.message||j.error||r.status; alert(j.message||j.error||r.status); load(); }
function addSym(){ const sym=$('nsym').value.trim().toUpperCase(); if(!sym){alert('اكتب الرمز');return;}
 const st=['scalp','rsi2','breakout'].filter(x=>$('st_'+x).checked); if(!st.length){alert('اختر استراتيجية واحدة على الأقل');return;}
 wl('add',{symbol:sym,'class':$('ncls').value,strategies:st},'إضافة '+sym+'؟ (يعيد البوت التشغيل)'); }
async function loadCharts(){ try{ const r=await fetch('api/charts',{cache:'no-store'}); if(!r.ok) return; CH=await r.json(); if(LAST) renderWL(LAST);}catch(e){} }
load(); setInterval(load,5000); loadCharts(); setInterval(loadCharts,15000); loadTicker(); setInterval(loadTicker,3000); loadNews(); setInterval(loadNews,60000);
</script></body></html>"""
