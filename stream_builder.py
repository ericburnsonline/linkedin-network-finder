"""
Context Builder — interactive local web UI for generating streams.yaml
----------------------------------------------------------------------
Launches a local browser-based wizard that interviews you about each
match context and uses Claude AI to generate scoring signals.

Usage:
    python src/stream_builder.py
    python src/stream_builder.py --port 8765
    python src/stream_builder.py --output config/my_streams.yaml
"""

import argparse
import json
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

OUTPUT_PATH = "config/streams.yaml"

# ---------------------------------------------------------------------------
# Signal generation via Claude
# ---------------------------------------------------------------------------

def _resolve_model_builder(client) -> str:
    """Auto-discover best available model. Checks ANTHROPIC_MODEL env var first."""
    env_model = os.environ.get("ANTHROPIC_MODEL", "").strip()
    if env_model:
        return env_model
    try:
        models_page = client.models.list()
        available = [m.id for m in models_page.data]
        for model_id in available:
            if "sonnet" in model_id.lower():
                return model_id
        if available:
            return available[0]
    except Exception:
        pass
    return "claude-sonnet-4-5"


def generate_signals(answers: dict, api_key: str = None) -> dict:
    if not HAS_ANTHROPIC:
        return {"error": "anthropic package not installed. Run: pip install anthropic"}
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "ANTHROPIC_API_KEY not set. Export it or pass --api-key."}

    client = anthropic.Anthropic(api_key=key)
    model  = _resolve_model_builder(client)

    prompt = f"""You are building a LinkedIn contact scoring config. Based on the match context details below, generate a complete scoring signal set.

CONTEXT NAME: {answers.get('name','')}
DESCRIPTION: {answers.get('basics_desc','')}
IDEAL BUYER TITLES: {', '.join(answers.get('ideal_titles', []))}
IDEAL COMPANY PROFILE: {answers.get('ideal_company','')}
BUYER PAIN / TRIGGER: {answers.get('ideal_pain','')}
PAST WINS (examples): {answers.get('wins','')}
WIN PATTERN: {answers.get('win_pattern','')}
DEAD ENDS (never convert): {answers.get('deadends','')}
AVOID: {answers.get('avoid','')}
DEAL SIZE: {answers.get('deal_size','unknown')}
SALES CYCLE: {answers.get('sales_cycle','unknown')}
RECURRENCE: {answers.get('recurrence','unknown')}
EXTRA CONTEXT: {answers.get('extra','')}

Return ONLY valid JSON, no prose, no markdown fences:
{{
  "id": "snake_case_id",
  "description": "one sentence",
  "max_score": <integer>,
  "title_signals": [{{"term":"...", "weight": 1-8}}, ...],
  "company_signals": [{{"term":"...", "weight": 1-5}}, ...],
  "negative_signals": [{{"term":"...", "weight": 1-10}}, ...],
  "reasoning": "2-3 sentences on key weighting decisions"
}}

Rules:
- title_signals: 8-15 terms. Weight 6-8 exact match, 3-5 strong, 1-2 weak.
- company_signals: 5-10 terms for industry/stage/model keywords.
- negative_signals: 4-8 terms. 6-10 hard disqualifiers, 2-4 soft.
- max_score: realistic sum a perfect contact would match (2-3 title + 1-2 company signals).
- Terms lowercase, 1-4 words, as they appear in LinkedIn title/company."""

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def signals_to_yaml(signals: dict, outreach_template: str = "") -> str:
    sid = signals.get("id", "context")
    desc = signals.get("description", "")
    max_s = signals.get("max_score", 10)
    ts = "\n".join(f'      - {{ term: "{k["term"]}", weight: {k["weight"]} }}'
                   for k in signals.get("title_signals", []))
    cs = "\n".join(f'      - {{ term: "{k["term"]}", weight: {k["weight"]} }}'
                   for k in signals.get("company_signals", []))
    ns = "\n".join(f'      - {{ term: "{k["term"]}", weight: {k["weight"]} }}'
                   for k in signals.get("negative_signals", []))
    tmpl = outreach_template or f"templates/{sid}.md"
    return f"""  - id: {sid}
    description: "{desc}"
    max_score: {max_s}

    title_signals:
{ts}

    company_signals:
{cs}

    negative_signals:
{ns}

    outreach_template: {tmpl}
"""


def write_yaml(streams: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(signals_to_yaml(s["signals"]) for s in streams if s.get("signals"))
    content = (
        "# streams.yaml — generated by context_builder.py\n"
        "# Keep this file private — it is in .gitignore\n\n"
        "streams:\n\n" + body
    )
    with open(output_path, "w") as f:
        f.write(content)
    return content


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    output_path = OUTPUT_PATH
    api_key = None

    def log_message(self, format, *args):
        pass  # silence default access log

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == "/status":
            self.send_json({"has_anthropic": HAS_ANTHROPIC,
                            "has_api_key": bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY")),
                            "output_path": self.output_path})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/generate":
            result = generate_signals(body.get("answers", {}), self.api_key)
            self.send_json(result)

        elif self.path == "/export":
            streams = body.get("streams", [])
            try:
                yaml_text = write_yaml(streams, self.output_path)
                self.send_json({"ok": True, "path": self.output_path, "yaml": yaml_text})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


# ---------------------------------------------------------------------------
# Embedded HTML UI
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Context Builder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --blue:#185FA5;--green:#1D9E75;--amber:#BA7517;
  --border:rgba(0,0,0,0.1);--bg:#F7F6F3;--surface:#fff;
  --text:#1a1a18;--muted:#666;--radius:8px;
}
body{font-family:'Fraunces',Georgia,serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.6;min-height:100vh}
a{color:var(--blue)}
.page-header{background:var(--text);color:#fff;padding:1.5rem 2rem;display:flex;align-items:baseline;justify-content:space-between}
.page-header h1{font-size:24px;font-weight:300;font-style:italic}
.page-header .sub{font-family:'DM Mono',monospace;font-size:11px;color:rgba(255,255,255,0.45)}
.layout{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 60px)}
.sidebar{background:#fff;border-right:1px solid var(--border);padding:1.25rem}
.sidebar h3{font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:.75rem}
.stream-list{display:flex;flex-direction:column;gap:6px;margin-bottom:1rem}
.stream-item{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:var(--radius);border:1px solid var(--border);cursor:pointer;font-size:13px;transition:border-color .15s}
.stream-item:hover{border-color:rgba(0,0,0,0.25)}
.stream-item.active{border-color:var(--blue);background:#E6F1FB20}
.stream-item .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.stream-item .sname{flex:1;font-weight:500}
.stream-item .badge{font-family:'DM Mono',monospace;font-size:9px;padding:1px 5px;border-radius:6px}
.badge-done{background:#EAF3DE;color:#27500A}
.badge-wip{background:#FAEEDA;color:#412402}
.btn-add{width:100%;font-size:12px;padding:7px;border:1px dashed var(--border);border-radius:var(--radius);background:transparent;color:var(--muted);cursor:pointer;font-family:'DM Mono',monospace;transition:all .15s}
.btn-add:hover{border-color:var(--blue);color:var(--blue)}
.btn-export{width:100%;margin-top:.75rem;font-size:12px;padding:8px;border:1px solid var(--green);border-radius:var(--radius);background:transparent;color:var(--green);cursor:pointer;font-family:'DM Mono',monospace;transition:all .15s}
.btn-export:hover{background:var(--green);color:#fff}
.main{padding:1.75rem 2rem;max-width:680px}
.step-bar{display:flex;gap:5px;align-items:center;margin-bottom:1.5rem}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--border);flex-shrink:0;transition:all .2s}
.sdot.done{background:var(--green)}
.sdot.active{background:var(--blue);width:10px;height:10px}
.step-lbl{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);margin-left:6px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem;margin-bottom:1rem}
.ql{font-size:13px;font-weight:500;margin-bottom:3px}
.qh{font-size:12px;color:var(--muted);margin-bottom:8px;line-height:1.5}
textarea,input[type=text]{width:100%;font-size:13px;padding:8px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);color:var(--text);resize:vertical;font-family:'Fraunces',Georgia,serif;line-height:1.5}
textarea:focus,input[type=text]:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 2px rgba(24,95,165,0.12)}
.tag-row{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.tag{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:2px 8px;border-radius:10px;background:#F1EFE8;border:1px solid var(--border)}
.tag button{background:none;border:none;cursor:pointer;color:var(--muted);padding:0;font-size:13px;line-height:1}
.add-row{display:flex;gap:6px;margin-top:6px}
.add-row input{flex:1}
.add-row button{font-size:12px;padding:6px 12px;border:1px solid var(--border);border-radius:var(--radius);background:transparent;color:var(--muted);cursor:pointer;white-space:nowrap;font-family:'DM Mono',monospace}
.add-row button:hover{background:#F1EFE8}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:12px;padding:4px 10px;border-radius:10px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;font-family:'DM Mono',monospace;transition:all .15s}
.chip:hover{border-color:rgba(0,0,0,0.3)}
.chip.sel{background:#E6F1FB;color:var(--blue);border-color:var(--blue)}
.nav-row{display:flex;justify-content:space-between;align-items:center;margin-top:1rem}
.nav-row button{font-size:13px;padding:7px 16px;border:1px solid var(--border);border-radius:var(--radius);background:transparent;color:var(--text);cursor:pointer;font-family:'DM Mono',monospace}
.nav-row button:hover{background:#F1EFE8}
.btn-primary{background:var(--blue)!important;color:#fff!important;border-color:var(--blue)!important}
.btn-primary:hover{background:#0C447C!important}
.btn-primary:disabled{opacity:.4;cursor:not-allowed!important}
.sp-title{font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:5px}
.pills{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.pill{font-size:11px;padding:2px 8px;border-radius:10px}
.pill-t{background:#E6F1FB;color:#0C447C}
.pill-c{background:#EAF3DE;color:#27500A}
.pill-n{background:#FCEBEB;color:#791F1F}
.pill .w{opacity:.6;font-weight:500}
.yaml-box{font-family:'DM Mono',monospace;font-size:11px;background:#F7F6F3;border:1px solid var(--border);border-radius:var(--radius);padding:12px;white-space:pre;overflow-x:auto;line-height:1.6;color:var(--text);margin-top:.75rem}
.reasoning{font-size:12px;color:var(--muted);font-style:italic;line-height:1.6;padding:10px 12px;background:#F7F6F3;border-radius:var(--radius);margin-bottom:1rem}
.err{font-size:12px;color:#A32D2D;margin-top:6px;padding:8px 10px;background:#FCEBEB;border-radius:var(--radius)}
.loading-row{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;font-style:italic;padding:.5rem 0}
.spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.success-banner{display:flex;align-items:center;gap:8px;padding:10px 14px;background:#EAF3DE;border-radius:var(--radius);font-size:13px;color:#27500A;margin-top:.75rem}
.home-empty{padding:3rem;text-align:center;color:var(--muted);font-style:italic}
</style>
</head>
<body>
<div class="page-header">
  <h1>Context Builder</h1>
  <span class="sub">linkedin-scorer · config generator</span>
</div>
<div class="layout">
  <div class="sidebar">
    <h3>Revenue streams</h3>
    <div class="stream-list" id="sidebar-list"></div>
    <button class="btn-add" id="btn-add">+ Add context</button>
    <button class="btn-export" id="btn-export" style="display:none">Export streams.yaml</button>
    <div id="export-result"></div>
  </div>
  <div class="main" id="main-panel"></div>
</div>

<script>
const STEPS=["Basics","Ideal buyer","Past wins","Dead ends","Deal context","Generate"];
const COLORS=["#185FA5","#1D9E75","#BA7517","#534AB7","#D85A30","#D4537E"];
const state={streams:[],cur:null,step:0,answers:{},result:null,loading:false,err:null};

function dot(i){return`<span class="sdot ${i<state.step-1?'done':i===state.step-1?'active':''}"></span>`}

function render(){
  renderSidebar();
  if(state.cur===null){renderHome();return;}
  if(state.step===6){renderResult();return;}
  renderStep();
}

function renderSidebar(){
  const list=document.getElementById("sidebar-list");
  list.innerHTML=state.streams.map((s,i)=>`
    <div class="stream-item ${state.cur===i?'active':''}" onclick="selectStream(${i})">
      <span class="dot" style="background:${COLORS[i%COLORS.length]}"></span>
      <span class="sname">${esc(s.name||"Untitled")}</span>
      <span class="badge ${s.done?'badge-done':'badge-wip'}">${s.done?"done":"wip"}</span>
    </div>`).join("");
  const hasDone=state.streams.some(s=>s.done);
  document.getElementById("btn-export").style.display=hasDone?"":"none";
}

function renderHome(){
  const p=document.getElementById("main-panel");
  p.innerHTML=state.streams.length
    ?`<div class="home-empty">Select a context on the left, or add a new one.</div>`
    :`<div class="home-empty">Click "+ Add context" to get started.<br><br>You'll answer 5 quick questions per context.<br>Claude generates the scoring signals automatically.</div>`;
}

function renderStep(){
  const p=document.getElementById("main-panel");
  const dots=STEPS.map((_,i)=>dot(i)).join("");
  const lbl=STEPS[state.step-1]||"";
  p.innerHTML=`<div class="step-bar">${dots}<span class="step-lbl">${lbl}</span></div>
    <div id="step-body"></div>
    <div class="nav-row">
      <button id="btn-back">${state.step===1?"Cancel":"← Back"}</button>
      ${state.step<5?`<button id="btn-next" class="btn-primary">Next →</button>`:""}
      ${state.step===5?`<button id="btn-gen" class="btn-primary"${state.loading?" disabled":""}>
        ${state.loading?'<span style="display:flex;align-items:center;gap:6px"><span class="spinner"></span>Generating…</span>':'Generate signals →'}
      </button>`:""}
    </div>`;
  document.getElementById("step-body").innerHTML=getStepHTML();
  bindStep();
}

function getStepHTML(){
  const a=state.answers;
  if(state.step===1)return`
    <div class="card">
      <div class="ql">Context name</div>
      <div class="qh">A short internal label for this context — what are you trying to do?</div>
      <input type="text" id="q-name" value="${esc(a.name||'')}" placeholder="e.g. Leadership workshop attendees">
      <div class="ql" style="margin-top:1rem">Describe what you're looking for in plain English.</div>
      <div class="qh">Who would be the right person, and what would you want from them? Include any relevant context like deal size, event details, or collaboration type.</div>
      <textarea id="q-desc" rows="3" placeholder="e.g. Looking for HR and L&D leaders to invite to a leadership workshop. Or: finding potential collaborators for an open source project. Or: identifying warm leads for a consulting engagement.">${esc(a.basics_desc||'')}</textarea>
    </div>`;

  if(state.step===2)return`
    <div class="card">
      <div class="ql">Ideal buyer title(s)</div>
      <div class="qh">The person who signs the check or champions the project. Add a few variants.</div>
      <div class="tag-row" id="title-tags">${(a.ideal_titles||[]).map(v=>`<span class="tag">${esc(v)}<button onclick="removeTitle('${esc(v)}')">×</button></span>`).join("")}</div>
      <div class="add-row"><input type="text" id="title-inp" placeholder="e.g. Head of L&D, VP People, Director of Training"><button id="btn-add-title">Add</button></div>
      <div class="ql" style="margin-top:1rem">What types of companies are the best fit?</div>
      <div class="qh">Industry, stage, size, geography, business model — anything that characterizes your best accounts.</div>
      <textarea id="q-co" rows="3" placeholder="e.g. Mid-size tech companies, 50-500 employees, US-based. Or: nonprofits with an L&D budget. Or: any industry where this role exists.">${esc(a.ideal_company||'')}</textarea>
      <div class="ql" style="margin-top:1rem">What pain makes them ready to buy?</div>
      <div class="qh">The trigger that makes this context relevant — helps weight urgency signals.</div>
      <textarea id="q-pain" rows="2" placeholder="e.g. They recently reorganized and are investing in culture. Or: they just scaled rapidly and need leadership development.">${esc(a.ideal_pain||'')}</textarea>
    </div>`;

  if(state.step===3)return`
    <div class="card">
      <div class="ql">Name 3–5 past wins (or dream customers)</div>
      <div class="qh">Real company names + contact title if you remember. These never leave this page — Claude uses them to reverse-engineer the pattern.</div>
      <textarea id="q-wins" rows="5" placeholder="e.g.&#10;- Head of L&D @ Acme Corp (250 employees, tech sector)&#10;- VP People @ HealthCo (fast-growing, invested in culture)&#10;- Director of OD @ MidCo (post-merger, rebuilding team)">${esc(a.wins||'')}</textarea>
      <div class="ql" style="margin-top:1rem">What did those wins have in common?</div>
      <div class="qh">Stage, size, trigger, tone of first conversation — anything you noticed.</div>
      <textarea id="q-pat" rows="2" placeholder="e.g. All were mid-size companies going through a growth phase. All had a People leader who cared about culture.">${esc(a.win_pattern||'')}</textarea>
    </div>`;

  if(state.step===4)return`
    <div class="card">
      <div class="ql">What looks like a fit but never converts?</div>
      <div class="qh">These become negative signals — the most important filter in the scorer.</div>
      <textarea id="q-dead" rows="4" placeholder="e.g.&#10;- Very small companies under 20 people — no L&D budget&#10;- Pure technical roles — not decision makers for this&#10;- Companies actively downsizing — wrong timing">${esc(a.deadends||'')}</textarea>
      <div class="ql" style="margin-top:1rem">Any roles or company types to explicitly exclude?</div>
      <div class="qh">People who create pipeline noise, or that you've had bad experiences with.</div>
      <textarea id="q-avoid" rows="2" placeholder="e.g. Avoid companies in obvious cost-cutting mode. Avoid industries where this type of engagement rarely lands.">${esc(a.avoid||'')}</textarea>
    </div>`;

  if(state.step===5){
    const ds=a.deal_size, sc=a.sales_cycle, rc=a.recurrence;
    return`
    <div class="card">
      <div class="ql">Typical deal size</div>
      <div class="qh">Calibrates which seniority levels to target — who can approve without procurement.</div>
      <div class="chips">${["Under $5k","$5k–$25k","$25k–$100k","$100k–$500k","$500k+"].map(v=>`<span class="chip ${ds===v?'sel':''}" onclick="setPick('deal_size','${v}')">${v}</span>`).join("")}</div>
      <div class="ql" style="margin-top:1rem">Sales cycle length</div>
      <div class="chips">${["Days","1–4 weeks","1–3 months","3–6 months","6+ months"].map(v=>`<span class="chip ${sc===v?'sel':''}" onclick="setPick('sales_cycle','${v}')">${v}</span>`).join("")}</div>
      <div class="ql" style="margin-top:1rem">One-time or recurring?</div>
      <div class="chips">${["One-time","Project-based","Retainer / recurring","Subscription"].map(v=>`<span class="chip ${rc===v?'sel':''}" onclick="setPick('recurrence','${v}')">${v}</span>`).join("")}</div>
      <div class="ql" style="margin-top:1rem">Anything else Claude should know?</div>
      <div class="qh">Geography, procurement triggers, seasonal patterns, compliance signals — anything unusual about your buyers.</div>
      <textarea id="q-extra" rows="2" placeholder="e.g. Best time to reach is post-reorg or after a leadership change. Certain industries invest more heavily in this type of work.">${esc(a.extra||'')}</textarea>
      ${state.err?`<div class="err">${esc(state.err)}</div>`:""}
    </div>`;
  }
  return "";
}

function bindStep(){
  document.getElementById("btn-back")?.addEventListener("click",()=>{
    saveStep();
    if(state.step===1){state.cur=null;state.step=0;}
    else state.step--;
    render();
  });
  document.getElementById("btn-next")?.addEventListener("click",()=>{saveStep();state.step++;render();});
  document.getElementById("btn-gen")?.addEventListener("click",doGenerate);
  document.getElementById("btn-add-title")?.addEventListener("click",addTitle);
  document.getElementById("title-inp")?.addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();addTitle();}});
}

function addTitle(){
  const inp=document.getElementById("title-inp");
  const v=inp?.value?.trim();
  if(!v)return;
  state.answers.ideal_titles=[...(state.answers.ideal_titles||[]),v];
  inp.value="";
  render();
}
function removeTitle(v){
  state.answers.ideal_titles=(state.answers.ideal_titles||[]).filter(x=>x!==v);
  render();
}
function setPick(k,v){state.answers[k]=v;render();}

function saveStep(){
  const a=state.answers;
  const g=id=>document.getElementById(id)?.value||"";
  if(state.step===1){a.name=g("q-name");a.basics_desc=g("q-desc");state.streams[state.cur].name=a.name||"Untitled";}
  if(state.step===2){a.ideal_company=g("q-co");a.ideal_pain=g("q-pain");}
  if(state.step===3){a.wins=g("q-wins");a.win_pattern=g("q-pat");}
  if(state.step===4){a.deadends=g("q-dead");a.avoid=g("q-avoid");}
  if(state.step===5){a.extra=g("q-extra");}
  state.streams[state.cur].answers={...a};
}

async function doGenerate(){
  saveStep();
  state.loading=true;state.err=null;render();
  try{
    const r=await fetch("/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({answers:state.answers})});
    const d=await r.json();
    if(d.error){state.err=d.error;state.loading=false;render();return;}
    state.result=d;
    state.streams[state.cur].signals=d;
    state.streams[state.cur].done=true;
    state.loading=false;
    state.step=6;
    render();
  }catch(e){state.err=String(e);state.loading=false;render();}
}

function renderResult(){
  const p=document.getElementById("main-panel");
  const r=state.result;
  if(!r){p.innerHTML="<p>No result.</p>";return;}
  const ts=r.title_signals||[],cs=r.company_signals||[],ns=r.negative_signals||[];
  const yaml=signalToYaml(r);
  p.innerHTML=`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1D9E75" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>
      <span style="font-size:16px;font-weight:500">Signals generated for "${esc(state.streams[state.cur]?.name||"")}"</span>
    </div>
    ${r.reasoning?`<div class="reasoning">${esc(r.reasoning)}</div>`:""}
    <div class="card">
      <div class="sp-title">Title signals (${ts.length})</div>
      <div class="pills">${ts.map(k=>`<span class="pill pill-t">${esc(k.term)}<span class="w"> ×${k.weight}</span></span>`).join("")}</div>
      <div class="sp-title">Company signals (${cs.length})</div>
      <div class="pills">${cs.map(k=>`<span class="pill pill-c">${esc(k.term)}<span class="w"> ×${k.weight}</span></span>`).join("")}</div>
      <div class="sp-title">Negative signals (${ns.length})</div>
      <div class="pills">${ns.map(k=>`<span class="pill pill-n">${esc(k.term)}<span class="w"> −${k.weight}</span></span>`).join("")}</div>
      <div style="font-size:12px;color:var(--muted);margin-top:4px">Max score calibration: <strong>${r.max_score}</strong></div>
    </div>
    <div class="sp-title" style="margin-bottom:6px">YAML preview</div>
    <div class="yaml-box" id="yaml-preview">${esc(yaml)}</div>
    <div style="display:flex;justify-content:flex-end;margin-top:6px">
      <button onclick="copyYaml()" style="font-size:12px;padding:5px 12px;border:1px solid var(--border);border-radius:var(--radius);background:transparent;cursor:pointer;font-family:'DM Mono',monospace" id="copy-btn">Copy YAML</button>
    </div>
    <div class="nav-row" style="margin-top:1rem">
      <button onclick="goEdit()">← Edit answers</button>
      <button class="btn-primary" onclick="goHome()">Done — back to streams</button>
    </div>`;
}

function goEdit(){state.step=1;render();}
function goHome(){state.cur=null;state.step=0;render();}

function copyYaml(){
  const r=state.result;
  if(!r)return;
  navigator.clipboard.writeText(signalToYaml(r));
  const b=document.getElementById("copy-btn");
  if(b){b.textContent="Copied!";setTimeout(()=>b.textContent="Copy YAML",1500);}
}

function signalToYaml(r){
  const ts=(r.title_signals||[]).map(k=>`      - { term: "${k.term}", weight: ${k.weight} }`).join("\n");
  const cs=(r.company_signals||[]).map(k=>`      - { term: "${k.term}", weight: ${k.weight} }`).join("\n");
  const ns=(r.negative_signals||[]).map(k=>`      - { term: "${k.term}", weight: ${k.weight} }`).join("\n");
  return `  - id: ${r.id}
    description: "${r.description}"
    max_score: ${r.max_score}

    title_signals:
${ts}

    company_signals:
${cs}

    negative_signals:
${ns}

    outreach_template: templates/${r.id}.md`;
}

function selectStream(i){state.cur=i;state.step=state.streams[i].done?6:1;state.answers={...state.streams[i].answers};state.result=state.streams[i].signals||null;render();}

document.getElementById("btn-add").addEventListener("click",()=>{
  state.streams.push({name:"",answers:{},done:false,signals:null});
  state.cur=state.streams.length-1;
  state.step=1;state.answers={};state.result=null;state.err=null;
  render();
});

document.getElementById("btn-export").addEventListener("click",async()=>{
  const el=document.getElementById("export-result");
  el.innerHTML=`<div class="loading-row" style="margin-top:.5rem"><span class="spinner"></span>Writing file…</div>`;
  try{
    const r=await fetch("/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({streams:state.streams.filter(s=>s.done)})});
    const d=await r.json();
    if(d.ok){el.innerHTML=`<div class="success-banner" style="margin-top:.5rem"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>Saved to ${esc(d.path)}</div>`;}
    else{el.innerHTML=`<div class="err" style="margin-top:.5rem">${esc(d.error)}</div>`;}
  }catch(e){el.innerHTML=`<div class="err" style="margin-top:.5rem">${esc(String(e))}</div>`;}
});

function esc(s){return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}

render();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run(port: int = 8764, output_path: str = OUTPUT_PATH, api_key: str = None):
    Handler.output_path = output_path
    Handler.api_key = api_key

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\n  Context Builder running at {url}")
    print(f"  Output will be written to: {output_path}")
    print(f"  Press Ctrl+C to stop.\n")

    # Open browser after short delay
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nContext Builder stopped.")


def main():
    parser = argparse.ArgumentParser(description="Interactive context config builder")
    parser.add_argument("--port",    type=int, default=8764)
    parser.add_argument("--output",  default=OUTPUT_PATH, help="Output YAML path")
    parser.add_argument("--api-key", default=None,        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    run(port=args.port, output_path=args.output, api_key=args.api_key)


if __name__ == "__main__":
    main()
