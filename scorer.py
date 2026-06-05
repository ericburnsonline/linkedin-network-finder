"""
LinkedIn Connection Scorer
--------------------------
Finds relevant people in your LinkedIn network across configurable match contexts.
Outputs a scored CSV and a rich interactive HTML report with sortable columns,
tier badges, LinkedIn profile links, and per-stream score breakdown.

Usage:
    python src/scorer.py --input connections.csv
    python src/scorer.py --input connections.csv --stream my_context --top 50
    python src/scorer.py --input connections.csv --ai-enrich --top 100
    python src/scorer.py --input connections.csv --no-html   # CSV only

LinkedIn export includes these columns (among others):
    First Name, Last Name, URL, Email Address, Company, Position,
    Connected On, Location (may vary by export version)
"""

import csv
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/streams.yaml") -> dict:
    p = Path(config_path)
    # Also check next to the script itself (for flat directory layouts)
    if not p.exists():
        script_dir = Path(__file__).parent
        alt = script_dir.parent / config_path
        if alt.exists():
            p = alt
        else:
            # Last resort: look for streams.yaml in cwd or script dir
            for candidate in [Path("streams.yaml"), script_dir / "streams.yaml", script_dir.parent / "streams.yaml"]:
                if candidate.exists():
                    p = candidate
                    break
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Run: python stream_builder.py  (or create streams.yaml in the same folder as scorer.py)"
        )
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def keyword_score(text: str, keywords: list) -> tuple[int, list]:
    """Returns (total_score, list_of_matched_terms)."""
    norm = normalize(text)
    total, hits = 0, []
    for kw in keywords:
        term = normalize(kw["term"])
        weight = kw.get("weight", 1)
        if re.search(r"\b" + re.escape(term) + r"\b", norm):
            total += weight
            hits.append(kw["term"])
    return total, hits


def score_contact(row: dict, stream: dict) -> dict:
    title   = row.get("Position", row.get("Title", ""))
    company = row.get("Company", "")
    full    = f"{title} {company}"

    ts, t_hits = keyword_score(title,   stream.get("title_signals",   []))
    cs, c_hits = keyword_score(company, stream.get("company_signals", []))
    ns, n_hits = keyword_score(full,    stream.get("negative_signals",[]))

    raw  = max(0, ts + cs - ns)
    norm = min(100, int((raw / max(stream.get("max_score", 10), 1)) * 100))

    tier = "skip"
    if norm >= 70: tier = "A"
    elif norm >= 40: tier = "B"
    elif norm >= 15: tier = "C"

    return {
        "stream": stream["id"],
        "raw_score": raw,
        "score": norm,
        "tier": tier,
        "title_hits": t_hits,
        "company_hits": c_hits,
        "negative_hits": n_hits,
    }


def tier_label(score: int) -> str:
    if score >= 70: return "A"
    if score >= 40: return "B"
    if score >= 15: return "C"
    return "skip"


# ---------------------------------------------------------------------------
# Model auto-discovery
# ---------------------------------------------------------------------------

_resolved_model = None

def resolve_model(client) -> str:
    """
    Return the best available model, auto-discovered from the API.
    Cached for the lifetime of the process.

    Override with the ANTHROPIC_MODEL env var to pin a specific version:
        set ANTHROPIC_MODEL=claude-opus-4-5
    """
    global _resolved_model
    if _resolved_model:
        return _resolved_model

    env_model = os.environ.get("ANTHROPIC_MODEL", "").strip()
    if env_model:
        print(f"  Using model from ANTHROPIC_MODEL: {env_model}")
        _resolved_model = env_model
        return _resolved_model

    try:
        models_page = client.models.list()
        available = [m.id for m in models_page.data]
        # API returns newest first; prefer sonnet, fall back to first available
        for model_id in available:
            if "sonnet" in model_id.lower():
                print(f"  Auto-selected model: {model_id}")
                _resolved_model = model_id
                return _resolved_model
        if available:
            print(f"  No Sonnet found, using: {available[0]}")
            _resolved_model = available[0]
            return _resolved_model
    except Exception as e:
        print(f"  [warn] Model auto-discovery failed ({e}), using fallback")

    _resolved_model = "claude-sonnet-4-5"
    return _resolved_model


# ---------------------------------------------------------------------------
# AI enrichment
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    """Strip characters that break JSON string literals in model output."""
    # Replace smart quotes, dashes, and other unicode punctuation with ASCII equivalents
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", " ": " ",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    # Strip any remaining non-ASCII that could break JSON
    return s.encode("ascii", errors="ignore").decode("ascii").strip()


def _parse_json_lenient(raw: str) -> list:
    """Try strict parse first, then attempt to salvage partial JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw).get("scores", [])
    except json.JSONDecodeError:
        # Try to extract individual score objects with regex
        scores = []
        pattern = r'\{\s*"index"\s*:\s*(\d+)\s*,\s*"stream_scores"\s*:\s*(\{[^}]+\})'
        for m in re.finditer(pattern, raw):
            try:
                idx = int(m.group(1))
                stream_scores = json.loads(m.group(2))
                scores.append({"index": idx, "stream_scores": stream_scores, "reasoning": ""})
            except Exception:
                continue
        return scores


def ai_enrich_batch(contacts: list, streams: list, api_key: Optional[str] = None) -> list:
    """
    Score a batch of contacts via Claude.
    Raises RuntimeError on auth/config failures (fast-fail).
    Returns empty list on JSON parse errors (soft-fail, logs warning).
    """
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. Fix: pip install anthropic")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Fix: $env:ANTHROPIC_API_KEY = \"sk-ant-...\""
        )

    client = anthropic.Anthropic(api_key=key)
    model  = resolve_model(client)

    stream_summary = "\n".join([
        f"- {s['id']}: {s.get('description','')} | targets: "
        f"{[k['term'] for k in s.get('title_signals',[])[:5]]}"
        for s in streams
    ])

    # Sanitize contact strings — special chars in names/titles break JSON output
    contact_lines = []
    for i, c in enumerate(contacts):
        name    = _sanitize(f"{c.get('First Name','')} {c.get('Last Name','')}")
        title   = _sanitize(c.get('Position', c.get('Title', '')))
        company = _sanitize(c.get('Company', ''))
        loc     = _sanitize(c.get('Location', '') or c.get('Geography', ''))
        contact_lines.append(f"{i+1}. {name} | {title} @ {company} | {loc}")
    contact_list = "\n".join(contact_lines)

    prompt = f"""Score each LinkedIn contact for relevance to each context.

Revenue streams:
{stream_summary}

Contacts (index | name | title @ company | location):
{contact_list}

Return ONLY valid JSON. No prose, no markdown fences, no trailing commas.
Use only plain ASCII in the reasoning field - no special characters.
{{
  "scores": [
    {{"index": 1, "stream_scores": {{"stream_id": 0}}, "reasoning": "one line ascii only"}},
    ...
  ]
}}

Score 0=no fit, 100=perfect. Be conservative - most contacts score under 30."""

    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    results = _parse_json_lenient(raw)
    if not results:
        print(f" [warn] JSON parse failed for this batch - skipping")
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def score_connections(
    input_csv: str,
    config_path: str = "config/streams.yaml",
    stream_filter: Optional[str] = None,
    ai_enrich: bool = False,
    top_n: Optional[int] = None,
    min_tier: str = "C",
) -> tuple[list, list]:
    """Returns (results, streams_config)."""
    config  = load_config(config_path)
    streams = config["streams"]

    if stream_filter:
        streams = [s for s in streams if s["id"] == stream_filter]
        if not streams:
            raise ValueError(f"Stream '{stream_filter}' not found in config.")

    tier_order = {"A": 0, "B": 1, "C": 2, "skip": 3}
    min_tier_val = tier_order.get(min_tier, 2)

    # Try UTF-8 first (standard), fall back to latin-1 for Windows LinkedIn exports
    for _enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(input_csv, newline="", encoding=_enc) as f:
                lines = f.readlines()
            # Verify it looks right by checking for mangled chars in first 500 chars
            sample = "".join(lines[:5])
            if "Ã" not in sample:  # latin-1 mojibake signature
                break
        except UnicodeDecodeError:
            continue

    # Find the header row (starts with "First Name")
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("First Name"):
            header_idx = i
            break

    import io
    csv_text = "".join(lines[header_idx:])
    contacts = list(csv.DictReader(io.StringIO(csv_text)))

    print(f"Loaded {len(contacts)} contacts from {input_csv}")

    # AI enrichment — fast-fail on first error so we don't waste time/money
    # on 92 batches if the API key or model is wrong.
    ai_lookup: dict = {}
    if ai_enrich:
        total_batches = (len(contacts) + 24) // 25
        try:
            for i in range(0, len(contacts), 25):
                batch = contacts[i:i+25]
                batch_num = i//25 + 1
                print(f"  AI enriching batch {batch_num}/{total_batches}...", end="", flush=True)
                results_batch = ai_enrich_batch(batch, streams)
                for r in results_batch:
                    ai_lookup[r["index"] - 1 + i] = r
                print(f" done ({len(results_batch)} scored)")
        except RuntimeError as e:
            print(f"[error] AI enrichment stopped: {e}")
            print("[info]  Continuing with keyword-only scoring for all contacts.")
            ai_lookup = {}
        except Exception as e:
            print(f"[error] Unexpected AI enrichment error: {e}")
            print("[info]  Continuing with keyword-only scoring for all contacts.")
            ai_lookup = {}

    results = []
    for idx, contact in enumerate(contacts):
        name     = f"{contact.get('First Name','').strip()} {contact.get('Last Name','').strip()}".strip()
        title    = contact.get("Position", contact.get("Title", "")).strip()
        company  = contact.get("Company", "").strip()
        # LinkedIn export URL field names vary
        li_url   = (contact.get("URL") or contact.get("Profile URL") or
                    contact.get("LinkedIn URL") or contact.get("linkedin_url") or "").strip()
        location = (contact.get("Location") or contact.get("Geography") or "").strip()
        email    = contact.get("Email Address", "").strip()
        connected= contact.get("Connected On", "").strip()

        best_score, best_stream, best_tier = 0, None, "skip"
        stream_scores, stream_hits = {}, {}

        for stream in streams:
            sc = score_contact(contact, stream)
            if idx in ai_lookup:
                ai_sc = ai_lookup[idx].get("stream_scores", {}).get(stream["id"])
                if ai_sc is not None:
                    sc["score"] = int(sc["score"] * 0.4 + ai_sc * 0.6)
                    sc["tier"]  = tier_label(sc["score"])

            stream_scores[stream["id"]] = sc["score"]
            stream_hits[stream["id"]]   = {
                "title_hits":    sc["title_hits"],
                "company_hits":  sc["company_hits"],
                "negative_hits": sc["negative_hits"],
            }
            if sc["score"] > best_score:
                best_score  = sc["score"]
                best_stream = stream["id"]
                best_tier   = sc["tier"]

        if not best_stream or tier_order.get(best_tier, 3) > min_tier_val:
            continue

        results.append({
            "name":        name,
            "title":       title,
            "company":     company,
            "location":    location,
            "li_url":      li_url,
            "email":       email,
            "connected":   connected,
            "best_stream": best_stream,
            "best_score":  best_score,
            "tier":        best_tier,
            "stream_scores": stream_scores,
            "stream_hits":   stream_hits,
            "ai_reasoning":  ai_lookup.get(idx, {}).get("reasoning", ""),
            # flat score columns for CSV
            **{f"score_{sid}": stream_scores.get(sid, 0) for sid in [s["id"] for s in streams]},
        })

    results.sort(key=lambda x: x["best_score"], reverse=True)
    if top_n:
        results = results[:top_n]

    return results, streams


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(results: list, output_path: str, streams: list):
    if not results:
        print("No results to write.")
        return
    # Build flat fieldset
    stream_ids = [s["id"] for s in streams]
    fields = ["name", "title", "company", "location", "li_url", "email",
              "connected", "best_stream", "best_score", "tier",
              *[f"score_{sid}" for sid in stream_ids], "ai_reasoning"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"CSV  → {output_path}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

STREAM_COLORS = {
    0: ("#185FA5", "#E6F1FB", "#0C447C"),
    1: ("#1D9E75", "#E1F5EE", "#085041"),
    2: ("#BA7517", "#FAEEDA", "#412402"),
    3: ("#534AB7", "#EEEDFE", "#26215C"),
    4: ("#D85A30", "#FAECE7", "#4A1B0C"),
    5: ("#D4537E", "#FBEAF0", "#4B1528"),
}

TIER_STYLES = {
    "A": ("background:#EAF3DE;color:#27500A", "Tier A"),
    "B": ("background:#FAEEDA;color:#412402", "Tier B"),
    "C": ("background:#F1EFE8;color:#2C2C2A", "Tier C"),
    "skip": ("background:#FCEBEB;color:#791F1F", "—"),
}


def esc(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def score_bar_html(score: int, color: str) -> str:
    pct = min(100, max(0, score))
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;height:5px;background:#eee;border-radius:3px;min-width:50px">'
        f'<div style="width:{pct}%;height:5px;background:{color};border-radius:3px"></div></div>'
        f'<span style="font-size:12px;font-weight:500;min-width:26px;text-align:right">{score}</span>'
        f'</div>'
    )


def build_row_data(r: dict, streams: list) -> dict:
    """Prepare all row data as JSON-serializable dict for the JS table."""
    tier_style, tier_text = TIER_STYLES.get(r["tier"], ("", r["tier"]))
    li_url  = r.get("li_url", "")
    name_cell = (
        f'<div style="display:flex;flex-direction:column;gap:2px">'
        f'<span style="font-weight:500">{esc(r["name"])}</span>'
        + (f'<a href="{esc(li_url)}" target="_blank" rel="noopener noreferrer" '
           f'style="font-size:11px;color:#185FA5;text-decoration:none" '
           f'title="Open LinkedIn profile">'
           f'<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" style="vertical-align:-1px;margin-right:3px"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>'
           f'LinkedIn</a>'
           if li_url else "") +
        f'</div>'
    )

    stream_badges = ""
    for i, s in enumerate(streams):
        sid   = s["id"]
        sc    = r["stream_scores"].get(sid, 0)
        color, bg, text_color = STREAM_COLORS.get(i % len(STREAM_COLORS), ("#888","#eee","#333"))
        hits  = r["stream_hits"].get(sid, {})
        all_hits = hits.get("title_hits", []) + hits.get("company_hits", [])
        neg_hits = hits.get("negative_hits", [])
        tooltip_parts = []
        if all_hits:  tooltip_parts.append("Matches: " + ", ".join(all_hits))
        if neg_hits:  tooltip_parts.append("Negatives: " + ", ".join(neg_hits))
        tooltip = " | ".join(tooltip_parts) if tooltip_parts else "No signal matches"

        stream_badges += (
            f'<div title="{esc(tooltip)}" style="margin-bottom:4px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">'
            f'<span style="font-size:11px;color:#666">{esc(sid.replace("_"," "))}</span>'
            f'<span style="font-size:11px;font-weight:500">{sc}</span>'
            f'</div>'
            f'<div style="height:4px;background:#eee;border-radius:2px">'
            f'<div style="width:{min(100,sc)}%;height:4px;background:{color};border-radius:2px"></div>'
            f'</div></div>'
        )

    best_color = STREAM_COLORS.get(
        next((i for i, s in enumerate(streams) if s["id"] == r["best_stream"]), 0),
        ("#888","#eee","#333")
    )[0]

    return {
        "name":         r["name"],
        "name_cell":    name_cell,
        "title":        esc(r["title"]),
        "company":      esc(r["company"]),
        "location":     esc(r["location"]),
        "li_url":       esc(r.get("li_url", "")),
        "email":        esc(r.get("email", "")),
        "connected":    esc(r.get("connected", "")),
        "best_stream":  esc(r["best_stream"]),  # keep underscores to match filter dropdown
        "best_score":   r["best_score"],
        "tier":         r["tier"],
        "tier_style":   tier_style,
        "tier_text":    tier_text,
        "score_bar":    score_bar_html(r["best_score"], best_color),
        "stream_bars":  stream_badges,
        "ai_note":      esc(r.get("ai_reasoning", "")),
    }


def write_html(results: list, streams: list, output_path: str, source_file: str = ""):
    rows_data = [build_row_data(r, streams) for r in results]
    stream_ids = [s["id"] for s in streams]
    stream_names = [s["id"].replace("_", " ") for s in streams]

    # Summary stats
    from collections import Counter
    tier_counts   = Counter(r["tier"] for r in results)
    stream_counts = Counter(r["best_stream"] for r in results)
    total = len(results)
    gen_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Build stream color legend
    legend_html = ""
    for i, s in enumerate(streams):
        color = STREAM_COLORS.get(i % len(STREAM_COLORS), ("#888","#eee","#333"))[0]
        count = stream_counts.get(s["id"], 0)
        legend_html += (
            f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{color};display:inline-block"></span>'
            f'{esc(s["id"].replace("_"," "))} <span style="color:#999">({count})</span></div>'
        )

    rows_json = json.dumps(rows_data, ensure_ascii=False)
    streams_json = json.dumps(stream_names, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LinkedIn Network Finder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300&display=swap" rel="stylesheet">
<style>
  :root {{
    --blue:   #185FA5;
    --green:  #1D9E75;
    --amber:  #BA7517;
    --purple: #534AB7;
    --coral:  #D85A30;
    --border: rgba(0,0,0,0.1);
    --bg:     #F7F6F3;
    --surface:#FFFFFF;
    --text:   #1a1a18;
    --muted:  #666;
    --radius: 8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Fraunces', Georgia, serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
  }}
  a {{ color: var(--blue); }}

  /* Header */
  .page-header {{
    background: var(--text);
    color: #fff;
    padding: 2rem 2.5rem 1.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
  }}
  .page-header h1 {{
    font-size: 28px;
    font-weight: 300;
    font-style: italic;
    letter-spacing: -0.5px;
  }}
  .page-header .meta {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    text-align: right;
    line-height: 1.7;
  }}

  /* Stats bar */
  .stats-bar {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
  }}
  .stat {{
    display: flex;
    flex-direction: column;
    gap: 1px;
  }}
  .stat-num {{
    font-size: 22px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    line-height: 1;
  }}
  .stat-lbl {{
    font-size: 11px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: .05em;
  }}
  .stat-div {{
    width: 1px;
    height: 32px;
    background: var(--border);
  }}
  .tier-A {{ color: #3B6D11; }}
  .tier-B {{ color: #854F0B; }}
  .tier-C {{ color: #5F5E5A; }}

  /* Legend */
  .legend-row {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-left: auto;
  }}

  /* Toolbar */
  .toolbar {{
    padding: .75rem 2.5rem;
    display: flex;
    align-items: center;
    gap: .75rem;
    flex-wrap: wrap;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 20;
  }}
  .toolbar input {{
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--text);
    width: 220px;
  }}
  .toolbar input:focus {{ outline: none; border-color: var(--blue); }}
  .toolbar select {{
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
  }}
  .toolbar select:focus {{ outline: none; border-color: var(--blue); }}
  .count-badge {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-left: auto;
  }}

  /* Table */
  .table-wrap {{
    padding: 0 2.5rem 3rem;
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
    background: var(--surface);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: 13px;
  }}
  thead th {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    background: #F7F6F3;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    user-select: none;
    cursor: pointer;
  }}
  thead th:hover {{ background: #EFEDE8; color: var(--text); }}
  thead th.sorted {{ color: var(--blue); }}
  thead th .sort-arrow {{ margin-left: 4px; opacity: 0.5; font-size: 9px; }}
  thead th.sorted .sort-arrow {{ opacity: 1; }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background .1s;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: #FAFAF8; }}
  td {{
    padding: 11px 14px;
    vertical-align: middle;
  }}
  td.name-cell {{ min-width: 180px; }}
  td.title-cell {{ min-width: 180px; color: var(--muted); font-style: italic; }}
  td.company-cell {{ min-width: 140px; font-weight: 500; }}
  td.location-cell {{ min-width: 120px; color: var(--muted); font-size: 12px; }}
  td.stream-cell {{ min-width: 110px; }}
  td.score-cell {{ min-width: 110px; }}
  td.tier-cell {{ min-width: 70px; text-align: center; }}
  td.streams-cell {{ min-width: 180px; }}
  td.connected-cell {{ font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }}
  td.email-cell {{ font-family: 'DM Mono', monospace; font-size: 11px; }}
  td.ai-cell {{ font-size: 11px; color: var(--muted); font-style: italic; max-width: 200px; }}

  .tier-badge {{
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 10px;
    letter-spacing: .04em;
  }}
  .stream-tag {{
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 8px;
    background: #F1EFE8;
    color: #444;
  }}

  /* Empty state */
  .empty {{
    text-align: center;
    padding: 3rem;
    color: var(--muted);
    font-style: italic;
  }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Network matches</h1>
  <div class="meta">
    Source: {esc(Path(source_file).name) if source_file else "LinkedIn export"}<br>
    Generated: {gen_time}<br>
    {total} qualifying contacts
  </div>
</div>

<div class="stats-bar">
  <div class="stat">
    <span class="stat-num">{total}</span>
    <span class="stat-lbl">Total</span>
  </div>
  <div class="stat-div"></div>
  <div class="stat">
    <span class="stat-num tier-A">{tier_counts.get("A",0)}</span>
    <span class="stat-lbl">Tier A</span>
  </div>
  <div class="stat">
    <span class="stat-num tier-B">{tier_counts.get("B",0)}</span>
    <span class="stat-lbl">Tier B</span>
  </div>
  <div class="stat">
    <span class="stat-num tier-C">{tier_counts.get("C",0)}</span>
    <span class="stat-lbl">Tier C</span>
  </div>
  <div class="stat-div"></div>
  <div class="legend-row">{legend_html}</div>
</div>

<div class="toolbar">
  <input type="text" id="search" placeholder="Search name, title, company…" oninput="applyFilters()">
  <select id="tier-filter" onchange="applyFilters()">
    <option value="">All tiers</option>
    <option value="A">Tier A</option>
    <option value="B">Tier B</option>
    <option value="C">Tier C</option>
  </select>
  <select id="stream-filter" onchange="applyFilters()">
    <option value="">All contexts</option>
    {chr(10).join(f'<option value="{esc(sid)}">{esc(sid.replace("_"," "))}</option>' for sid in stream_ids)}
  </select>
  <span class="count-badge" id="count-badge">{total} contacts</span>
</div>

<div class="table-wrap">
<table id="main-table">
<thead>
  <tr>
    <th onclick="sortBy('name')" data-col="name">Name <span class="sort-arrow">↕</span></th>
    <th onclick="sortBy('title')" data-col="title">Title <span class="sort-arrow">↕</span></th>
    <th onclick="sortBy('company')" data-col="company">Company <span class="sort-arrow">↕</span></th>
    <th onclick="sortBy('location')" data-col="location">Location <span class="sort-arrow">↕</span></th>
    <th onclick="sortBy('best_stream')" data-col="best_stream">Best match <span class="sort-arrow">↕</span></th>
    <th onclick="sortBy('best_score')" data-col="best_score" class="sorted">Score <span class="sort-arrow">↓</span></th>
    <th onclick="sortBy('tier')" data-col="tier">Tier <span class="sort-arrow">↕</span></th>
    <th>All contexts</th>
    <th onclick="sortBy('connected')" data-col="connected">Connected <span class="sort-arrow">↕</span></th>
    <th>Email</th>
    <th>AI note</th>
  </tr>
</thead>
<tbody id="tbody"></tbody>
</table>
<div class="empty" id="empty-state" style="display:none">No contacts match your filters.</div>
</div>

<script>
const ALL_ROWS = {rows_json};
const STREAM_NAMES = {streams_json};

let sortCol = "best_score";
let sortAsc = false;
let filtered = [...ALL_ROWS];

const TIER_ORDER = {{A:0, B:1, C:2, skip:3}};
const STREAM_COLORS_JS = {json.dumps({s["id"]: STREAM_COLORS.get(i%len(STREAM_COLORS),("#888","",""))[0] for i,s in enumerate(streams)})};

function tierStyle(t) {{
  const m = {{A:"background:#EAF3DE;color:#27500A", B:"background:#FAEEDA;color:#412402", C:"background:#F1EFE8;color:#2C2C2A", skip:"background:#FCEBEB;color:#791F1F"}};
  return m[t]||"";
}}

function renderRows() {{
  const tbody = document.getElementById("tbody");
  const badge = document.getElementById("count-badge");
  const empty = document.getElementById("empty-state");
  if (!filtered.length) {{
    tbody.innerHTML = "";
    empty.style.display = "";
    badge.textContent = "0 contacts";
    return;
  }}
  empty.style.display = "none";
  badge.textContent = filtered.length + " contact" + (filtered.length===1?"":"s");

  tbody.innerHTML = filtered.map(r => {{
    const streamColor = STREAM_COLORS_JS[r.best_stream] || "#888";
    const liLink = r.li_url
      ? `<a href="${{r.li_url}}" target="_blank" rel="noopener noreferrer" style="font-size:11px;color:#185FA5;text-decoration:none;display:block;margin-top:2px"><svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" style="vertical-align:-1px;margin-right:2px"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>LinkedIn</a>`
      : "";

    const scoreBar = `<div style="display:flex;align-items:center;gap:6px"><div style="width:60px;height:5px;background:#eee;border-radius:3px"><div style="width:${{Math.min(100,r.best_score)}}%;height:5px;background:${{streamColor}};border-radius:3px"></div></div><span style="font-size:12px;font-weight:500;font-family:'DM Mono',monospace">${{r.best_score}}</span></div>`;

    return `<tr>
      <td class="name-cell"><span style="font-weight:500">${{r.name}}</span>${{liLink}}</td>
      <td class="title-cell">${{r.title}}</td>
      <td class="company-cell">${{r.company}}</td>
      <td class="location-cell">${{r.location}}</td>
      <td class="stream-cell"><span class="stream-tag" style="background:${{streamColor}}22;color:${{streamColor}}">${{r.best_stream.replace(/_/g," ")}}</span></td>
      <td class="score-cell">${{scoreBar}}</td>
      <td class="tier-cell"><span class="tier-badge" style="${{tierStyle(r.tier)}}">${{r.tier==="skip"?"—":"Tier "+r.tier}}</span></td>
      <td class="streams-cell"><div style="font-size:11px">${{r.stream_bars}}</div></td>
      <td class="connected-cell">${{r.connected}}</td>
      <td class="email-cell">${{r.email ? `<a href="mailto:${{r.email}}">${{r.email}}</a>` : ""}}</td>
      <td class="ai-cell">${{r.ai_note}}</td>
    </tr>`;
  }}).join("");
}}

function applyFilters() {{
  const q = document.getElementById("search").value.toLowerCase();
  const tier = document.getElementById("tier-filter").value;
  const stream = document.getElementById("stream-filter").value;

  filtered = ALL_ROWS.filter(r => {{
    if (q && !((r.name||"").toLowerCase().includes(q) ||
               (r.title||"").toLowerCase().includes(q) ||
               (r.company||"").toLowerCase().includes(q) ||
               (r.location||"").toLowerCase().includes(q))) return false;
    if (tier && r.tier !== tier) return false;
    if (stream && r.best_stream !== stream) return false;
    return true;
  }});

  doSort();
}}

function sortBy(col) {{
  if (sortCol === col) {{ sortAsc = !sortAsc; }}
  else {{ sortCol = col; sortAsc = col !== "best_score"; }}
  document.querySelectorAll("thead th").forEach(th => {{
    th.classList.remove("sorted");
    const arrow = th.querySelector(".sort-arrow");
    if (arrow) arrow.textContent = "↕";
  }});
  const active = document.querySelector(`thead th[data-col="${{sortCol}}"]`);
  if (active) {{
    active.classList.add("sorted");
    const arrow = active.querySelector(".sort-arrow");
    if (arrow) arrow.textContent = sortAsc ? "↑" : "↓";
  }}
  doSort();
}}

function doSort() {{
  const tierOrd = {{A:0, B:1, C:2, skip:3}};
  filtered.sort((a, b) => {{
    let av = a[sortCol] ?? "", bv = b[sortCol] ?? "";
    if (sortCol === "tier") {{ av = tierOrd[av]??9; bv = tierOrd[bv]??9; }}
    if (sortCol === "best_score") {{ av = Number(av); bv = Number(bv); }}
    let cmp = typeof av === "number" ? av - bv : String(av).localeCompare(String(bv));
    return sortAsc ? cmp : -cmp;
  }});
  renderRows();
}}

doSort();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML → {output_path}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_scored_csv(csv_path: str, config_path: str) -> tuple[list, list]:
    """
    Load a previously scored CSV and reconstruct results + streams for HTML regeneration.
    Used by --from-csv to avoid re-scoring (and re-spending on --ai-enrich).
    """
    config = load_config(config_path)
    streams = config["streams"]
    stream_ids = [s["id"] for s in streams]

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        # Reconstruct stream_scores dict from flat score_ columns
        stream_scores = {}
        for sid in stream_ids:
            val = row.get(f"score_{sid}", 0)
            try:
                stream_scores[sid] = int(float(val))
            except (ValueError, TypeError):
                stream_scores[sid] = 0

        # Reconstruct stream_hits as empty (not stored in CSV, not needed for HTML display)
        stream_hits = {sid: {"title_hits": [], "company_hits": [], "negative_hits": []} for sid in stream_ids}

        results.append({
            "name":          row.get("name", ""),
            "title":         row.get("title", ""),
            "company":       row.get("company", ""),
            "location":      row.get("location", ""),
            "li_url":        row.get("li_url", ""),
            "email":         row.get("email", ""),
            "connected":     row.get("connected", ""),
            "best_stream":   row.get("best_stream", ""),
            "best_score":    int(float(row.get("best_score", 0))),
            "tier":          row.get("tier", "C"),
            "stream_scores": stream_scores,
            "stream_hits":   stream_hits,
            "ai_reasoning":  row.get("ai_reasoning", ""),
        })

    return results, streams


def main():
    parser = argparse.ArgumentParser(
        description="Find relevant people in your LinkedIn network across match contexts"
    )

    # Main input — mutually exclusive: either score from raw LI export, or regenerate HTML from scored CSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input",    help="LinkedIn connections CSV export (scores from scratch)")
    input_group.add_argument("--from-csv", help="Previously scored CSV — regenerates HTML only, no re-scoring or API calls",
                             metavar="SCORED_CSV")

    parser.add_argument("--output",    default="scored_connections.csv", help="Output CSV path")
    parser.add_argument("--html",      default="",     help="HTML report path (default: auto-named next to CSV)")
    parser.add_argument("--no-html",   action="store_true", help="Skip HTML report generation")
    parser.add_argument("--config",    default="config/streams.yaml")
    parser.add_argument("--stream",    help="Filter to a single stream ID")
    parser.add_argument("--top",       type=int,       help="Top N results only")
    parser.add_argument("--min-tier",  default="C",    choices=["A","B","C"])
    parser.add_argument("--ai-enrich", action="store_true",
                        help="Use Claude AI for fuzzy scoring (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--json",      action="store_true", help="Also print JSON to stdout")
    args = parser.parse_args()

    # --from-csv path: load existing scored CSV, skip all scoring
    if args.from_csv:
        print(f"Loading scored CSV: {args.from_csv}")
        results, streams = load_scored_csv(args.from_csv, args.config)
        print(f"Loaded {len(results)} contacts from scored CSV")
        html_path = args.html or str(Path(args.from_csv).with_suffix(".html"))
        write_html(results, streams, html_path, source_file=args.from_csv)
        return

    # --input path: full scoring pipeline
    results, streams = score_connections(
        input_csv=args.input,
        config_path=args.config,
        stream_filter=args.stream,
        ai_enrich=args.ai_enrich,
        top_n=args.top,
        min_tier=args.min_tier,
    )

    print(f"\nFound {len(results)} qualifying contacts")
    if results:
        from collections import Counter
        print(f"Tiers:   {dict(Counter(r['tier'] for r in results))}")
        print(f"Streams: {dict(Counter(r['best_stream'] for r in results))}")

    write_csv(results, args.output, streams)

    if not args.no_html:
        html_path = args.html or str(Path(args.output).with_suffix(".html"))
        write_html(results, streams, html_path, source_file=args.input)

    if args.json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
