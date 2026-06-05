# linkedin-network-finder

Find the right people in your LinkedIn network — for anything.

Define what you're looking for and the tool surfaces your best matches. Promoting a webinar? Inviting people to an event? Looking for collaborators, beta testers, or someone to connect with a speaker? It works for any situation where you need to find specific people among hundreds of connections.

You describe what you're looking for in plain English. Claude generates the matching logic. The tool does the rest — producing a sortable, filterable HTML report with direct LinkedIn profile links, relevance scores, and priority tiers.

## Quick start

```bash
# Install dependencies
pip install pyyaml anthropic

# Step 1 — define what you're looking for (opens in browser)
python stream_builder.py

# Step 2 — find your matches
python scorer.py --input connections.csv
```

This produces two files:
- `scored_connections.csv` — flat data with LinkedIn URLs, all match scores
- `scored_connections.html` — interactive report, open in any browser

## Getting your LinkedIn connections

1. **LinkedIn → Settings → Data Privacy → Get a copy of your data**
2. Select **"Want something in particular?"** then check **Connections** only
3. Request the archive — usually arrives within minutes as a small download

The export includes: `First Name`, `Last Name`, `URL`, `Email Address`,
`Company`, `Position`, `Connected On`. Note: LinkedIn's export includes
a short disclaimer at the top of the CSV — the tool handles this automatically.

## Step 1: Define what you're looking for

```bash
python stream_builder.py
# Opens http://localhost:8764 in your browser
```

A guided interview — 12 questions across 5 steps per context. You describe your ideal person in plain English; Claude generates the keyword signals and relevance weights automatically. When you're done, click Export to save your config.

**What it asks:**

*Step 1 — Basics*
1. What do you call this context? (short label)
2. Describe what you're looking for in plain English — who, what, rough deal size or scope

*Step 2 — Ideal person*
3. What job titles are the best fit? (add multiple variants)
4. What types of organizations do they work in?
5. What pain or trigger makes them ready to engage?

*Step 3 — Past wins*
6. Name 3–5 past matches or dream examples (helps Claude learn the pattern)
7. What did those have in common?

*Step 4 — Dead ends*
8. What looks like a fit but never works out?
9. Any roles or organizations to explicitly exclude?

*Step 5 — Context*
10. Typical deal size or engagement scope (click to select)
11. How long does it take from first contact to agreement?
12. One-time, project-based, or recurring?
13. Anything else useful — timing, seasonal patterns, industries that respond well

```bash
# Custom output path or port:
python stream_builder.py --output config/my_contexts.yaml --port 8765
```

## Step 2: Run the matcher

```bash
# Basic run — produces CSV and HTML report
python scorer.py --input connections.csv

# Focus on one context only
python scorer.py --input connections.csv --stream my_context

# Show only strong matches (tier A and B)
python scorer.py --input connections.csv --min-tier B

# Top 50 results only
python scorer.py --input connections.csv --top 50

# AI-assisted matching — better at ambiguous titles (needs Anthropic API key)
export ANTHROPIC_API_KEY=sk-ant-...
python scorer.py --input connections.csv --ai-enrich

# Custom output paths
python scorer.py --input connections.csv --output out/matches.csv --html out/report.html

# CSV only, skip HTML
python scorer.py --input connections.csv --no-html

# Regenerate HTML from a previously scored CSV — no re-scoring, no API calls
python scorer.py --from-csv scored_connections.csv
```

## The HTML report

Open it in any browser — no server needed, fully self-contained.

- **Sortable columns** — click any header to sort; click again to reverse
- **Live search** — filter by name, title, company, or location instantly
- **Priority filter** — show only A, B, or C tier matches
- **Context filter** — focus on a single context
- **LinkedIn links** — direct profile link on every row, opens in new tab
- **Email links** — `mailto:` links where available in the export
- **Match bars** — visual relevance bar per context
- **Priority badges** — color-coded A / B / C
- **Summary stats** — totals by tier and context in the header
- **AI notes** — reasoning column when `--ai-enrich` was used

## Output columns

| Column | Description |
|--------|-------------|
| `name` | Full name |
| `title` | Job title |
| `company` | Company |
| `location` | Location if available |
| `li_url` | LinkedIn profile URL |
| `email` | Email if included in export |
| `connected` | Date connected |
| `best_stream` | Highest-scoring context (column name kept as-is for CSV compatibility) |
| `best_score` | Relevance score 0–100 |
| `tier` | A / B / C |
| `score_{context_id}` | Per-context score columns |
| `ai_reasoning` | AI note if `--ai-enrich` used |

## Priority tiers

| Tier | Score | What it means |
|------|-------|---------------|
| A | 70–100 | Strong match — worth reaching out to directly |
| B | 40–69 | Good signal — worth a look |
| C | 15–39 | Weak signal — lower priority |

## AI-assisted matching

Without `--ai-enrich`, matching is keyword-based and runs entirely locally — fast,
free, and good for clear titles. With `--ai-enrich`, Claude scores each contact
using context and reasoning, which helps with ambiguous or non-standard titles.

The tool auto-selects the best available model from your API account — no model
names to track or update. Set `ANTHROPIC_MODEL` in your environment to pin a
specific version if needed.

Approximate cost for AI enrichment: under $1 for 2,000+ connections.

## Config format

Your config lives in `config/contexts.yaml` (gitignored by default — see Privacy).
The public repo ships `config/contexts.example.yaml` as a starting point.

Each entry defines one context — a group of people you're trying to find:

```yaml
contexts:
  - id: workshop_attendees
    description: "People who'd benefit from a leadership workshop"
    max_score: 10

    title_signals:
      - { term: "director of people", weight: 6 }
      - { term: "head of hr",         weight: 6 }
      - { term: "vp people",          weight: 5 }

    company_signals:
      - { term: "saas",  weight: 2 }
      - { term: "scale", weight: 2 }

    negative_signals:
      - { term: "staffing", weight: 6 }

    outreach_template: templates/workshop_attendees.md
```

**max_score** calibration: set it just above the highest raw score a perfect
match would realistically get. The tool normalizes raw scores against this
ceiling — so getting it roughly right matters. The context builder sets it
automatically; you can tune it manually after seeing your results.

## Privacy

- `config/contexts.yaml` is gitignored — your matching logic stays on your machine
- `*.csv` files are gitignored — LinkedIn exports contain personal data
- `--ai-enrich` sends job title and company name to the Anthropic API only — no names, emails, or URLs
- The stream builder sends your interview answers to the Anthropic API to generate signals; they are not stored or used for training

## License

MIT
