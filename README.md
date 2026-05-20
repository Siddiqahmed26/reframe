---
title: reframe
emoji: 📄
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Reframe your resume for any role. Same fonts, same layout, only the words change.
---

# reframe

> Reframe your resume for any role. A multi-agent pipeline rewrites your bullets to match the job description, in the exact same layout you spent hours formatting.

(Previously: Tailorlock, Resume Match AI.)

A multi-agent product that takes a resume (.docx or .pdf) and a job description, then produces a tailored resume **in the exact same visual format as the original** — same fonts, sizes, margins, colors, layout. Only the words change. Optionally also generates a matching cover letter.

## Architecture

Five specialist agents coordinated by an orchestrator, exposed as MCP tools and wrapped in a FastAPI web service.

```
                          ┌────────────────────────┐
                          │   Frontend (HTML/JS)   │
                          │ upload .docx + paste JD│
                          └───────────┬────────────┘
                                      │ HTTP
                          ┌───────────▼────────────┐
                          │   FastAPI backend      │
                          │   /tailor  /cover      │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │     Orchestrator       │
                          │ (claude-sonnet-4-6)    │
                          └───────────┬────────────┘
                                      │ MCP tool calls
        ┌─────────────────┬───────────┼───────────┬───────────────┐
        ▼                 ▼           ▼           ▼               ▼
  JD Analyzer    Resume Analyzer   Matcher    Rewriter      ATS Optimizer
  (extract       (extract           (gap     (truthful      (keyword
   keywords,      content,          analysis) rewrite        density,
   skills,        sections)                   per bullet)    score)
   seniority)

  + CoverLetter agent (generates matched cover letter)

  Format-preserving DOCX writer puts rewritten text back into
  the original document, preserving every formatting attribute.
```

## What you need to do when you wake up

1. **Get an Anthropic API key** from https://console.anthropic.com/
2. **Copy `.env.example` to `.env`** and paste your key in:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. **Install + run** (two options):

   **Option A — local Python:**
   ```bash
   cd resume-match-ai
   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
   Then open http://localhost:8000

   **Option B — Docker:**
   ```bash
   docker compose up --build
   ```
   Then open http://localhost:8000

4. **Push to GitHub** (I didn't do this for you — needs your account):
   ```bash
   cd resume-match-ai
   # If a stale .git directory exists from the sandbox, remove it first:
   rm -rf .git              # Linux/macOS
   # rmdir /s /q .git       # Windows cmd
   git init -b main && git add . && git commit -m "Initial commit: Resume Match AI"
   gh repo create resume-match-ai --public --source=. --push
   # OR manually create on github.com then:
   # git remote add origin git@github.com:<your-username>/resume-match-ai.git
   # git push -u origin main
   ```

5. **Host it.** Easiest options:
   - **Railway / Render / Fly.io** — point at the repo, set `ANTHROPIC_API_KEY` env var, deploy. The Dockerfile is ready.
   - **Self-host on a VPS** — see [Deploy](#deploy) below.

## Deploy

The stack includes a Caddy reverse proxy that provisions Let's Encrypt TLS certificates automatically. To go live:

1. Point a DNS **A record** for your domain at the server's public IP.
2. Open ports **80** and **443** on the server's firewall.
3. Set in `.env`:
   ```
   DOMAIN=reframe.your-domain.com
   ACME_EMAIL=you@your-domain.com
   CORS_ORIGINS=https://reframe.your-domain.com
   ALLOWED_HOSTS=reframe.your-domain.com
   ```
4. Bring it up:
   ```bash
   docker compose up -d --build
   ```
   On the first HTTPS request Caddy will obtain a certificate from Let's Encrypt and cache it in the `caddy_data` volume.

5. Verify:
   ```bash
   curl -I https://reframe.your-domain.com/health
   # expect: HTTP/2 200, plus Strict-Transport-Security, X-Content-Type-Options, CSP, etc.
   curl -I http://reframe.your-domain.com
   # expect: 308 redirect to https
   ```

The `api` container is NOT exposed to the public network. Only Caddy listens on 80/443; it reverse-proxies internally to `api:8000` over the docker network.

### Production deployment — provider keys for maximum uptime

For best uptime, set **all** of these in your Space Secrets (or VPS `.env`) so the auto-fallback has options to rotate through:

```bash
LLM_PROVIDER=auto
GROQ_API_KEY=gsk_...        # free, no card, generous quotas — console.groq.com
GEMINI_API_KEY=AIza...      # free, no card — aistudio.google.com/apikey
XAI_API_KEY=xai-...         # free credits on signup — console.x.ai
# Optional paid fallbacks
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Recommended models (free tier, large daily quotas):

```bash
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_FAST_MODEL=gemini-2.5-flash-lite
XAI_MODEL=grok-3
XAI_FAST_MODEL=grok-3-mini
```

Daily-quota + fallback tuning knobs (defaults shown):

```bash
SHARED_DAILY_QUOTA=5              # per-IP /tailor runs on the shared pool
BYOK_FALLBACK_TO_SHARED=true      # BYOK quota-out falls through silently
MAX_REQUEST_SECONDS=90            # per-request wall budget for graceful skips
PROVIDER_COOLDOWN_QUOTA_S=21600   # 6h after a daily-quota hit
PROVIDER_COOLDOWN_TRANSIENT_S=60  # 60s after a 429 / 5xx
PROVIDER_COOLDOWN_AUTH_S=86400    # 24h after auth/billing failure
```

How it works at runtime: every request picks the healthiest provider; failed providers cool down for the right duration based on the failure category; quota-exhausted Gemini stays cold for 6 hours, then re-enters the rotation; auth errors disable a provider for 24 hours (the key likely needs the operator's attention). Healthy providers rotate per request to spread load. Watch `GET /health` for live provider state.


**Security defaults applied:**
- HSTS with `preload` and `includeSubDomains`, 1 year.
- Strict CSP allowing only `cdnjs.cloudflare.com` for scripts and Google Fonts for stylesheets.
- `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` disabling camera, mic, geolocation.
- Rate limits per IP: `/tailor` 10/min and 60/hour, `/cover-letter` 20/min, `/download/*` 60/min, `/health` unlimited.
- Upload guard: 415 on unsupported content types, 413 over the configured size limit, filename sanitization.
- `TrustedHostMiddleware` rejects requests with a Host header outside `ALLOWED_HOSTS`.

## Local dev

If you don't want Caddy locally, bring up only the api service:
```bash
docker compose up api
# OR a one-off bind to localhost:
docker compose run --rm --service-ports api
```
Open http://localhost:8000.

You can also run uvicorn directly without Docker:
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## How it works (the agent flow)

1. User uploads `.docx` (preferred) or `.pdf` and pastes a job description.
2. **JD Analyzer** extracts required skills, must-have keywords, nice-to-haves, seniority level, and culture signals.
3. **Resume Analyzer** parses the resume's logical structure (header, summary, experience bullets, skills, education, projects).
4. **Matcher** computes a gap analysis — which JD requirements the resume already covers, which are weakly covered, which are missing.
5. **Rewriter** rewrites each bullet truthfully to surface the relevant skills and keywords. It is instructed never to invent experience.
6. **ATS Optimizer** verifies keyword coverage and pushes a final pass if any must-have keywords are still missing.
7. **DOCX Writer** writes the rewritten text back into the original document, replacing run-level text while preserving paragraph styles, fonts, sizes, colors, margins, and section/page layout.
8. (Optional) **Cover Letter** agent generates a tailored cover letter referencing specific JD points.

## Format preservation — how

The trick is that `.docx` is a zip of XML. `python-docx` exposes paragraphs and *runs* (formatting spans inside a paragraph). The writer:

- Reads the full text per paragraph and asks the rewriter for a 1:1 rewritten version.
- When writing back, it places the new text inside the **first run** of the original paragraph (which carries the inline formatting), and clears the text of subsequent runs while leaving their style objects intact.
- Paragraph-level properties (alignment, spacing, indents, bullet markers, section breaks, page margins) are untouched.

Result: the rewritten document is byte-for-byte identical in formatting to the original, only the text content of bullets and summary lines differs. Section headers like "Experience" and "Education" are detected and left alone.

> **PDF input note:** if you upload a `.pdf`, the system extracts text but the output is a `.docx` because round-tripping arbitrary PDF layout perfectly is not solvable in the general case. For exact format preservation, upload `.docx`.

## MCP server

The agents are also exposed as standalone MCP tools in `mcp_server/server.py`. You can plug this server into Claude Desktop or any MCP-compatible client and call tools like `analyze_jd`, `rewrite_resume`, `generate_cover_letter` directly. See `mcp_server/README.md` inside that folder.

## Project layout

```
resume-match-ai/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── main.py            FastAPI app, endpoints
│   ├── config.py
│   ├── models.py          Pydantic models
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── jd_analyzer.py
│   │   ├── resume_analyzer.py
│   │   ├── matcher.py
│   │   ├── rewriter.py
│   │   ├── ats_optimizer.py
│   │   └── cover_letter.py
│   ├── parsers/
│   │   ├── docx_parser.py
│   │   └── pdf_parser.py
│   └── writers/
│       └── docx_writer.py
├── mcp_server/
│   └── server.py          FastMCP server exposing agents as tools
├── frontend/
│   └── index.html         Upload form + JD textarea + download
└── tests/
    └── test_pipeline.py
```

## Cost ballpark

Each tailoring run makes ~5–7 Claude calls (one per agent + orchestration). With Claude Sonnet 4.6 at current pricing that's roughly 5–15 cents per resume. Cover letter adds about 2–4 cents.

## Limitations / known issues

- PDF input → DOCX output (can't perfectly round-trip arbitrary PDFs).
- Heavy use of tables in resumes is supported but mid-cell inline formatting changes may not be preserved exactly.
- The rewriter is instructed never to invent experience, but always review output before sending.
- No login / multi-tenancy yet — single-user app. Easy to add Auth0 or Clerk if you want.

## License

MIT. Yours to do whatever with.
