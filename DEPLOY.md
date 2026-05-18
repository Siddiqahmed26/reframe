# Deploying reframe

Two pieces:

1. **Hugging Face Spaces** hosts the backend (FastAPI + the agent pipeline) plus the embedded frontend. TLS terminated by HF. URL pattern: `https://<user>-reframe.hf.space`.
2. **Vercel** (optional) hosts a static mirror of the same frontend pointing at the HF Space API. Useful for a custom domain or faster edge serving of the page itself.

Do step 1 first. Step 2 is optional.

---

## Step 1. Hugging Face Spaces (backend + frontend)

### Prereqs

- Free Hugging Face account: <https://huggingface.co/join>
- An HF access token with `write` scope: <https://huggingface.co/settings/tokens>
- `.env` values handy: `LLM_PROVIDER`, plus the `*_API_KEY` of whichever provider you're using
- Git installed locally

### 1.1 Create the Space

1. <https://huggingface.co/new-space>
2. Owner: your username
3. Space name: `reframe` (this becomes part of the URL)
4. License: `mit`
5. Space SDK: **Docker**
6. Docker template: **Blank**
7. Hardware: **CPU basic (free)**
8. Visibility: **Public**
9. Click **Create Space**

You'll land on an empty Space with git clone instructions.

### 1.2 Initialize git locally and push

From the project root:

```bash
git init -b main
git add -A
git commit -m "Initial reframe deploy"

# Add HF Space as remote (replace YOUR-USERNAME)
git remote add space https://huggingface.co/spaces/YOUR-USERNAME/reframe
git push space main
```

When prompted, use your HF username + the **access token** as the password.

The Space reads the `sdk: docker` line in the README front-matter and starts building the Dockerfile automatically. First build is ~3 to 5 minutes.

### 1.3 Set secrets in the Space UI

1. Open the Space, click **Settings** (gear icon)
2. Scroll to **Variables and secrets**
3. Click **New secret** for each that applies:

   | Name | Value |
   |---|---|
   | `LLM_PROVIDER` | `groq` or `anthropic` or `xai` |
   | `GROQ_API_KEY` | `gsk_...` (if using Groq) |
   | `GROQ_MODEL` | `openai/gpt-oss-120b` or whichever |
   | `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` |
   | `ANTHROPIC_API_KEY` | `sk-ant-...` (if using Anthropic) |
   | `XAI_API_KEY` | `xai-...` (if using xAI) |
   | `ALLOWED_HOSTS` | `*` |
   | `CORS_ORIGINS` | `*` for now, narrow once Vercel is live |

   Use **Secrets** (encrypted), not Variables.

4. Click **Restart** at the top of the Space to apply.

### 1.4 Open the Space

URL pattern:

```
https://YOUR-USERNAME-reframe.hf.space
```

If it loads and `/health` returns 200, you're live. Write that URL down, you need it for Vercel.

---

## Step 2. Vercel mirror (optional)

The Vercel deploy serves only the static `frontend/index.html` and points its API calls at the HF Space. The frontend reads `<meta name="reframe-api">` to find the API origin.

### 2.1 Update the meta tag

In [frontend/index.html](frontend/index.html), find the `reframe-api` meta tag and set its content to your HF Space URL:

```html
<meta name="reframe-api" content="https://YOUR-USERNAME-reframe.hf.space" />
```

Commit:

```bash
git add frontend/index.html
git commit -m "Point frontend at HF Space API"
```

### 2.2 Tighten CORS on the HF Space

Once you know the Vercel URL (you can predict it: `https://reframe-<random>.vercel.app` for previews and your custom domain for production), update the HF Space secret:

- `CORS_ORIGINS` = `https://reframe.vercel.app,https://your-custom-domain.com`
- `ALLOWED_HOSTS` = `YOUR-USERNAME-reframe.hf.space`

Restart the Space.

### 2.3 Deploy to Vercel

Two paths.

**Via web UI (easiest)**:

1. Push the repo to GitHub (`gh repo create reframe --public --source=. --push` or your usual flow)
2. <https://vercel.com/new> imports it
3. Framework preset: **Other**
4. Build/output settings are picked up from [vercel.json](vercel.json) (no build, serves `frontend/`)
5. No env vars needed for the static deploy
6. Deploy

**Via CLI**:

```bash
npm i -g vercel
vercel login
vercel --prod
```

Vercel reads `vercel.json` and serves `frontend/` as a static site with security headers. The browser fetches `/tailor`, `/health`, etc. directly from the HF Space because the frontend's `apiUrl()` prefixes the meta URL.

### 2.4 Verify

```bash
curl -I https://reframe.vercel.app
# expect 200, X-Frame-Options DENY, nosniff, Referrer-Policy

curl -I https://YOUR-USERNAME-reframe.hf.space/health
# expect 200
```

Then open the Vercel URL in a browser, submit a real resume + JD, watch the request go to the HF Space.

---

## Updating after changes

```bash
git add -A
git commit -m "your message"

# push to both
git push space main       # HF Space rebuilds
git push origin main      # GitHub triggers Vercel redeploy
```

---

## Local dev still works

Nothing in the deploy flow breaks local dev. Without the meta tag set, the frontend falls back to same-origin (relative paths), so:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

still serves both at `http://localhost:8000`.

---

## Free-tier characteristics

| | Hugging Face Spaces (free) | Vercel (Hobby) |
|---|---|---|
| Cold start | ~30s after idle | n/a, static |
| RAM | 16 GB | n/a |
| Persistent storage | None. Resumes are ephemeral. | n/a |
| Concurrency | 1 container, requests queue | High, edge CDN |
| Custom domain | No on free tier | Yes |
| Request timeout | None imposed beyond CPU runtime | n/a for static (the browser talks to HF directly) |
| Rate limits | Inherit from your LLM provider | n/a |
