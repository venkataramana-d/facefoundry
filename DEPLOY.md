# Deploying FaceFoundry

## ⚠️ Vercel will NOT run this app
FaceFoundry is a long-running FastAPI server: it polls Kaggle jobs on background
threads for 10–20 min, writes SQLite + files under `jobs/`, and shells out to the
Kaggle CLI. Vercel (and any serverless platform) runs short-lived, stateless
functions with an ephemeral read-only filesystem — so a Vercel deploy will build
but the app will fail the moment a job starts. Use a **persistent host** instead.

## Recommended: Render (or Railway / Fly.io / a VM)
This repo includes a `Dockerfile` and `render.yaml`.

**Render (easiest):**
1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo (it reads `render.yaml`).
3. In the service **Environment**, set:
   - `KAGGLE_USERNAME` — your Kaggle username
   - `KAGGLE_KEY` — your Kaggle API key (from `kaggle.json`)
   - `FACEFOUNDRY_PASSWORD` — (optional) site password to gate the UI
   - `FACEFOUNDRY_USER` — (optional) username to pair with the password
     (defaults to `team`)
4. Deploy. The app comes up at `https://<name>.onrender.com`.

The blueprint uses Render's **free tier** — no card required, no persistent
disk. That means:

- The service **sleeps after ~15 min of inactivity** and cold-starts on the next
  request (~30 s).
- `jobs/` and the SQLite DB **reset on every restart** — finished jobs vanish
  from history. Download approved headshots before the service sleeps.
- Health checks hit `/healthz`, which bypasses the password gate.

For always-on with persistent history, upgrade the plan to `starter` and attach
a disk mounted at `/app/jobs` in `render.yaml`.

**Railway / Fly.io:** point them at the `Dockerfile` and set the same env vars.

**A VM (DigitalOcean/EC2):**
```bash
git clone <repo> && cd facefoundry
pip install -r requirements.txt
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Credentials
The app reads Kaggle creds from `~/.kaggle/kaggle.json` **or**, if that's absent,
from `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars (which it writes to `kaggle.json` on
startup). Never commit `kaggle.json` — it's in `.gitignore`.

New Kaggle `KGAT_` tokens are handled automatically and passed to the CLI via
`KAGGLE_API_TOKEN`.

## Password gate
If `FACEFOUNDRY_PASSWORD` is set, every route except `/healthz` requires HTTP
basic auth with that password and the `FACEFOUNDRY_USER` username. Leave the env
var unset to run the app open (fine for `localhost`, not for a public URL).

## Not committed (see .gitignore)
`jobs/` (employee photos, outputs, DB), `kaggle.json`, `*.db`, caches. Keep it that
way — those contain personal data and must not be public.
