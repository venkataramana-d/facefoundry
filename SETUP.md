# Setup - do this once

Kaggle needs your phone number to unlock GPU access, so the account
setup ~10 minutes are yours. Once done, the app takes it from there.

## Step 1 - Create a Kaggle account (~3 min)
1. Go to **https://kaggle.com** → **Register** → sign up (email or Google).
2. Confirm your email.

## Step 2 - Verify phone to unlock GPU (~2 min) - REQUIRED
1. Top-right avatar → **Settings**.
2. Find **Phone Verification** → enter your number → enter the SMS code.
3. Without this, Kaggle blocks all GPU. Non-negotiable.

## Step 3 - Generate your API token (~1 min)
1. Still in **Settings** → scroll to the **API** section.
2. Click **Create New Token**.
3. A file named **`kaggle.json`** downloads. It looks like:
   ```json
   {"username":"yourname","key":"xxxxxxxxxxxxxxxxxxxx"}
   ```
   (Newer accounts hand out `KGAT_…` tokens - those work too; the app handles
   both formats automatically.)

## Step 4 - Put the token where the tool can find it

Pick **one** of these:

**A) Drop `kaggle.json` in the standard location** (recommended for local use):

- **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
  (e.g. `C:\Users\<you>\.kaggle\kaggle.json` - create the `.kaggle` folder if
  it doesn't exist)
- **macOS / Linux:** `~/.kaggle/kaggle.json`

**B) Use environment variables** (recommended for hosted / Docker setups):

```bash
export KAGGLE_USERNAME=yourname
export KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxx
```

The app checks `~/.kaggle/kaggle.json` first, then falls back to
`KAGGLE_USERNAME` / `KAGGLE_KEY`. Deploying to Render / Railway / Fly?
See [DEPLOY.md](DEPLOY.md).

## Step 5 - Launch the app

```bat
run.bat
```

(or `pip install -r requirements.txt` then
`python -m uvicorn app.server:app --port 8000`)

Open <http://localhost:8000> and create a job. First run of the day on Kaggle
spends ~10-15 min downloading models; later jobs in the same session are fast.

---

### What NOT to do
- Don't share the contents of `kaggle.json` in chat or commit it to git - it's
  a secret key. It's already in `.gitignore`; leave it that way.
- Don't skip phone verification. Kaggle refuses to allocate a GPU without it,
  and the worker will fail before it starts generating.
