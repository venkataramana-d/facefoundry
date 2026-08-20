# Setup — do this once (your part)

I can't create the account for you (it needs your phone for verification), so these ~10 minutes are yours. Once done, I take over.

## Step 1 — Create a Kaggle account (~3 min)
1. Go to **https://kaggle.com** → **Register** → sign up (email or Google).
2. Confirm your email.

## Step 2 — Verify phone to unlock GPU (~2 min) — REQUIRED
1. Top-right avatar → **Settings**.
2. Find **Phone Verification** → enter your number → enter the SMS code.
3. Without this, Kaggle blocks all GPU. Non-negotiable.

## Step 3 — Generate your API token (~1 min)
1. Still in **Settings** → scroll to the **API** section.
2. Click **Create New Token**.
3. A file named **`kaggle.json`** downloads. It looks like:
   ```json
   {"username":"yourname","key":"xxxxxxxxxxxxxxxxxxxx"}
   ```

## Step 4 — Put the token where the tool can find it
On Windows, copy `kaggle.json` to:
```
C:\Users\Ramana\.kaggle\kaggle.json
```
(Create the `.kaggle` folder if it doesn't exist.)

## Step 5 — Tell me "done"
Reply **done** and I'll run the Phase 0 validation script (`worker/phase0_validate.py`) to confirm we can drive Kaggle's free GPU automatically. If that passes, we build the rest.

---

### What NOT to do
- Don't share the contents of `kaggle.json` in chat — it's a secret key. Just place the file at the path above; the tool reads it locally.
