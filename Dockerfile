# FaceFoundry control panel — deployable to any persistent host (Render, Railway,
# Fly.io, a VM). NOT for Vercel/serverless: the app runs long-lived background
# threads, writes SQLite + files, and shells out to the Kaggle CLI.
FROM python:3.12-slim

WORKDIR /app

# system deps kept minimal; the heavy ML runs on Kaggle, not here
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root runtime. The app writes to jobs/ and ~/.kaggle — pre-create and chown
# so it never needs root to bootstrap.
RUN groupadd -r ff && useradd -r -g ff -d /app -s /sbin/nologin ff \
 && mkdir -p /app/jobs /home/ff/.kaggle \
 && chown -R ff:ff /app /home/ff
USER ff
ENV HOME=/home/ff

# Kaggle credentials come from env at runtime (KAGGLE_USERNAME / KAGGLE_KEY).
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
