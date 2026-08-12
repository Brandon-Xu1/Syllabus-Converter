FROM python:3.12-slim

# tesseract + poppler enable the OCR fallback for scanned PDFs
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ocr.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Single worker: SQLite writes and the in-memory rate limiter assume one
# process. Scale with threads; move to Postgres/Redis before adding workers.
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 8 app:app
