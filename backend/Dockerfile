FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Generate data + train models at build time
RUN mkdir -p data/models && \
    python -m app.data.generate_synthetic && \
    python -m app.ml.train

# Railway provides PORT env var
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
