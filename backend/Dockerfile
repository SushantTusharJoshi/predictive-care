FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD uvicorn app.main_pg:app --host 0.0.0.0 --port ${PORT:-8000}
