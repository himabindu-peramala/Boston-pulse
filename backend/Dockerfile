# Boston Pulse Backend — Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/

ENV PORT=8080
EXPOSE 8080

# Run Flask app with gunicorn (WSGI server)
# Workers = 2 * CPU + 1, using 2 for Cloud Run's single vCPU default
CMD ["sh", "-c", "gunicorn app.main:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120"]
