FROM python:3.11-slim

WORKDIR /app

# Copy shared packages
COPY packages/rag /app/packages/rag
COPY packages/calendar_tools /app/packages/calendar_tools

# Copy voice app
COPY apps/voice /app/apps/voice

# Install dependencies
RUN pip install --no-cache-dir -r /app/apps/voice/requirements.txt

WORKDIR /app/apps/voice

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
