FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Chromium is installed (Playwright image usually has it, but this is safe)
RUN python -m playwright install chromium

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["bash","-lc","uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
