FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# SQLite DB lives in a mounted volume (see DATABASE_URL / compose).
EXPOSE 8000

# Single process: the app runs an in-process background scheduler, so do NOT
# add --workers (that would start duplicate schedulers).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
