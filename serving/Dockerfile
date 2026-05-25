# 9.1 Base image
FROM python:3.11-slim

WORKDIR /app

# 9.2 Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 9.3 Copy model artifacts and app
COPY app.py .
COPY model.ubj .
COPY encoding_maps.json .

EXPOSE 8080

# 9.4 Start server on port 8080 (Cloud Run default)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
