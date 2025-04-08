FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg + Python deps
RUN apt-get update && \
    apt-get install -y ffmpeg sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including model)
COPY . .

CMD ["python", "main.py"]