# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . /app

# Create necessary directories if they don't exist
RUN mkdir -p /app/HealthCareData /app/processed /app/models/finetuned /app/docs/summaries /app/docs/changelog

# Expose port for the server
EXPOSE 5000

# Set entrypoint and default command
ENTRYPOINT ["python"]
CMD ["scripts/serve.py"]