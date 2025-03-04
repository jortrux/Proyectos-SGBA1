# Start with the slim Python image
FROM python:3.12-slim

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# The entrypoint will be the python command
ENTRYPOINT ["python"]
