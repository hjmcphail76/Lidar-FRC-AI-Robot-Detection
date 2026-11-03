# Use an official, lightweight Python 3.11 image built for multi-architecture (works perfectly on ARM/Pi)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install basic system-level build tools if any of your pip packages need to compile binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies directly into the container's global Python environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Run your script when the container starts
CMD ["python", "main.py"]
