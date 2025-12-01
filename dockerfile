# Use Python 3.11 slim image
FROM python:3.11-slim-buster

# Set working directory
WORKDIR /app

# Fix old buster repos if needed
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
    && sed -i '/security.debian.org/d' /etc/apt/sources.list

# Install system dependencies for OpenCV, Pygame, and building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . /app

# Optional: expose a port only if your app uses HTTP
# EXPOSE 80

# Default environment variable
ENV NAME World

# Run your main Python script
CMD ["python", "main.py"]
