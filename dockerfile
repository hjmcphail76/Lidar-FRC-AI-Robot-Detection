# Use official Python 3.11 slim image
FROM python:3.11-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, Pygame, etc.
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if needed (optional)
EXPOSE 80

# Environment variables
ENV NAME World

# Run main.py by default
CMD ["python", "main.py"]
