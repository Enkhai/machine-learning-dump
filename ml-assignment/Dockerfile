# syntax=docker/dockerfile:1
# Base image per requirement
FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies: Python 3, pip, and common utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       python3-venv \
       ca-certificates \
       git \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" command exists (some tools assume it)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies (if the file is empty, this will be a no-op)
RUN if [ -s requirements.txt ]; then \
      pip3 install --no-cache-dir -r requirements.txt; \
    else \
      echo "requirements.txt is empty; skipping pip install"; \
    fi

# Copy the rest of the project
COPY . .

# Default command keeps the container alive for interactive use
# You can override this in docker-compose or docker run
CMD ["bash"]
