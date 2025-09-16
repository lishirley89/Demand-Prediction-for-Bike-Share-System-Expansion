# Multi-stage build for smaller final image

FROM python:3.9-slim AS base

# System dependencies for GeoPandas stack (GDAL/GEOS/PROJ/rtree)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libspatialindex-dev \
    proj-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_DIR=/tmp

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project code
COPY code /app/code
COPY result /app/result
COPY data /app/data
COPY app.py /app/app.py

# Expose port for the FastAPI app
EXPOSE 8080

# Default command: run the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]


