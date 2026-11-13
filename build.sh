#!/usr/bin/env bash
# Exit immediately if any command exits with a non-zero status
set -e

echo "=== 1/2: Installing Python Dependencies (CPU-Optimized PyTorch) ==="
pip install --no-cache-dir -r requirements.txt

echo "=== 2/2: Building React Frontend Production Artifacts ==="
cd frontend
npm install
npm run build
cd ..

echo "=== Build Complete! Frontend Dist Mounted in FastAPI. ==="

