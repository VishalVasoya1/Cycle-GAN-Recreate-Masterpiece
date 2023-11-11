#!/bin/bash
cd /var/www/Cycle-GAN-Recreate-Masterpiece

source ../venv/bin/activate

uvicorn Deployment-FastAPI.main:app --uds=/tmp/uvicorn.sock