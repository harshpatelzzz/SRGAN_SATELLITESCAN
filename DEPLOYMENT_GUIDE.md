# 🚀 Deployment Guide

Complete guide for deploying the SRGAN system.

## Quick Start

### 1. Backend (FastAPI)

```bash
cd api
pip install -r requirements.txt
python main.py
```

API runs on: **http://localhost:8000**

### 2. Frontend (Next.js)

```bash
cd frontend-next
npm install
npm run dev
```

Frontend runs on: **http://localhost:3000**

## Production Deployment

### Backend Deployment

#### Option 1: Using Uvicorn

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Option 2: Using Gunicorn

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment

#### Build for Production

```bash
cd frontend-next
npm run build
npm start
```

#### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

## Environment Variables

Create `.env` files:

**api/.env:**
```
MODEL_PATH=../checkpoints/generator_final.pth
DEVICE=cuda
```

**frontend-next/.env.local:**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Docker Deployment

### Backend Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## Google Colab Training

1. Upload `colab/SRGAN_Training.ipynb` to Colab
2. Run cells sequentially
3. Training uses free GPU
4. Download checkpoints when done

## System Requirements

- **Python**: 3.8+
- **Node.js**: 18+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: Optional but recommended for training
- **Storage**: 10GB+ for dataset and models
