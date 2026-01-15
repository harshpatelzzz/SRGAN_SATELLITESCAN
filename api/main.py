"""
FastAPI backend for SRGAN Satellite Imagery Super-Resolution
Production-ready REST API with comprehensive endpoints
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import torch
from PIL import Image
import io
import base64
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
# Insert at beginning to ensure project modules are found first
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.generator import Generator
from evaluate.metrics import calculate_psnr, calculate_ssim
from utils.config import Config

app = FastAPI(
    title="SRGAN Satellite Imagery Super-Resolution API",
    description="REST API for 4× upscaling of satellite images using SRGAN",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = project_root / "api" / "uploads"
OUTPUT_DIR = project_root / "api" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global model cache
generator = None
device = torch.device(Config.DEVICE)

class UpscaleResponse(BaseModel):
    success: bool
    original_size: str
    upscaled_size: str
    upscaled_image_base64: str
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    processing_time: float

class ModelInfo(BaseModel):
    model_available: bool
    model_path: Optional[str]
    generator_params: Optional[int]
    device: str
    scale_factor: int

def load_model():
    """Load the trained generator model"""
    global generator
    try:
        if generator is None:
            # Try final model first, then pretrained
            model_paths = [
                project_root / "checkpoints" / "generator_final.pth",
                project_root / "checkpoints" / "generator_pretrained_final.pth"
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break
            
            if not model_path:
                print(f"Warning: No model found in {[str(p) for p in model_paths]}")
                return False
            
            print(f"Loading model from: {model_path}")
            generator = Generator(
                num_residual_blocks=Config.GENERATOR_RESIDUAL_BLOCKS,
                num_features=Config.GENERATOR_FEATURES,
                scale_factor=Config.SCALE_FACTOR,
                num_channels=3
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['model_state_dict'])
            elif 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                generator.load_state_dict(checkpoint)
            
            generator.eval()
            print("Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SRGAN Satellite Imagery Super-Resolution API",
        "version": "1.0.0",
        "endpoints": {
            "/api/upscale": "POST - Upscale an image",
            "/api/model/info": "GET - Get model information",
            "/api/health": "GET - Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = generator is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_available": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    model_loaded = load_model()
    
    if model_loaded:
        params = sum(p.numel() for p in generator.parameters())
        model_path = None
        for path in [
            project_root / "checkpoints" / "generator_final.pth",
            project_root / "checkpoints" / "generator_pretrained_final.pth"
        ]:
            if path.exists():
                model_path = str(path)
                break
    else:
        params = None
        model_path = None
    
    return ModelInfo(
        model_available=model_loaded,
        model_path=model_path,
        generator_params=params,
        device=str(device),
        scale_factor=Config.SCALE_FACTOR
    )

@app.post("/api/upscale", response_model=UpscaleResponse)
async def upscale_image(file: UploadFile = File(...)):
    """
    Upscale an image using SRGAN
    
    - **file**: Image file to upscale (PNG, JPG, JPEG, BMP, TIF)
    - Returns: Upscaled image (4×) with metrics
    """
    start_time = datetime.now()
    
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Load model
    if not load_model():
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = f"{image.width}×{image.height}"
        
        # Limit image size to prevent memory issues
        # For 4x upscaling, limit input to 512x512 (becomes 2048x2048)
        # This prevents OOM errors while still allowing good results
        max_size = 512
        if image.width > max_size or image.height > max_size:
            # Resize if too large, maintaining aspect ratio
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            original_size = f"{image.width}×{image.height} (resized from original)"
        
        # Preprocess
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image).unsqueeze(0).to(device)
        
        # Upscale
        with torch.no_grad():
            upscaled_tensor = generator(image_tensor)
            upscaled_tensor = torch.clamp(upscaled_tensor, 0.0, 1.0)
        
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        upscaled_image = to_pil(upscaled_tensor.squeeze(0).cpu())
        upscaled_size = f"{upscaled_image.width}×{upscaled_image.height}"
        
        # Calculate metrics (if we had HR image)
        # For now, we'll skip PSNR/SSIM as we don't have ground truth
        
        # Convert to base64
        buffered = io.BytesIO()
        upscaled_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return UpscaleResponse(
            success=True,
            original_size=original_size,
            upscaled_size=upscaled_size,
            upscaled_image_base64=f"data:image/png;base64,{img_base64}",
            psnr=None,
            ssim=None,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {error_details}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/metrics")
async def get_training_metrics():
    """Get training metrics from evaluation"""
    try:
        # Check if evaluation results exist
        results_file = project_root / "results" / "evaluation_results.json"
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "message": "No evaluation results available. Run evaluation first.",
                "srgan_psnr": None,
                "srgan_ssim": None,
                "bicubic_psnr": None,
                "bicubic_ssim": None
            }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
