"""
Flask web frontend for SRGAN Satellite Imagery Super-Resolution
Simple web interface to upload and upscale images
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.generator import Generator
from utils.config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = project_root / 'frontend' / 'uploads'
app.config['OUTPUT_FOLDER'] = project_root / 'frontend' / 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(parents=True, exist_ok=True)

# Load model (lazy loading)
generator = None
device = torch.device(Config.DEVICE)

def load_model():
    """Load the trained generator model"""
    global generator
    if generator is None:
        model_path = project_root / "checkpoints" / "generator_final.pth"
        if not model_path.exists():
            model_path = project_root / "checkpoints" / "generator_pretrained_final.pth"
        
        if model_path.exists():
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
            return True
        else:
            return False
    return True

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale():
    """Handle image upscaling"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIF'}), 400
    
    try:
        # Load model if not loaded
        if not load_model():
            return jsonify({'error': 'Model not found. Please train the model first.'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(input_path)
        
        # Load and preprocess image
        image = Image.open(input_path).convert('RGB')
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image).unsqueeze(0).to(device)
        
        # Upscale
        with torch.no_grad():
            upscaled_tensor = generator(image_tensor)
            upscaled_tensor = torch.clamp(upscaled_tensor, 0.0, 1.0)
        
        # Convert to PIL Image
        to_pil = transforms.ToPILImage()
        upscaled_image = to_pil(upscaled_tensor.squeeze(0).cpu())
        
        # Save output
        output_filename = f"upscaled_{filename}"
        output_path = app.config['OUTPUT_FOLDER'] / output_filename
        upscaled_image.save(output_path)
        
        # Return result
        return jsonify({
            'success': True,
            'output_url': f'/output/{output_filename}',
            'original_size': f'{image.width}x{image.height}',
            'upscaled_size': f'{upscaled_image.width}x{upscaled_image.height}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<filename>')
def output_file(filename):
    """Serve upscaled images"""
    return send_file(app.config['OUTPUT_FOLDER'] / filename)

@app.route('/status')
def status():
    """Check if model is available"""
    model_path = project_root / "checkpoints" / "generator_final.pth"
    pretrained_path = project_root / "checkpoints" / "generator_pretrained_final.pth"
    
    has_model = model_path.exists() or pretrained_path.exists()
    
    return jsonify({
        'model_available': has_model,
        'model_path': str(model_path) if model_path.exists() else str(pretrained_path) if pretrained_path.exists() else None
    })

if __name__ == '__main__':
    print("=" * 60)
    print("SRGAN Web Frontend")
    print("=" * 60)
    print(f"Starting server on http://localhost:5000")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
