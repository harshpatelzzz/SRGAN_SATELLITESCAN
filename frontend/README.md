# SRGAN Web Frontend

Simple web interface for SRGAN satellite imagery super-resolution.

## Features

- 📤 Upload images via drag & drop or file picker
- 🚀 4× image upscaling
- 📊 Side-by-side comparison (original vs upscaled)
- 🎨 Modern, responsive UI

## Installation

```bash
# Install Flask (if not already installed)
pip install flask werkzeug
```

## Running the Frontend

```bash
# From project root
cd frontend
python app.py
```

Or from project root:
```bash
python frontend/app.py
```

Then open your browser to: **http://localhost:5000**

## Usage

1. Open http://localhost:5000 in your browser
2. Upload an image (drag & drop or click to browse)
3. Click "Upscale Image (4×)"
4. View the result side-by-side

## Requirements

- Trained model in `checkpoints/generator_final.pth` or `checkpoints/generator_pretrained_final.pth`
- Flask web framework

## File Structure

```
frontend/
├── app.py              # Flask backend server
├── templates/
│   └── index.html      # Web interface
├── uploads/            # Temporary uploaded files
└── outputs/            # Upscaled images
```
