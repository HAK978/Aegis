# Model Download Instructions

This project requires two AI models that are too large to include in the repository. Follow these instructions to download them.

## Required Models

### 1. YOLOv8x (Object Detection)
- **Size**: ~340 MB
- **Auto-download**: The model will automatically download on first run
- **Manual download** (optional):
  ```bash
  mkdir -p models
  wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt -O models/yolov8x.pt
  ```
- **Location**: `models/yolov8x.pt`
- **Cache**: Models are also cached at `~/.cache/torch/hub/`

### 2. Depth-Anything V2 Large (Depth Estimation)
- **Size**: ~1.3 GB
- **Download from**: [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/tree/main)
- **Instructions**:
  ```bash
  cd Depth-Anything-V2
  mkdir -p checkpoints
  cd checkpoints
  
  # Download the model file
  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
  
  # Verify file size (should be ~1.3 GB)
  ls -lh depth_anything_v2_vitl.pth
  ```
- **Location**: `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth`
- **Checksum** (optional verification):
  ```bash
  md5sum depth_anything_v2_vitl.pth
  # Expected: [add MD5 if known]
  ```

## Alternative: Smaller Models

If you have limited storage or slower hardware, you can use smaller variants:

### YOLOv8 Variants
- **YOLOv8n** (nano): 6 MB - Fastest, least accurate
- **YOLOv8s** (small): 22 MB
- **YOLOv8m** (medium): 52 MB
- **YOLOv8l** (large): 88 MB
- **YOLOv8x** (extra-large): 340 MB - Most accurate (recommended)

To use a different variant, edit `amd_backend_no_vllm.py`:
```python
self.yolo_model = YOLO('yolov8l.pt')  # Change from 'yolov8x.pt'
```

### Depth-Anything V2 Variants
- **Small**: ~98 MB - `depth_anything_v2_vits.pth`
- **Base**: ~196 MB - `depth_anything_v2_vitb.pth`
- **Large**: 335 MB - `depth_anything_v2_vitl.pth` (recommended)

To use a different variant, edit `amd_backend_no_vllm.py`:
```python
self.model_configs = {
    'encoder': 'vitb'  # Change from 'vitl' to 'vits' or 'vitb'
}
```

## Verification

After downloading, verify the models are correctly placed:

```bash
# Check YOLO model
python -c "from ultralytics import YOLO; m = YOLO('yolov8x.pt'); print('✅ YOLOv8x loaded successfully')"

# Check Depth-Anything V2 model
ls -lh Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth

# Expected output:
# -rw-r--r-- 1 user user 1.3G Oct 25 10:00 depth_anything_v2_vitl.pth
```

## First Run

On first run, the server will:
1. Load YOLOv8x (1-2 minutes if auto-downloading)
2. Load Depth-Anything V2 (30-60 seconds)
3. Initialize AMD GPU (a few seconds)

**Total setup time**: 2-5 minutes on first run, <10 seconds on subsequent runs.

## Troubleshooting

### YOLOv8x won't download?
- Check internet connection
- Try manual download (see above)
- Check disk space: `df -h`

### Depth-Anything V2 import error?
```bash
# Ensure the model file exists
ls Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth

# Check the Depth-Anything-V2 submodule
cd Depth-Anything-V2
git pull origin main
```

### Out of disk space?
- YOLOv8x: 340 MB
- Depth-Anything V2: 1.3 GB
- **Total required**: ~1.7 GB

### GPU out of memory?
Very unlikely with AMD MI300X (205 GB VRAM). Current usage:
- YOLOv8x: ~800 MB VRAM
- Depth-Anything V2: ~700 MB VRAM
- **Total VRAM**: ~1.5 GB / 205 GB (0.7% usage)

---

**Note**: These models are excluded from the Git repository via `.gitignore` due to their large size. GitHub has a 100 MB file limit.
