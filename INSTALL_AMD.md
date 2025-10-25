# AMD ROCm Installation Guide

This project is optimized for AMD MI300X GPUs with ROCm 7.0. Follow these steps for proper installation.

## Prerequisites

### System Requirements
- **GPU**: AMD MI300X (or other ROCm-compatible AMD GPU)
- **OS**: Ubuntu 22.04 LTS or later (recommended)
- **ROCm**: Version 7.0 or later
- **Python**: 3.10+
- **VRAM**: Minimum 2 GB (recommended 16+ GB)

### Check Your GPU

```bash
# Install ROCm drivers first (if not already installed)
# See: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

# Verify GPU is detected
rocm-smi

# Expected output:
# GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
# 0    45°C   50W     1200MHz 1600MHz  30%  auto  300W    1%     0%
```

### Verify ROCm Version

```bash
rocminfo | grep "ROCm"
# Expected: ROCm version 7.0 or higher
```

## Installation Steps

### 1. Create Conda Environment

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart shell

# Create environment for ROCm
conda create -n rocmserve python=3.10 -y
conda activate rocmserve
```

### 2. Install PyTorch with ROCm Support

**IMPORTANT**: Install PyTorch BEFORE other packages to ensure ROCm compatibility.

```bash
# Install PyTorch nightly with ROCm 7.0 support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

**Verify PyTorch ROCm installation:**

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Device count:', torch.cuda.device_count())"
python -c "import torch; print('Device name:', torch.cuda.get_device_name(0))"

# Expected output:
# PyTorch version: 2.10.0.dev20251023+rocm7.0
# CUDA available: True
# Device count: 1
# Device name: AMD Instinct MI300X VF
```

**Troubleshooting**: If `torch.cuda.is_available()` returns `False`:
- Check ROCm drivers: `rocm-smi`
- Verify ROCm version: `rocminfo | grep "ROCm"`
- Ensure you installed the correct PyTorch variant (rocm7.0)

### 3. Install Project Dependencies

```bash
cd ~/mi300x-serve
pip install -r requirements.txt
```

This will install:
- Flask web framework
- Ultralytics YOLOv8
- OpenCV for image processing
- NumPy, Pillow, scipy
- timm (for Depth-Anything V2)

### 4. Download AI Models

#### YOLOv8x (Auto-downloads on first run)

```bash
# Will auto-download on first run, or manually:
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt -O models/yolov8x.pt
```

#### Depth-Anything V2 Large

```bash
cd Depth-Anything-V2
mkdir -p checkpoints
cd checkpoints

# Download model weights (~1.3 GB)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

# Verify file size
ls -lh depth_anything_v2_vitl.pth
# Expected: ~1.3 GB
```

### 5. Test Installation

```bash
# Test PyTorch + ROCm
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Test YOLOv8
python -c "
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
print('✅ YOLOv8x loaded successfully')
"

# Test Depth-Anything V2
python -c "
import sys
sys.path.append('Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
print('✅ Depth-Anything V2 imported successfully')
"
```

### 6. Start the Server

```bash
python server.py
```

**Expected output:**
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your-ip>:5000
```

## Performance Tuning for AMD MI300X

### VRAM Optimization

The MI300X has 205+ GB of VRAM. You can load larger models or multiple models simultaneously.

**Check VRAM usage:**
```bash
rocm-smi --showmeminfo vram

# During inference:
watch -n 1 rocm-smi
```

**Current VRAM usage:**
- YOLOv8x: ~800 MB
- Depth-Anything V2: ~700 MB
- **Total**: ~1.5 GB / 205 GB (0.7%)

### GPU Utilization

```bash
# Monitor GPU usage in real-time
watch -n 1 rocm-smi

# Expected during inference:
# GPU%: 10-15% (lightweight workload for MI300X)
# Power: 50-100W
# Temp: 40-60°C
```

### Batch Processing (Optional)

For higher throughput, you can process multiple frames in batches:

```python
# In amd_backend_no_vllm.py
# Current: processes 1 frame at a time
# To enable batching: modify process_frame() to accept list of frames
```

## Common Issues & Solutions

### Issue 1: "RuntimeError: No HIP GPUs are available"

**Cause**: ROCm drivers not installed or not detected

**Solution**:
```bash
# Check ROCm installation
rocm-smi

# If not found, install ROCm:
# https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

# Verify environment variable
echo $ROCM_HOME
# Should be: /opt/rocm or similar
```

### Issue 2: "ImportError: libamdhip64.so.6: cannot open shared object file"

**Cause**: ROCm libraries not in library path

**Solution**:
```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:$PATH

# Reload
source ~/.bashrc
```

### Issue 3: PyTorch uses CPU instead of GPU

**Cause**: Wrong PyTorch version installed (CPU-only or wrong CUDA/ROCm version)

**Solution**:
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with ROCm support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Verify
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Issue 4: Out of Memory (unlikely with MI300X)

**Cause**: Model too large or memory leak

**Solution**:
```bash
# Check VRAM usage
rocm-smi --showmeminfo vram

# Use smaller YOLOv8 variant
# In amd_backend_no_vllm.py:
# self.yolo_model = YOLO('yolov8l.pt')  # Instead of yolov8x.pt

# Or use smaller Depth-Anything variant
# encoder: 'vitb' or 'vits' instead of 'vitl'
```

### Issue 5: Low Performance

**Check these factors:**
```bash
# 1. GPU is being used
python -c "import torch; print(torch.cuda.current_device())"

# 2. ROCm is properly configured
rocminfo | grep "Name"

# 3. No CPU throttling
cat /proc/cpuinfo | grep MHz

# 4. Network bandwidth (if using remote access)
ping -c 10 <server-ip>
```

## Production Deployment

### Using systemd Service

Create `/etc/systemd/system/visual-guidance.service`:

```ini
[Unit]
Description=AI Visual Guidance System
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/mi300x-serve
Environment="PATH=/home/your-username/miniconda3/envs/rocmserve/bin"
ExecStart=/home/your-username/miniconda3/envs/rocmserve/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable visual-guidance
sudo systemctl start visual-guidance
sudo systemctl status visual-guidance
```

### Using Docker (Advanced)

```dockerfile
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.5.0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "server.py"]
```

Build and run:
```bash
docker build -t visual-guidance:rocm .
docker run --device=/dev/kfd --device=/dev/dri -p 5000:5000 visual-guidance:rocm
```

## Benchmarking

Test your system performance:

```bash
# Download test script (if available)
python benchmark_models.sh

# Expected results on MI300X:
# YOLOv8x inference: 60-80ms
# Depth-Anything V2: 30-50ms
# Total latency: 60-65ms
```

## Additional Resources

- **ROCm Documentation**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/
- **AMD GPU Cloud**: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Depth-Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2

---

**Successfully installed?** Proceed to the main [README.md](README.md) for usage instructions!
