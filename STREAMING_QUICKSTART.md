# Streaming Mode Quick Start Guide

## Installation

1. Follow standard installation from [docs/INSTALL.md](docs/INSTALL.md)
2. Ensure you have a working camera or video file

## Basic Usage

### Track from Webcam

```bash
# Track animals
python run_streaming.py --source 0 --text-prompt animal

# Track people
python run_streaming.py --source 0 --text-prompt person

# Track birds
python run_streaming.py --source 0 --text-prompt bird
```

### Track from Video File

```bash
python run_streaming.py --source /path/to/video.mp4 --text-prompt animal
```

### Save Output Video

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --output-video outputs/result.mp4
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `r` | Reset tracker |

## Common Options

### Detection Prompts

```bash
--text-prompt person          # Detect people
--text-prompt animal          # Detect common animals
--text-prompt bird            # Detect birds
--text-prompt vehicle         # Detect vehicles
--text-prompt chimpanzee      # Detect apes/chimpanzees
```

### Custom Detection

```bash
--text-prompt custom --custom-prompt "dog,cat,rabbit"
```

### Performance Tuning

```bash
# Fast mode (use smaller model)
--sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml

# Limit FPS for slower hardware
--target-fps 15

# CPU mode (no GPU)
--sam2-device cpu --detector-device cpu
```

### Visualization

```bash
# Hide detection scores
--show-scores False

# Adjust mask transparency
--mask-alpha 0.7

# Hide masks (boxes only)
--show-masks False
```

## Troubleshooting

### Camera not found
```bash
# Try different camera indices
python run_streaming.py --source 0  # First camera
python run_streaming.py --source 1  # Second camera
```

### Low FPS
```bash
# Use smaller model
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml

# Limit processing rate
python run_streaming.py --source 0 --target-fps 10
```

### Too many false detections
```bash
# Increase detection threshold
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --detection-threshold 0.5
```

### Missing objects
```bash
# Decrease detection threshold
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --detection-threshold 0.2
```

## Example Commands

### High-quality tracking (requires powerful GPU)
```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --compile-sam2 True \
  --output-video outputs/high_quality.mp4
```

### Fast tracking (real-time on moderate GPU)
```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --target-fps 20
```

### CPU-only mode (slow but works without GPU)
```bash
python run_streaming.py \
  --source 0 \
  --text-prompt person \
  --sam2-device cpu \
  --detector-device cpu \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --target-fps 5
```

### Multiple animal species
```bash
python run_streaming.py \
  --source 0 \
  --text-prompt custom \
  --custom-prompt "dog,cat,rabbit,hamster,bird" \
  --detection-threshold 0.35
```

### Video file processing with output
```bash
python run_streaming.py \
  --source videos/wildlife.mp4 \
  --text-prompt animal \
  --output-video outputs/tracked_wildlife.mp4 \
  --show-fps True
```

### VOS-optimized tracking (2x faster, NEW!)
```bash
# VOS optimization for maximum speed (~40 FPS with small model)
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --vos-optimize True \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --num-maskmem 7 \
  --output-video outputs/vos_tracking.mp4

# Or use the VOS demo script
python examples/streaming_demo_vos.py --source 0 --prompt animal
```

**VOS Benefits:**
- 🚀 ~2x speed improvement
- 💾 Memory-stable for long sessions
- 🎯 Based on Gy920's optimizations

**Note:** First few frames slower (torch.compile warmup)

## Architecture Overview

```
┌─────────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────┐
│   Camera/   │────>│ Detector │────>│  SAM2   │────>│ Visualization│
│    Video    │     │  (Zero-  │     │(Segment)│     │  (Real-time) │
└─────────────┘     │  shot)   │     └─────────┘     └──────────────┘
                    └──────────┘          │
                         │                │
                         └────────────────┘
                         Adaptive Threshold
                        (Online Otsu Method)
```

## Performance Expectations

### GPU: RTX 3090

**Standard Mode:**

| Model | FPS | Quality |
|-------|-----|---------|
| `sam2.1_hiera_tiny.pt` | ~30 | Good |
| `sam2.1_hiera_small.pt` | ~20 | Better |
| `sam2.1_hiera_base_plus.pt` | ~12 | Great |
| `sam2.1_hiera_large.pt` | ~8 | Best |

**VOS-Optimized Mode (--vos-optimize True):**

| Model | FPS (Standard) | FPS (VOS) | Speedup |
|-------|----------------|-----------|---------|
| `sam2.1_hiera_tiny.pt` | ~30 | ~60 | 2.0x |
| `sam2.1_hiera_small.pt` | ~20 | ~40 | 2.0x |
| `sam2.1_hiera_base_plus.pt` | ~12 | ~24 | 2.0x |
| `sam2.1_hiera_large.pt` | ~8 | ~16 | 2.0x |

### GPU: RTX 3060

| Model | FPS | Quality |
|-------|-----|---------|
| `sam2.1_hiera_tiny.pt` | ~20 | Good |
| `sam2.1_hiera_small.pt` | ~12 | Better |
| `sam2.1_hiera_base_plus.pt` | ~7 | Great |
| `sam2.1_hiera_large.pt` | ~4 | Best |

### CPU Only

| Model | FPS | Quality |
|-------|-----|---------|
| `sam2.1_hiera_tiny.pt` | ~1-2 | Good |
| `sam2.1_hiera_small.pt` | ~0.5-1 | Better |

*Note: FPS depends on number of tracked objects, image resolution, and hardware configuration*

## Features

✅ Real-time processing
✅ Multiple object tracking
✅ Adaptive detection thresholds
✅ Zero-shot (no training needed)
✅ USB camera support
✅ Video file support
✅ Interactive visualization
✅ Video output
✅ Pause/resume capability
✅ Tracker reset

## Next Steps

- Read full documentation: [docs/STREAMING.md](docs/STREAMING.md)
- Try the simple demo: `python examples/streaming_demo.py`
- Explore all options: `python run_streaming.py --help`

## Support

For detailed information, see:
- [STREAMING.md](docs/STREAMING.md) - Complete streaming documentation
- [TRACKING.md](docs/TRACKING.md) - Batch processing documentation
- [DETECTION.md](docs/DETECTION.md) - Detection model information
- [INSTALL.md](docs/INSTALL.md) - Installation instructions
