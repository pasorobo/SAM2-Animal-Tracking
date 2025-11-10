# Real-time Streaming Tracking

This document describes how to use the real-time streaming mode for tracking objects from USB cameras or video files.

## Overview

The streaming mode processes video frames in real-time, supporting:
- **USB webcams** (or any camera device)
- **Video files** (MP4, AVI, etc.)
- **Multiple object tracking** with automatic ID assignment
- **Adaptive detection thresholds** using online Otsu method
- **Real-time visualization** with masks, boxes, and IDs
- **Video output** for saving results

## Quick Start

### 1. USB Camera Tracking

Track objects from your webcam (camera 0):

```bash
python run_streaming.py --source 0 --text-prompt animal
```

For camera 1, 2, etc.:

```bash
python run_streaming.py --source 1 --text-prompt person
```

### 2. Video File Tracking

Track objects in a video file:

```bash
python run_streaming.py --source /path/to/video.mp4 --text-prompt animal
```

### 3. Save Output Video

Save the tracking results to a video file:

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --output-video outputs/tracking_result.mp4
```

## Command Line Arguments

### Input Settings

- `--source`: Input source
  - Camera index: `0`, `1`, `2`, ...
  - Video file: `/path/to/video.mp4`
  - Default: `0` (first camera)

- `--target-fps`: Target FPS for processing
  - Limits processing speed (useful for slower hardware)
  - Default: `None` (process all frames)
  - Example: `--target-fps 15`

### Detection Settings

- `--text-prompt`: Predefined text prompt for detection
  - `person`: Detect people
  - `animal`: Detect animals (dog, cat, bird, horse, sheep, cow)
  - `chimpanzee`: Detect apes/chimpanzees
  - `bird`: Detect birds
  - `vehicle`: Detect vehicles
  - `all`: Detect all objects
  - `custom`: Use custom prompt (requires `--custom-prompt`)
  - Default: `animal`

- `--custom-prompt`: Custom detection prompt (comma-separated)
  - Example: `--text-prompt custom --custom-prompt "dog,cat,rabbit"`

- `--detector-model`: HuggingFace detector model
  - Default: `iSEE-Laboratory/llmdet_large`
  - Alternatives: `google/owlv2-base-patch16-ensemble`, `IDEA-Research/grounding-dino-base`

- `--detection-threshold`: Initial detection threshold (0-1)
  - Default: `0.3`
  - Lower = more detections (more false positives)
  - Higher = fewer detections (may miss objects)

- `--high-conf-threshold`: High confidence threshold for adding new objects
  - Default: `0.4`

- `--use-adaptive-threshold`: Enable adaptive threshold (Otsu method)
  - Default: `True`
  - Automatically adjusts thresholds based on detection score distribution

### SAM2 Settings

- `--sam2-checkpoint`: SAM2 checkpoint path
  - Default: `sam2/checkpoints/sam2.1_hiera_large.pt`
  - Available: `sam2.1_hiera_large.pt`, `sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_tiny.pt`
  - Smaller models = faster but less accurate

- `--sam2-config`: SAM2 config file
  - Default: `configs/sam2.1/sam2.1_hiera_l.yaml`
  - Must match checkpoint size

- `--sam2-device`: Device for SAM2
  - Default: `cuda`
  - Use `cpu` if no GPU available (much slower)

- `--compile-sam2`: Compile SAM2 encoder for faster inference
  - Default: `True`
  - Requires PyTorch 2.0+

### Visualization Settings

- `--show-masks`: Show segmentation masks
  - Default: `True`

- `--show-boxes`: Show bounding boxes
  - Default: `True`

- `--show-ids`: Show track IDs
  - Default: `True`

- `--show-scores`: Show detection scores
  - Default: `False`

- `--show-fps`: Show FPS counter
  - Default: `True`

- `--mask-alpha`: Mask transparency (0-1)
  - Default: `0.5`
  - Lower = more transparent, Higher = more opaque

### Output Settings

- `--output-video`: Output video file path
  - Default: `None` (no output)
  - Example: `outputs/result.mp4`

- `--output-fps`: Output video FPS
  - Default: `30`

## Keyboard Controls

During execution:
- **`q`**: Quit the application
- **`p`**: Pause/Resume tracking
- **`r`**: Reset tracker (clear all tracks and reinitialize)

## Usage Examples

### Example 1: Track Animals from Webcam

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --show-masks True \
  --show-ids True \
  --output-video outputs/webcam_tracking.mp4
```

### Example 2: Track People in Video File

```bash
python run_streaming.py \
  --source videos/crowd.mp4 \
  --text-prompt person \
  --detection-threshold 0.4 \
  --output-video outputs/crowd_tracking.mp4
```

### Example 3: Track Birds with Custom Settings

```bash
python run_streaming.py \
  --source videos/birds.mp4 \
  --text-prompt bird \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --mask-alpha 0.7 \
  --output-video outputs/birds_tracking.mp4
```

### Example 4: Track Multiple Animal Species

```bash
python run_streaming.py \
  --source 1 \
  --text-prompt custom \
  --custom-prompt "dog,cat,rabbit,hamster" \
  --detection-threshold 0.35 \
  --show-fps True
```

### Example 5: CPU-only Mode (No GPU)

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

### Example 6: High-Performance Tracking

```bash
python run_streaming.py \
  --source videos/test.mp4 \
  --text-prompt animal \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --compile-sam2 True \
  --use-adaptive-threshold True \
  --output-video outputs/high_perf.mp4
```

## Performance Optimization Tips

### 1. Use Smaller SAM2 Model

For real-time performance, use a smaller SAM2 model:

| Model | Speed | Accuracy |
|-------|-------|----------|
| `sam2.1_hiera_tiny.pt` | Fastest | Lower |
| `sam2.1_hiera_small.pt` | Fast | Good |
| `sam2.1_hiera_base_plus.pt` | Medium | Better |
| `sam2.1_hiera_large.pt` | Slower | Best |

```bash
# Fast tracking
--sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml
```

### 2. Enable Model Compilation

PyTorch 2.0+ model compilation can significantly improve speed:

```bash
--compile-sam2 True
```

### 3. Limit Target FPS

Process fewer frames per second:

```bash
--target-fps 15  # Process only 15 FPS
```

### 4. Use GPU

Always use GPU if available:

```bash
--sam2-device cuda --detector-device cuda
```

### 5. Adjust Detection Threshold

Higher threshold = fewer detections = faster processing:

```bash
--detection-threshold 0.5  # Only high-confidence detections
```

### 6. Disable Unnecessary Visualization

```bash
--show-scores False  # Disable score display
```

## Architecture Details

### Online Processing Pipeline

```
Frame Input → Detection → SAM2 Propagation → Tracking Logic → Visualization
     ↓            ↓              ↓                  ↓              ↓
  Camera/     Zero-shot      Segmentation      Add/Remove      Real-time
   Video      Detection         Masks           Objects         Display
```

### Key Differences from Batch Mode

| Feature | Batch Mode (run.py) | Streaming Mode (run_streaming.py) |
|---------|---------------------|-----------------------------------|
| Input | Image folders | Camera/Video stream |
| Processing | Offline (all frames) | Online (frame-by-frame) |
| Detection | Pre-computed | Real-time |
| Threshold | Otsu (global) | Otsu (online, adaptive) |
| Latency | N/A | Low (real-time) |
| Use case | Evaluation | Live applications |

### Adaptive Threshold

The streaming mode uses an **online Otsu method** for adaptive detection thresholds:

1. Maintains a history of last 100 detection scores
2. Recalculates Otsu threshold every frame
3. Adapts to changing scene conditions
4. No need for manual threshold tuning

## Troubleshooting

### Issue: Low FPS

**Solutions:**
- Use smaller SAM2 model (`sam2.1_hiera_small.pt` or `sam2.1_hiera_tiny.pt`)
- Enable `--compile-sam2 True`
- Reduce `--target-fps`
- Increase `--detection-threshold` to reduce number of objects
- Use GPU (`--sam2-device cuda`)

### Issue: Camera Not Found

**Solutions:**
- Check camera index (try `--source 0`, `--source 1`, etc.)
- Verify camera is connected: `ls /dev/video*` (Linux)
- Install camera drivers
- Try different source index

### Issue: Too Many False Positives

**Solutions:**
- Increase `--detection-threshold` (e.g., `0.4` or `0.5`)
- Increase `--high-conf-threshold`
- Use more specific text prompts (e.g., `--text-prompt custom --custom-prompt "golden retriever"`)

### Issue: Missing Objects

**Solutions:**
- Decrease `--detection-threshold` (e.g., `0.2` or `0.25`)
- Use broader text prompts (e.g., `--text-prompt animal` instead of `bird`)
- Disable adaptive threshold: `--use-adaptive-threshold False`

### Issue: CUDA Out of Memory

**Solutions:**
- Use smaller SAM2 model
- Process fewer FPS: `--target-fps 10`
- Use CPU mode: `--sam2-device cpu --detector-device cpu`

## Technical Details

### Memory Management

- Maintains only recent frame history (configurable)
- Automatically cleans up old tracking data
- Uses temporary directory for SAM2 frame storage

### Threading Model

Currently single-threaded. Future improvements:
- Separate detection and tracking threads
- Asynchronous visualization
- Frame buffer queue

### Latency

Typical latency (per frame):
- Detection: 20-50ms (depends on detector model)
- SAM2 propagation: 30-100ms (depends on SAM2 model and number of objects)
- Visualization: 5-10ms
- **Total: ~60-160ms (6-17 FPS)** on RTX 3090

For better real-time performance:
- Use `sam2.1_hiera_tiny.pt`: ~30 FPS
- Use `sam2.1_hiera_small.pt`: ~20 FPS
- Use `sam2.1_hiera_large.pt`: ~8 FPS

## Limitations

1. **Hardware Requirements**: Requires GPU for real-time performance
2. **Single-threaded**: No parallel processing yet
3. **Memory**: Accumulates temporary frame files
4. **Object Persistence**: Long-lost objects may be re-assigned new IDs

## Future Enhancements

- [ ] Multi-threaded processing
- [ ] Frame queue buffering
- [ ] WebRTC support for remote streaming
- [ ] REST API for integration
- [ ] Mobile device support
- [ ] Multiple camera support

## Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/ecker-lab/SAM2-Animal-Tracking/issues)
- Check existing documentation: `docs/TRACKING.md`, `docs/DETECTION.md`
