"""
Real-time Streaming Tracking with SAM2
Supports USB cameras and video files
"""

import argparse
import os
import torch
import sys
import json
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tracker'))

from tracker.streaming_input import StreamingInputHandler
from tracker.online_sam2tracker import OnlineSAM2Tracker
from tracker.online_sam2tracker_vos import OnlineSAM2TrackerVOS
from tracker.realtime_visualizer import RealtimeVisualizer, VideoWriter
from tracker.utils.misc import str2bool


def parse_args():
    """Parse command line arguments"""

    # Predefined text prompts for common scenarios
    text_prompts = {
        "person": [["person"]],
        "animal": [["animal", "dog", "cat", "bird", "horse", "sheep", "cow"]],
        "chimpanzee": [["ape", "chimpanzee"]],
        "bird": [["bird"]],
        "vehicle": [["car", "truck", "bus", "motorcycle"]],
        "all": [["object"]],  # Detect all objects
    }

    parser = argparse.ArgumentParser(
        description="Real-time multi-object tracking with SAM2"
    )

    # Input source
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (0, 1, ...) or video file path"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Target FPS for processing (default: process all frames)"
    )

    # Detection settings
    parser.add_argument(
        "--detector-model",
        type=str,
        default="iSEE-Laboratory/llmdet_large",
        help="HuggingFace detector model"
    )
    parser.add_argument(
        "--detector-device",
        type=str,
        default="cuda",
        help="Detector device (cuda/cpu)"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="animal",
        choices=list(text_prompts.keys()) + ["custom"],
        help="Text prompt preset for detection"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="Custom text prompt (comma-separated, e.g., 'dog,cat,bird')"
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.3,
        help="Initial detection threshold"
    )
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=0.4,
        help="High confidence threshold for new objects"
    )
    parser.add_argument(
        "--use-adaptive-threshold",
        type=str2bool,
        default=True,
        help="Use adaptive detection threshold (Otsu method)"
    )

    # SAM2 settings
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path"
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path"
    )
    parser.add_argument(
        "--sam2-device",
        type=str,
        default="cuda",
        help="SAM2 device (cuda/cpu)"
    )

    # Visualization settings
    parser.add_argument(
        "--show-masks",
        type=str2bool,
        default=True,
        help="Show segmentation masks"
    )
    parser.add_argument(
        "--show-boxes",
        type=str2bool,
        default=True,
        help="Show bounding boxes"
    )
    parser.add_argument(
        "--show-ids",
        type=str2bool,
        default=True,
        help="Show track IDs"
    )
    parser.add_argument(
        "--show-scores",
        type=str2bool,
        default=False,
        help="Show detection scores"
    )
    parser.add_argument(
        "--show-fps",
        type=str2bool,
        default=True,
        help="Show FPS counter"
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.5,
        help="Mask transparency (0-1)"
    )

    # Output settings
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video file path (default: no output)"
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
        help="Output video FPS"
    )

    # Performance settings
    parser.add_argument(
        "--compile-sam2",
        type=str2bool,
        default=True,
        help="Compile SAM2 encoder for faster inference"
    )

    # VOS Optimization settings (Advanced)
    parser.add_argument(
        "--vos-optimize",
        type=str2bool,
        default=False,
        help="Enable VOS optimizations (torch.compile all modules, ~2x speed, requires CUDA)"
    )
    parser.add_argument(
        "--num-maskmem",
        type=int,
        default=7,
        help="Number of frames to keep in memory (VOS mode, prevents memory explosion)"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (max-autotune=fastest but slower startup)"
    )

    args = parser.parse_args()

    # Build text prompt
    if args.text_prompt == "custom":
        if args.custom_prompt is None:
            raise ValueError("--custom-prompt must be specified when using --text-prompt custom")
        text_prompt = [[p.strip() for p in args.custom_prompt.split(",")]]
    else:
        text_prompt = text_prompts[args.text_prompt]

    # Build config dictionary
    config = {
        "SOURCE": args.source,
        "TARGET_FPS": args.target_fps,
        "DETECTOR": {
            "MODEL_SRC": "huggingface",
            "MODEL": args.detector_model,
            "DEVICE": args.detector_device,
            "LOAD_DETS": False,
            "BATCH_SIZE": 1,
            "COMPILE": False,
            "USE_NMS": False,
            "TH_NMS": 0.95,
            "TH_DET": args.detection_threshold,
            "OUT_FORMAT": "xyxy",
            "TEXT_PROMPT": text_prompt,
            "CHECKPOINT_PATH": "",
            "CONFIG_PATH": ""
        },
        "SAM2": {
            "CHECKPOINT": args.sam2_checkpoint,
            "CONFIG": args.sam2_config,
            "COMPILE_ENCODER": args.compile_sam2,
            "DEVICE": args.sam2_device,
            "VOS_COMPILE": args.vos_optimize,
            "NUM_MASKMEM": args.num_maskmem,
            "COMPILE_MODE": args.compile_mode
        },
        "SAM2MOT": {
            "USE_ADAPTIVE_THRESHOLD": args.use_adaptive_threshold,
            "USE_OTSU": args.use_adaptive_threshold,
            "TH_OTSU": 0.1,
            "TH_DET": args.detection_threshold,
            "TH_OVERLAP": -0.5,
            "TH_HIGH_CONF": args.high_conf_threshold,
            "TH_MIN_GIOU": 0.2,
            "TH_MASK_EMPTY": 0.4,
            "COST_GIOU": 1,
            "TH_IOU_DIFF": 0.3,
            "TH_RELIABLE": 8,
            "TH_PENDING": 6,
            "TH_SUSPICIOUS": 2,
            "TOL_FRAMES": 25,
            "N_FRAMES": 10,
            "TH_MIOU": 0.8,
            "TH_SCORE_DIFF": 2,
            "TH_STD_DIFF": 0.2,
            "MASK_NMS": True,
            "TH_NMS_MIOU": 0.95
        },
        "VISUALIZATION": {
            "SHOW_MASKS": args.show_masks,
            "SHOW_BOXES": args.show_boxes,
            "SHOW_IDS": args.show_ids,
            "SHOW_SCORES": args.show_scores,
            "SHOW_FPS": args.show_fps,
            "MASK_ALPHA": args.mask_alpha
        },
        "OUTPUT": {
            "VIDEO": args.output_video,
            "FPS": args.output_fps
        }
    }

    return config


def main():
    """Main function for real-time streaming tracking"""

    # Parse arguments
    config = parse_args()

    print("=" * 80)
    print("SAM2 Real-time Multi-Object Tracking")
    print("=" * 80)
    print(f"Input source: {config['SOURCE']}")
    print(f"Detector: {config['DETECTOR']['MODEL']}")
    print(f"Text prompt: {config['DETECTOR']['TEXT_PROMPT']}")
    print(f"Detection threshold: {config['DETECTOR']['TH_DET']}")
    print(f"Adaptive threshold: {config['SAM2MOT']['USE_ADAPTIVE_THRESHOLD']}")
    print(f"Device: {config['SAM2']['DEVICE']}")
    print(f"VOS Optimization: {config['SAM2']['VOS_COMPILE']}")
    if config['SAM2']['VOS_COMPILE']:
        print(f"  - Compile mode: {config['SAM2']['COMPILE_MODE']}")
        print(f"  - Memory limit: {config['SAM2']['NUM_MASKMEM']} frames")
    print("=" * 80)
    print("\nPress 'q' to quit, 'p' to pause, 'r' to reset tracker")
    print()

    # Set device
    device = torch.device(config['SAM2']['DEVICE'])

    # Initialize tracker (VOS-optimized or standard)
    print("Initializing tracker...")
    if config['SAM2']['VOS_COMPILE']:
        print("Using VOS-optimized tracker (2x faster, initial warmup required)")
        tracker = OnlineSAM2TrackerVOS(config, device)
    else:
        print("Using standard tracker")
        tracker = OnlineSAM2Tracker(config, device)

    # Initialize input handler
    print("Opening input source...")
    input_handler = StreamingInputHandler(
        source=config['SOURCE'],
        target_fps=config['TARGET_FPS']
    )

    # Initialize visualizer
    print("Initializing visualizer...")
    visualizer = RealtimeVisualizer(
        window_name="SAM2 Real-time Tracking",
        show_masks=config['VISUALIZATION']['SHOW_MASKS'],
        show_boxes=config['VISUALIZATION']['SHOW_BOXES'],
        show_ids=config['VISUALIZATION']['SHOW_IDS'],
        show_scores=config['VISUALIZATION']['SHOW_SCORES'],
        show_fps=config['VISUALIZATION']['SHOW_FPS'],
        mask_alpha=config['VISUALIZATION']['MASK_ALPHA']
    )

    # Initialize video writer if output is specified
    video_writer = None
    if config['OUTPUT']['VIDEO']:
        print(f"Output video: {config['OUTPUT']['VIDEO']}")
        Path(config['OUTPUT']['VIDEO']).parent.mkdir(parents=True, exist_ok=True)
        video_writer = VideoWriter(
            output_path=config['OUTPUT']['VIDEO'],
            fps=config['OUTPUT']['FPS']
        )

    print("\nStarting real-time tracking...")
    print("=" * 80)

    try:
        frame_idx = -1
        paused = False

        for idx, frame, timestamp in input_handler:
            frame_idx = idx

            # Initialize tracker with first frame
            if frame_idx == 0:
                print("Initializing tracker with first frame...")
                detections = tracker.initialize(frame)
                video_segment = {}
            elif not paused:
                # Process frame
                video_segment, detections = tracker.process_frame(frame)

            # Get tracking info
            track_labels = tracker.get_current_tracks()
            adaptive_thresholds = tracker.get_adaptive_thresholds()

            # Prepare additional info for display
            additional_info = {
                "Frame": frame_idx,
                "Tracks": len(track_labels),
                "Detections": len(detections['boxes']) if 'boxes' in detections else 0
            }

            if config['SAM2MOT']['USE_ADAPTIVE_THRESHOLD']:
                additional_info["Det Th"] = f"{adaptive_thresholds['th_det']:.3f}"

            # Visualize
            frame_vis = visualizer.visualize_frame(
                frame,
                video_segment,
                detections,
                track_labels,
                additional_info
            )

            # Show frame
            key = visualizer.show(frame_vis)

            # Save to video if enabled
            if video_writer is not None:
                video_writer.write(frame_vis)

            # Handle keyboard input
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                print("\nResetting tracker...")
                tracker.reset()
                detections = tracker.initialize(frame)
                video_segment = {}
                paused = False

            # Print progress every 100 frames
            if frame_idx % 100 == 0 and frame_idx > 0:
                stats = input_handler.get_fps_stats()
                print(f"Frame {frame_idx}: {stats['fps']:.1f} FPS, "
                      f"{len(track_labels)} tracks")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        input_handler.release()
        visualizer.release()
        if video_writer is not None:
            video_writer.release()
        tracker.reset()

        # Print final statistics
        stats = input_handler.get_fps_stats()
        print("\n" + "=" * 80)
        print("Final Statistics:")
        print(f"  Total frames processed: {stats['frames_processed']}")
        print(f"  Total time: {stats['elapsed_time']:.2f}s")
        print(f"  Average FPS: {stats['fps']:.2f}")
        print("=" * 80)

    print("\nDone!")


if __name__ == '__main__':
    main()
