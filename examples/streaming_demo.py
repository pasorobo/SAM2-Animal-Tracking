"""
Simple Streaming Demo Script
Minimal example of real-time tracking with SAM2
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from tracker.streaming_input import StreamingInputHandler
from tracker.online_sam2tracker import OnlineSAM2Tracker
from tracker.realtime_visualizer import RealtimeVisualizer


def minimal_streaming_demo(source="0", text_prompt="animal"):
    """
    Minimal streaming tracking demo

    Args:
        source: Camera index or video file path
        text_prompt: What to detect (e.g., "animal", "person")
    """
    print("=" * 60)
    print("SAM2 Streaming Tracking - Minimal Demo")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Detecting: {text_prompt}")
    print("\nPress 'q' to quit")
    print("=" * 60)

    # Simple configuration
    config = {
        "DETECTOR": {
            "MODEL_SRC": "huggingface",
            "MODEL": "iSEE-Laboratory/llmdet_large",
            "DEVICE": "cuda",
            "LOAD_DETS": False,
            "BATCH_SIZE": 1,
            "COMPILE": False,
            "USE_NMS": False,
            "TH_NMS": 0.95,
            "TH_DET": 0.3,
            "OUT_FORMAT": "xyxy",
            "TEXT_PROMPT": [[text_prompt]],
            "CHECKPOINT_PATH": "",
            "CONFIG_PATH": ""
        },
        "SAM2": {
            "CHECKPOINT": "sam2/checkpoints/sam2.1_hiera_large.pt",
            "CONFIG": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "COMPILE_ENCODER": True,
        },
        "SAM2MOT": {
            "USE_ADAPTIVE_THRESHOLD": True,
            "USE_OTSU": True,
            "TH_OTSU": 0.1,
            "TH_DET": 0.3,
            "TH_OVERLAP": -0.5,
            "TH_HIGH_CONF": 0.4,
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
        }
    }

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("Loading tracker...")
    tracker = OnlineSAM2Tracker(config, device)

    print("Opening video source...")
    input_handler = StreamingInputHandler(source=source)

    print("Initializing visualizer...")
    visualizer = RealtimeVisualizer()

    print("\nStarting tracking...\n")

    try:
        for frame_idx, frame, timestamp in input_handler:
            # Initialize on first frame
            if frame_idx == 0:
                print("Initializing with first frame...")
                detections = tracker.initialize(frame)
                video_segment = {}
            else:
                # Process frame
                video_segment, detections = tracker.process_frame(frame)

            # Get tracking info
            track_labels = tracker.get_current_tracks()

            # Visualize
            frame_vis = visualizer.visualize_frame(
                frame, video_segment, detections, track_labels
            )

            # Display
            key = visualizer.show(frame_vis)

            # Quit on 'q'
            if key == ord('q'):
                break

            # Print status every 30 frames
            if frame_idx % 30 == 0 and frame_idx > 0:
                print(f"Frame {frame_idx}: {len(track_labels)} tracks")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        input_handler.release()
        visualizer.release()
        tracker.reset()
        print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple streaming demo")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or video file")
    parser.add_argument("--prompt", type=str, default="animal",
                        help="Text prompt for detection")
    args = parser.parse_args()

    minimal_streaming_demo(args.source, args.prompt)
