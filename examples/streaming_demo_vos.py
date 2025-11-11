"""
VOS-Optimized Streaming Demo Script
Demonstrates 2x faster tracking with VOS optimizations
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from tracker.streaming_input import StreamingInputHandler
from tracker.online_sam2tracker_vos import OnlineSAM2TrackerVOS
from tracker.realtime_visualizer import RealtimeVisualizer


def vos_optimized_demo(source="0", text_prompt="animal"):
    """
    VOS-optimized streaming tracking demo with 2x speed improvement

    Features:
    - torch.compile for all SAM2 modules (memory encoder, attention, etc.)
    - Memory management with num_maskmem limit
    - Faster inference after initial warmup

    Args:
        source: Camera index or video file path
        text_prompt: What to detect (e.g., "animal", "person")
    """
    print("=" * 60)
    print("SAM2 VOS-Optimized Streaming Tracking Demo")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Detecting: {text_prompt}")
    print("\nVOS Optimizations:")
    print("  ✓ torch.compile (all modules)")
    print("  ✓ Memory management (7 frame limit)")
    print("  ✓ ~2x speed improvement (after warmup)")
    print("\nNote: First few frames will be slower (compilation)")
    print("\nPress 'q' to quit, 'p' to pause")
    print("=" * 60)

    # VOS-optimized configuration
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
            # VOS-specific settings
            "VOS_COMPILE": True,  # Enable VOS optimizations
            "NUM_MASKMEM": 7,  # Memory limit
            "COMPILE_MODE": "max-autotune",  # Maximum optimization
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

    if device.type != "cuda":
        print("\nWarning: VOS optimizations require CUDA!")
        print("Falling back to standard mode (slower)")
        config["SAM2"]["VOS_COMPILE"] = False

    print("\nLoading VOS-optimized tracker...")
    print("(This may take a moment for compilation...)")
    tracker = OnlineSAM2TrackerVOS(config, device)

    print("\nOpening video source...")
    input_handler = StreamingInputHandler(source=source)

    print("Initializing visualizer...")
    visualizer = RealtimeVisualizer()

    print("\n" + "=" * 60)
    print("Starting tracking...")
    print("Note: First few frames may be slow (warmup)")
    print("=" * 60 + "\n")

    warmup_done = False

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

                # Notify after warmup
                if frame_idx == 5 and not warmup_done:
                    print("\n✓ Warmup complete! Speed should improve now.\n")
                    warmup_done = True

            # Get tracking info
            track_labels = tracker.get_current_tracks()

            # Get memory stats (VOS-specific)
            if frame_idx % 100 == 0 and frame_idx > 0:
                mem_stats = tracker.get_memory_stats()
                print(f"\nFrame {frame_idx}:")
                print(f"  Tracks: {len(track_labels)}")
                print(f"  Memory limit: {mem_stats['num_maskmem_limit']} frames")
                for obj_id, stats in list(mem_stats['objects'].items())[:3]:  # Show first 3
                    print(f"  Object {obj_id}: {stats['non_cond_frames']} frames in memory")

            # Visualize
            frame_vis = visualizer.visualize_frame(
                frame, video_segment, detections, track_labels
            )

            # Display
            key = visualizer.show(frame_vis)

            # Quit on 'q', pause on 'p'
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("\nPaused. Press 'p' again to resume...")
                while True:
                    key = visualizer.show(frame_vis)
                    if key == ord('p'):
                        print("Resumed")
                        break
                    elif key == ord('q'):
                        break

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

        # Print final stats
        stats = input_handler.get_fps_stats()
        print("\n" + "=" * 60)
        print("Session Summary:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Total time: {stats['elapsed_time']:.2f}s")
        print(f"  Average FPS: {stats['fps']:.2f}")
        print("=" * 60)
        print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VOS-optimized streaming demo")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or video file")
    parser.add_argument("--prompt", type=str, default="animal",
                        help="Text prompt for detection")
    args = parser.parse_args()

    vos_optimized_demo(args.source, args.prompt)
