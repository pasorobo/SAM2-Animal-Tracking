"""
Streaming Input Handler for Real-time Tracking
Supports USB cameras and video files
"""

import cv2
import numpy as np
from PIL import Image
from typing import Iterator, Tuple, Optional
import time


class StreamingInputHandler:
    """Handles input from USB cameras or video files for real-time processing"""

    def __init__(self, source: str = "0", target_fps: Optional[int] = None):
        """
        Initialize streaming input handler

        Args:
            source: Camera index (e.g., "0", "1") or video file path
            target_fps: Target FPS for processing (None = process all frames)
        """
        self.source = source
        self.target_fps = target_fps
        self.cap = None
        self.is_camera = False
        self.frame_count = 0
        self.start_time = None

        # Try to parse as camera index
        try:
            camera_idx = int(source)
            self.is_camera = True
            self.cap = cv2.VideoCapture(camera_idx)
        except ValueError:
            # Treat as video file path
            self.is_camera = False
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # For video files, get total frame count
        if not self.is_camera:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = None

        # Calculate frame skip for target FPS
        if self.target_fps and self.source_fps > 0:
            self.frame_skip = max(1, int(self.source_fps / self.target_fps))
        else:
            self.frame_skip = 1

        print(f"Streaming input initialized:")
        print(f"  Source: {'Camera ' + source if self.is_camera else source}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Source FPS: {self.source_fps}")
        if self.target_fps:
            print(f"  Target FPS: {self.target_fps} (frame skip: {self.frame_skip})")
        if self.total_frames:
            print(f"  Total frames: {self.total_frames}")

    def __iter__(self) -> Iterator[Tuple[int, Image.Image, float]]:
        """
        Iterate through frames

        Yields:
            Tuple of (frame_index, PIL.Image, timestamp)
        """
        self.frame_count = 0
        self.start_time = time.time()
        frame_idx = 0

        while True:
            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                if self.is_camera:
                    print("Warning: Failed to read camera frame")
                    continue
                else:
                    # End of video file
                    break

            # Skip frames if needed
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Calculate timestamp
            timestamp = time.time() - self.start_time

            yield frame_idx, pil_image, timestamp

            frame_idx += 1
            self.frame_count += 1

    def read_frame(self) -> Optional[Tuple[Image.Image, float]]:
        """
        Read a single frame (useful for manual frame-by-frame processing)

        Returns:
            Tuple of (PIL.Image, timestamp) or None if failed
        """
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Calculate timestamp
        if self.start_time is None:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time

        self.frame_count += 1

        return pil_image, timestamp

    def get_fps_stats(self) -> dict:
        """
        Get FPS statistics

        Returns:
            Dictionary with FPS information
        """
        if self.start_time is None or self.frame_count == 0:
            return {"fps": 0, "frames_processed": 0, "elapsed_time": 0}

        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            "fps": fps,
            "frames_processed": self.frame_count,
            "elapsed_time": elapsed
        }

    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def __del__(self):
        """Destructor"""
        self.release()


class FrameBuffer:
    """Simple frame buffer for temporal processing"""

    def __init__(self, max_size: int = 10):
        """
        Initialize frame buffer

        Args:
            max_size: Maximum number of frames to keep in buffer
        """
        self.max_size = max_size
        self.buffer = []
        self.frame_indices = []

    def add(self, frame_idx: int, frame: Image.Image):
        """Add frame to buffer"""
        self.buffer.append(frame)
        self.frame_indices.append(frame_idx)

        # Remove oldest frame if buffer is full
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.frame_indices.pop(0)

    def get(self, idx: int = -1) -> Optional[Image.Image]:
        """Get frame from buffer (default: most recent)"""
        if not self.buffer:
            return None
        return self.buffer[idx]

    def get_all(self) -> list:
        """Get all frames in buffer"""
        return self.buffer

    def clear(self):
        """Clear buffer"""
        self.buffer = []
        self.frame_indices = []

    def __len__(self):
        return len(self.buffer)
