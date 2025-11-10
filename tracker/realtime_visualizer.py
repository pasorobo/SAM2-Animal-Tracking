"""
Real-time Visualizer for Streaming Tracking
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Dict, Optional
import time


class RealtimeVisualizer:
    """
    Real-time visualization for streaming tracking
    Displays tracked objects with masks, bounding boxes, and IDs
    """

    def __init__(
        self,
        window_name: str = "SAM2 Animal Tracking",
        show_masks: bool = True,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_scores: bool = True,
        show_fps: bool = True,
        mask_alpha: float = 0.6,
        font_scale: float = 0.6
    ):
        """
        Initialize visualizer

        Args:
            window_name: Name of display window
            show_masks: Display segmentation masks
            show_boxes: Display bounding boxes
            show_ids: Display track IDs
            show_scores: Display detection scores
            show_fps: Display FPS counter
            mask_alpha: Transparency of masks (0-1)
            font_scale: Font size scale
        """
        self.window_name = window_name
        self.show_masks = show_masks
        self.show_boxes = show_boxes
        self.show_ids = show_ids
        self.show_scores = show_scores
        self.show_fps = show_fps
        self.mask_alpha = mask_alpha
        self.font_scale = font_scale

        # FPS tracking
        self.frame_times = []
        self.fps = 0

        # Color map for different objects
        self.cmap = plt.get_cmap("tab20")

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def visualize_frame(
        self,
        frame: Image.Image,
        video_segment: Dict,
        detections: Dict,
        track_labels: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Visualize tracking results on frame

        Args:
            frame: Input frame (PIL Image)
            video_segment: Dictionary of {track_id: mask}
            detections: Detection results
            track_labels: Dictionary of {track_id: class_label}
            additional_info: Additional info to display (e.g., thresholds)

        Returns:
            Visualized frame as numpy array (BGR for cv2.imshow)
        """
        # Update FPS
        self._update_fps()

        # Convert PIL to numpy
        frame_np = np.array(frame)

        # Create overlay for masks
        if self.show_masks:
            overlay = np.zeros_like(frame_np)

            for track_id, mask in video_segment.items():
                # Skip filler objects
                if track_id < 0:
                    continue

                # Get color for this track
                color = self._get_track_color(track_id)

                # Apply mask
                mask_2d = mask.squeeze()
                overlay[mask_2d] = color

            # Blend overlay with frame
            frame_np = cv2.addWeighted(
                frame_np,
                1.0,
                overlay,
                self.mask_alpha,
                0
            )

        # Draw bounding boxes and IDs
        if self.show_boxes or self.show_ids:
            for track_id, mask in video_segment.items():
                # Skip filler objects
                if track_id < 0:
                    continue

                # Get bounding box from mask
                mask_2d = mask.squeeze()
                rows, cols = np.where(mask_2d)

                if len(rows) > 0 and len(cols) > 0:
                    x_min, x_max = np.min(cols), np.max(cols)
                    y_min, y_max = np.min(rows), np.max(rows)

                    color = self._get_track_color(track_id)

                    # Draw bounding box
                    if self.show_boxes:
                        cv2.rectangle(
                            frame_np,
                            (x_min, y_min),
                            (x_max, y_max),
                            color,
                            2
                        )

                    # Draw track ID and label
                    if self.show_ids:
                        label = f"ID:{track_id}"
                        if track_labels and track_id in track_labels:
                            label += f" ({track_labels[track_id]})"

                        # Draw text background
                        (text_w, text_h), _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale,
                            2
                        )
                        cv2.rectangle(
                            frame_np,
                            (x_min, y_min - text_h - 10),
                            (x_min + text_w, y_min),
                            color,
                            -1
                        )
                        # Draw text
                        cv2.putText(
                            frame_np,
                            label,
                            (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale,
                            (255, 255, 255),
                            2
                        )

        # Draw detection boxes if enabled
        if self.show_scores and len(detections.get('boxes', [])) > 0:
            for i, box in enumerate(detections['boxes']):
                score = detections['scores'][i].item()
                if score > 0:  # Skip filler detections
                    # Draw semi-transparent detection box
                    x1, y1, x2, y2 = box[:4].int().tolist()
                    cv2.rectangle(
                        frame_np,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1
                    )
                    # Draw score
                    cv2.putText(
                        frame_np,
                        f"{score:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1
                    )

        # Draw FPS counter
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame_np,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        # Draw additional info
        if additional_info:
            y_offset = 60
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                cv2.putText(
                    frame_np,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset += 25

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        return frame_bgr

    def show(self, frame_bgr: np.ndarray) -> int:
        """
        Display frame and handle keyboard input

        Args:
            frame_bgr: Frame to display (BGR format)

        Returns:
            Key code pressed (-1 if no key pressed)
        """
        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1)

    def _get_track_color(self, track_id: int) -> tuple:
        """
        Get consistent color for track ID

        Args:
            track_id: Track identifier

        Returns:
            RGB color tuple
        """
        cmap_idx = track_id % 20
        color_normalized = self.cmap(cmap_idx)[:3]
        color = tuple(int(c * 255) for c in color_normalized)
        return color

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only last second of frame times
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]

        # Calculate FPS
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (current_time - self.frame_times[0])

    def release(self):
        """Release visualizer resources"""
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class VideoWriter:
    """
    Video writer for saving tracking results
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer

        Args:
            output_path: Output video file path
            fps: Frames per second
            codec: Video codec (e.g., 'mp4v', 'XVID')
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None

    def write(self, frame: np.ndarray):
        """
        Write frame to video

        Args:
            frame: Frame to write (BGR format)
        """
        if self.writer is None:
            # Initialize writer with first frame
            self.frame_size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )

        self.writer.write(frame)

    def release(self):
        """Release video writer"""
        if self.writer is not None:
            self.writer.release()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def __del__(self):
        """Destructor"""
        self.release()
