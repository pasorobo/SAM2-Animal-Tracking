"""
VOS-Optimized Online SAM2 Tracker for Real-time Streaming
Based on Gy920/segment-anything-2-real-time optimizations
"""

import numpy as np
import torch
import cv2
from PIL import Image
from collections import deque
from typing import Optional, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from utils.box_ops import mask_to_xyxy, generalized_box_iou, mask_iou
from detector import Detector
from online_sam2tracker import OnlineSAM2Tracker


class OnlineSAM2TrackerVOS(OnlineSAM2Tracker):
    """
    VOS-Optimized version of OnlineSAM2Tracker

    Key optimizations:
    1. torch.compile for all major SAM2 modules (2x speed improvement)
    2. Memory management with num_maskmem limit
    3. Efficient non-conditioning frame cleanup

    Based on Gy920's SAM2CameraPredictorVOS implementation.
    """

    def __init__(self, configs: dict, device: torch.device) -> None:
        """
        Initialize VOS-optimized online SAM2 tracker

        Args:
            configs (dict): model configs with VOS settings
            device (torch.device): processing device (cuda or cpu)
        """
        # Initialize parent class
        super().__init__(configs, device)

        # VOS-specific settings
        self.vos_compile = configs['SAM2'].get('VOS_COMPILE', True)
        self.num_maskmem = configs['SAM2'].get('NUM_MASKMEM', 7)
        self.compile_mode = configs['SAM2'].get('COMPILE_MODE', 'max-autotune')

        # Apply VOS optimizations if enabled
        if self.vos_compile and device.type == 'cuda':
            self._apply_vos_optimizations()
            print(f"VOS optimizations applied: torch.compile with mode='{self.compile_mode}'")
        else:
            if device.type != 'cuda':
                print("Warning: VOS compile optimization requires CUDA. Skipping compilation.")
            else:
                print("VOS compile optimization disabled")

        print(f"Memory management: num_maskmem={self.num_maskmem}")

    def _apply_vos_optimizations(self):
        """
        Apply VOS optimizations using torch.compile

        Compiles:
        - Memory encoder
        - Memory attention
        - Prompt encoder (SAM)
        - Mask decoder (SAM)

        Note: Initial inference will be slower due to compilation overhead,
        but subsequent frames will be significantly faster.
        """
        try:
            # Access SAM2 internal models
            sam2_model = self.sam2_predictor.model

            # Compile memory encoder
            if hasattr(sam2_model, 'memory_encoder'):
                print("Compiling memory encoder...")
                sam2_model.memory_encoder = torch.compile(
                    sam2_model.memory_encoder,
                    mode=self.compile_mode,
                    fullgraph=False  # Allow dynamic shapes
                )

            # Compile memory attention
            if hasattr(sam2_model, 'memory_attention'):
                print("Compiling memory attention...")
                sam2_model.memory_attention = torch.compile(
                    sam2_model.memory_attention,
                    mode=self.compile_mode,
                    fullgraph=False
                )

            # Compile SAM prompt encoder
            if hasattr(sam2_model, 'sam_prompt_encoder'):
                print("Compiling SAM prompt encoder...")
                sam2_model.sam_prompt_encoder = torch.compile(
                    sam2_model.sam_prompt_encoder,
                    mode=self.compile_mode,
                    fullgraph=False
                )

            # Compile SAM mask decoder
            if hasattr(sam2_model, 'sam_mask_decoder'):
                print("Compiling SAM mask decoder...")
                sam2_model.sam_mask_decoder = torch.compile(
                    sam2_model.sam_mask_decoder,
                    mode=self.compile_mode,
                    fullgraph=False
                )

            print("✓ VOS compilation complete")
            print("  Note: First inference will be slower (warmup), subsequent frames will be faster")

        except Exception as e:
            print(f"Warning: Failed to apply VOS optimizations: {e}")
            print("  Continuing with standard (non-compiled) mode")

    def _manage_memory_objects(self, current_frame_idx: int):
        """
        Manage memory for all tracked objects

        Removes old non-conditioning frame outputs when they exceed num_maskmem limit.
        This prevents memory explosion during long-running sessions.

        Args:
            current_frame_idx: Current frame index
        """
        if self.inference_state is None or self.num_maskmem <= 0:
            return

        # Iterate through all tracked objects
        for obj_idx, obj_output_dict in enumerate(
            self.inference_state.get('output_dict_per_obj', {}).values()
        ):
            non_cond_outputs = obj_output_dict.get('non_cond_frame_outputs', {})

            # Check if we exceed memory limit
            if len(non_cond_outputs) > self.num_maskmem:
                # Get frames to remove (oldest frames beyond limit)
                frame_indices = sorted(non_cond_outputs.keys())
                num_to_remove = len(frame_indices) - self.num_maskmem

                frames_to_remove = frame_indices[:num_to_remove]

                # Remove old frames
                for frame_idx in frames_to_remove:
                    if frame_idx in non_cond_outputs:
                        # Delete the frame output
                        del non_cond_outputs[frame_idx]

                # Optional: Clear CUDA cache periodically
                if obj_idx == 0 and len(frames_to_remove) > 0:
                    torch.cuda.empty_cache()

    def process_frame(self, frame: Image.Image) -> Tuple[Dict, Dict]:
        """
        Process a single frame with VOS optimizations

        Overrides parent method to add memory management.

        Args:
            frame: Input frame as PIL Image

        Returns:
            Tuple of (video_segments, detections) for current frame
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        self.frame_idx += 1

        # Save frame for SAM2
        frame_path = os.path.join(self.temp_dir, f"{self.frame_idx:05d}.jpg")
        frame.save(frame_path)

        # Detect objects
        detections = self.det_model([frame], self.frame_idx)[0]

        # Update adaptive thresholds
        if len(detections['scores']) > 0:
            self._update_adaptive_thresholds(detections['scores'])

        # Propagate SAM2 to current frame
        try:
            # Run propagation for single frame
            generator = self.sam2_predictor.propagate_in_video(self.inference_state)

            # Get results for current frame
            out_frame_idx, out_obj_ids, out_mask_logits = next(generator)

            # ===== VOS OPTIMIZATION: Memory Management =====
            # Clean up old frame outputs to prevent memory explosion
            self._manage_memory_objects(out_frame_idx)
            # ================================================

            # Extract score logits
            score_logits = {}
            for out_obj_id in out_obj_ids:
                try:
                    scores = self.inference_state["output_dict_per_obj"][
                        self.inference_state["obj_id_to_idx"].get(out_obj_id, None)
                    ]["non_cond_frame_outputs"][out_frame_idx]['object_score_logits']
                except:
                    try:
                        scores = self.inference_state["output_dict_per_obj"][
                            self.inference_state["obj_id_to_idx"].get(out_obj_id, None)
                        ]["cond_frame_outputs"][out_frame_idx]['object_score_logits']
                    except:
                        scores = torch.tensor(5.0)  # Default reliable score
                score_logits[out_obj_id] = scores

            # Create video segment for current frame
            video_segment = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Apply SAM2MOT functions
            new_objects, indices, ids = self._object_addition(video_segment, detections)
            recon_objects = self._quality_reconstruction(detections, indices, ids, score_logits)
            remove_ids = self._object_removal(score_logits)

            # Mask NMS
            if self.mask_nms:
                remove_nms_ids = self._mask_non_max_supp(out_mask_logits, out_obj_ids)
                remove_ids.update(remove_nms_ids)

            # Cross object interaction
            remove_mem_ids = self._cross_object_interaction(score_logits, video_segment)

            # Object addition
            label_mod = 0
            for new_id, (new_box, new_label) in new_objects.items():
                _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                    self.inference_state,
                    out_frame_idx,
                    new_id + label_mod,
                    box=new_box.float().numpy(),
                )

                # Check if candidate fulfills requirements
                new_obj_mask = (out_mask_logits[-1][0] > 0).cpu().numpy()
                for mask in out_mask_logits[:-1]:
                    mask = (mask[0] > 0).cpu().numpy()
                    miou = np.sum(new_obj_mask * mask) / np.sum(new_obj_mask)

                    if miou > self.th_mask_empty:
                        self.sam2_predictor.remove_object(
                            self.inference_state, new_id + label_mod, strict=True, need_output=False
                        )
                        label_mod -= 1
                        out_mask_logits = out_mask_logits[:-1, ...]
                        out_obj_ids = out_obj_ids[:-1]
                        break

                self.track_labels[new_id] = new_label

            # Cross object interaction - remove from memory
            for id in remove_mem_ids:
                if out_frame_idx in self.inference_state['output_dict_per_obj'][
                    self.inference_state['obj_id_to_idx'][id]
                ]["non_cond_frame_outputs"]:
                    del self.inference_state['output_dict_per_obj'][
                        self.inference_state['obj_id_to_idx'][id]
                    ]["non_cond_frame_outputs"][out_frame_idx]

            # Quality reconstruction
            for recon_id, recon_box in recon_objects.items():
                if recon_id in remove_mem_ids:
                    continue

                # Get boxes from masks and calculate IoUs
                iou_box = []
                for id, mask in video_segment.items():
                    mask_box = mask_to_xyxy(mask)
                    iou_box.append(
                        generalized_box_iou(recon_box.unsqueeze(0), torch.tensor(mask_box))
                    )

                if len(iou_box) == 1:
                    continue

                # Check difference between best and second-best match
                first_to_sec = max(iou_box) - sorted(iou_box)[-2]
                if first_to_sec < self.th_iou_diff:
                    continue

                # Reconstruct
                self.sam2_predictor.remove_object(
                    self.inference_state, recon_id, strict=True, need_output=False
                )
                _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                    self.inference_state,
                    out_frame_idx,
                    recon_id.item(),
                    box=recon_box.float().numpy(),
                )

            # Object removal
            for id in remove_ids:
                if len(out_obj_ids) - len(remove_ids) > 0:
                    if id not in out_obj_ids:
                        continue
                    self.sam2_predictor.remove_object(
                        self.inference_state, id, strict=True, need_output=False
                    )

            # Get final masks for current frame
            video_segment = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        except StopIteration:
            # No more frames
            video_segment = {}

        return video_segment, detections

    def get_memory_stats(self) -> Dict:
        """
        Get memory usage statistics

        Returns:
            Dictionary with memory statistics for each tracked object
        """
        if self.inference_state is None:
            return {}

        stats = {
            'num_maskmem_limit': self.num_maskmem,
            'objects': {}
        }

        for obj_id, obj_output_dict in self.inference_state.get('output_dict_per_obj', {}).items():
            non_cond_count = len(obj_output_dict.get('non_cond_frame_outputs', {}))
            cond_count = len(obj_output_dict.get('cond_frame_outputs', {}))

            stats['objects'][obj_id] = {
                'non_cond_frames': non_cond_count,
                'cond_frames': cond_count,
                'total_frames': non_cond_count + cond_count,
                'memory_limited': non_cond_count >= self.num_maskmem
            }

        return stats
