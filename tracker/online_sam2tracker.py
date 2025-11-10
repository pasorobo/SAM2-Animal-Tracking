"""
Online SAM2 Tracker for Real-time Streaming
Modified from sam2tracker.py for frame-by-frame processing
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


class OnlineSAM2Tracker:
    """
    Online version of SAM2Tracker for real-time streaming
    Processes frames one at a time without requiring the full sequence upfront
    """

    def __init__(self, configs: dict, device: torch.device) -> None:
        """
        Initialize online SAM2 tracker

        Args:
            configs (dict): model configs
            device (torch.device): processing device (cuda or cpu)
        """
        # Initialize models
        self.sam2_predictor = self._init_sam2(configs['SAM2'], device)
        self.det_model = Detector(configs['DETECTOR'])

        # Object addition properties
        self.use_adaptive_threshold = configs['SAM2MOT'].get('USE_ADAPTIVE_THRESHOLD', True)
        self.th_det = configs['SAM2MOT']['TH_DET']
        self.th_high_conf = configs['SAM2MOT']['TH_HIGH_CONF']
        self.th_mask_empty = configs['SAM2MOT']['TH_MASK_EMPTY']
        self.th_overlap = configs['SAM2MOT']['TH_OVERLAP']

        # Object removal and quality reconstruction properties
        self.th_iou_diff = configs['SAM2MOT']['TH_IOU_DIFF']
        self.th_reliable = configs['SAM2MOT']['TH_RELIABLE']
        self.th_pending = configs['SAM2MOT']['TH_PENDING']
        self.th_suspicious = configs['SAM2MOT']['TH_SUSPICIOUS']
        self.tol_frames = configs['SAM2MOT']['TOL_FRAMES']
        self.lost_count = {}

        # Cross object interaction properties
        self.n_frames = configs['SAM2MOT']['N_FRAMES']
        self.th_miou = configs['SAM2MOT']['TH_MIOU']
        self.th_score_diff = configs['SAM2MOT']['TH_SCORE_DIFF']
        self.th_std_diff = configs['SAM2MOT']['TH_STD_DIFF']
        self.score_queue = deque(maxlen=self.n_frames)

        # Mask NMS
        self.mask_nms = configs['SAM2MOT']['MASK_NMS']
        self.th_nms_miou = configs['SAM2MOT']['TH_NMS_MIOU']

        # Multiclass handling
        self.track_labels = {}
        self.known_ids = set()

        # Online processing state
        self.inference_state = None
        self.frame_idx = -1
        self.is_initialized = False

        # Adaptive threshold for online processing (moving statistics)
        self.detection_score_history = deque(maxlen=100)  # Keep last 100 detection scores
        self.adaptive_th_det = self.th_det
        self.adaptive_th_high_conf = self.th_high_conf

    def _init_sam2(self, sam2_configs: dict, device: torch.device) -> SAM2VideoPredictor:
        """
        Initialize SAM2

        Args:
            sam2_configs (dict): SAM2 configuration
            device (torch.device): processing device

        Returns:
            SAM2VideoPredictor
        """
        if sam2_configs['COMPILE_ENCODER']:
            extras = ['+compile_image_encoder=True']
        else:
            extras = []

        sam2_predictor = build_sam2_video_predictor(
            sam2_configs['CONFIG'],
            sam2_configs['CHECKPOINT'],
            hydra_overrides_extra=extras,
            device=device
        )
        return sam2_predictor

    def initialize(self, first_frame: Image.Image, video_source: str = "streaming"):
        """
        Initialize tracker with first frame

        Args:
            first_frame: First frame as PIL Image
            video_source: Source identifier (for SAM2 state)
        """
        # Create temporary directory for SAM2 (it expects video frames)
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

        # Save first frame
        frame_path = os.path.join(self.temp_dir, "00000.jpg")
        first_frame.save(frame_path)

        # Initialize SAM2 state
        self.inference_state = self.sam2_predictor.init_state(video_path=self.temp_dir)
        self.frame_idx = 0

        # Detect objects in first frame
        detections = self.det_model([first_frame], 0)[0]

        # Update adaptive thresholds with initial detections
        if len(detections['scores']) > 0:
            self._update_adaptive_thresholds(detections['scores'])

        input_boxes = detections['boxes']

        if len(input_boxes) == 0:
            # No detections, add dummy object (SAM2 requires at least one object)
            detections = {
                'scores': torch.tensor([0]),
                'boxes': torch.tensor([0, 0, 0, 0]).reshape((1, 4)),
                'labels': torch.tensor([-1])
            }
            input_boxes = detections['boxes']

        # Prompt SAM2 with detected boxes
        for object_id, box in enumerate(input_boxes):
            _, _, _ = self.sam2_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=object_id if torch.any(box != 0) else -1,
                box=box.float().numpy(),
            )
            if torch.any(box != 0):
                self.track_labels[object_id] = int(detections['labels'][object_id])
                self.known_ids.add(object_id)

        self.is_initialized = True

        return detections

    def _update_adaptive_thresholds(self, scores: torch.Tensor):
        """
        Update adaptive thresholds based on detection score history

        Args:
            scores: Detection scores from current frame
        """
        if not self.use_adaptive_threshold:
            return

        # Add scores to history
        self.detection_score_history.extend(scores.cpu().numpy().tolist())

        if len(self.detection_score_history) < 10:
            # Not enough history yet
            return

        # Calculate adaptive threshold using Otsu-like method on accumulated scores
        scores_array = np.array(list(self.detection_score_history))
        scores_scaled = (255 * scores_array).astype(np.uint8).reshape(-1, 1)

        try:
            otsu_threshold, _ = cv2.threshold(
                scores_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.adaptive_th_det = otsu_threshold / 255.0
            self.adaptive_th_high_conf = self.adaptive_th_det + 0.1
            self.det_model.th_det = self.adaptive_th_det
        except:
            # If Otsu fails, use default values
            pass

    def process_frame(self, frame: Image.Image) -> Tuple[Dict, Dict]:
        """
        Process a single frame

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

    def _object_addition(self, video_segment: dict, detections: dict) -> tuple:
        """Add new objects (same logic as original SAM2Tracker)"""
        # This is the same as sam2tracker.py:200-286
        ids_list = list(video_segment.keys())
        self.known_ids.update(ids_list)
        ids = np.array(ids_list)
        masks = np.array(list(video_segment.values())).squeeze(1)
        sam_boxes = torch.from_numpy(mask_to_xyxy(masks)).to(dtype=torch.float32)

        nonzero_mask = sam_boxes.abs().sum(dim=1) != 0
        sam_boxes = sam_boxes[nonzero_mask, :]
        ids = ids[np.array(nonzero_mask)]

        boxes = detections["boxes"]
        scores = detections["scores"].cpu()
        labels = detections["labels"].cpu()

        cand_mask = scores >= self.adaptive_th_high_conf
        high_conf_scores = scores[cand_mask]
        high_conf_boxes = boxes[cand_mask, :]
        high_conf_labels = labels[cand_mask]

        if len(high_conf_boxes) == 0 or len(sam_boxes) == 0:
            return {}, (np.array([]), np.array([])), ids

        from scipy.optimize import linear_sum_assignment
        cost_giou = -generalized_box_iou(high_conf_boxes, sam_boxes)
        C = cost_giou

        raw_indices = linear_sum_assignment(C)

        row_ind = []
        col_ind = []
        for i in range(len(raw_indices[0])):
            if C[raw_indices[0][i], raw_indices[1][i]] < -0.2:
                row_ind.append(raw_indices[0][i])
                col_ind.append(raw_indices[1][i])
        indices = (np.array(row_ind), np.array(col_ind))

        cand_boxes = [b for i, b in enumerate(high_conf_boxes) if i not in indices[0]]

        if not cand_boxes:
            return {}, indices, ids

        max_id = max(self.known_ids) if self.known_ids else 0
        new_ids = [max_id + i + 1 for i in range(len(cand_boxes))]
        new_labels = [label for i, label in enumerate(high_conf_labels) if i not in indices[0]]
        new_objects = {
            id: (box, label) for id, box, label in zip(new_ids, cand_boxes, new_labels)
        }

        return new_objects, indices, ids

    def _object_removal(self, seg_scores: dict) -> set:
        """Remove lost objects (same logic as original SAM2Tracker)"""
        remove_ids = []
        for id, score in seg_scores.items():
            if score <= self.th_suspicious:
                self.lost_count[id] = self.lost_count.get(id, 0) + 1
                if self.lost_count[id] >= self.tol_frames:
                    remove_ids.append(id)
            elif score > self.th_pending:
                self.lost_count[id] = 0

        return set(remove_ids)

    def _quality_reconstruction(self, detections: dict, indices: tuple, ids: np.ndarray, seg_scores: dict) -> dict:
        """Reconstruct uncertain objects (same logic as original SAM2Tracker)"""
        if len(indices[0]) == 0:
            return {}

        boxes = detections["boxes"].cpu()
        scores = detections["scores"].cpu()
        labels = detections["labels"].cpu()

        high_conf_mask = scores >= self.adaptive_th_high_conf
        high_conf_boxes = boxes[high_conf_mask, :]
        high_conf_labels = labels[high_conf_mask]

        matched_ids = ids[indices[1]]
        recon_cand_boxes = high_conf_boxes[indices[0], :]
        recon_labels = high_conf_labels[indices[0]]

        recon_mask = [
            True if self.th_pending <= seg_scores[id].item() <= self.th_reliable else False
            for id in matched_ids
        ]
        recon_boxes = recon_cand_boxes[recon_mask, :]
        recon_ids = matched_ids[recon_mask]

        box_labels = recon_labels[recon_mask]
        track_labels = np.array([self.track_labels[rid] for rid in recon_ids])
        label_mask = box_labels == track_labels
        recon_ids = [recon_ids[i] for i in range(len(recon_ids)) if label_mask[i]]
        recon_boxes = [recon_boxes[i] for i in range(len(recon_boxes)) if label_mask[i]]

        recon_objects = {id: box for id, box in zip(recon_ids, recon_boxes)}

        return recon_objects

    def _cross_object_interaction(self, score_logits: dict, video_segment: dict) -> set:
        """Handle cross-object interactions (same logic as original SAM2Tracker)"""
        self.score_queue.append(score_logits)

        ids = list(video_segment.keys())
        masks = list(video_segment.values())

        cands = []
        remove_mem = []

        for i in range(len(ids)):
            for j in range(len(ids)):
                if i <= j:
                    continue
                m_iou = mask_iou(masks[i], masks[j])
                if m_iou > self.th_miou:
                    cands.append((i, j))

        for (cand_i, cand_j) in cands:
            scores_i = torch.tensor([
                scores.get(ids[cand_i]) for scores in self.score_queue
                if scores.get(ids[cand_i]) is not None
            ])
            scores_j = torch.tensor([
                scores.get(ids[cand_j]) for scores in self.score_queue
                if scores.get(ids[cand_j]) is not None
            ])

            if len(scores_i) == 0 or len(scores_j) == 0:
                continue

            std_i = torch.std(scores_i)
            std_j = torch.std(scores_j)

            if scores_i[0] - scores_j[0] > self.th_score_diff:
                remove_mem.append(ids[cand_j])
            elif scores_j[0] - scores_i[0] > self.th_score_diff:
                remove_mem.append(ids[cand_i])
            elif std_i - std_j > self.th_std_diff:
                remove_mem.append(ids[cand_i])
            elif std_j - std_i > self.th_std_diff:
                remove_mem.append(ids[cand_j])

        return set(remove_mem)

    def _mask_non_max_supp(self, mask_logits: torch.Tensor, ids: list) -> list:
        """Perform mask NMS (same logic as original SAM2Tracker)"""
        remove_nms_ids = []

        for i, id1 in enumerate(ids):
            mask1 = (mask_logits[i, 0, :, :] > 0.0).cpu().numpy()
            for j, id2 in enumerate(ids):
                if id1 >= id2:
                    continue
                mask2 = (mask_logits[j, 0, :, :] > 0.0).cpu().numpy()
                miou = mask_iou(mask1, mask2)
                if miou > self.th_nms_miou:
                    age_id1 = min(list(
                        self.inference_state['output_dict_per_obj'][
                            self.inference_state['obj_id_to_idx'][id1]
                        ]['cond_frame_outputs'].keys()
                    ))
                    age_id2 = min(list(
                        self.inference_state['output_dict_per_obj'][
                            self.inference_state['obj_id_to_idx'][id2]
                        ]['cond_frame_outputs'].keys()
                    ))
                    remove_nms_ids.append(id1 if age_id1 >= age_id2 else id2)

        return remove_nms_ids

    def get_current_tracks(self) -> Dict:
        """
        Get current tracking state

        Returns:
            Dictionary with track_id: label pairs
        """
        return self.track_labels.copy()

    def get_adaptive_thresholds(self) -> Dict:
        """
        Get current adaptive thresholds

        Returns:
            Dictionary with threshold values
        """
        return {
            'th_det': self.adaptive_th_det,
            'th_high_conf': self.adaptive_th_high_conf,
            'score_history_size': len(self.detection_score_history)
        }

    def reset(self):
        """Reset tracker state"""
        self.lost_count = {}
        self.track_labels = {}
        self.known_ids = set()
        self.score_queue = deque(maxlen=self.n_frames)
        self.detection_score_history = deque(maxlen=100)
        self.frame_idx = -1
        self.is_initialized = False
        if self.inference_state is not None:
            self.inference_state.clear()

        # Clean up temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Destructor"""
        self.reset()
