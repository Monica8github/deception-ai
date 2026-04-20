import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
    try:
        _MEDIAPIPE_AVAILABLE = hasattr(mp, 'solutions')
    except Exception:
        _MEDIAPIPE_AVAILABLE = False
except ImportError:
    mp = None
    _MEDIAPIPE_AVAILABLE = False

import config


class VideoExtractor:
    """Extracts frame-level facial behavioral features from a video file."""

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_BROW = [336, 296, 334, 293, 300]
    RIGHT_BROW = [107, 66, 105, 63, 70]
    LIPS = [13, 14, 78, 308, 82, 312]
    NOSE = [1, 98, 327]
    HEAD_POSE_INDICES = [1, 152, 263, 33, 287, 57]

    def __init__(self) -> None:
        self.face_mesh = None
        self.use_face_detection = False

        if not _MEDIAPIPE_AVAILABLE:
            print("[WARN] MediaPipe not available or misconfigured — using fallback video extraction")
            return

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.use_face_detection = True
        except Exception as e:
            print(f"[WARN] Failed to initialize MediaPipe FaceMesh: {e} — using fallback")

    def extract(self, video_path: str) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
        """Extracts features and face crops from a video file.

        Returns a feature matrix with shape (T, 12) and a list of face crops.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f'Video path not found: {video_path}')

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f'Unable to open video file: {video_path}')

        source_fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
        frame_skip = max(1, int(source_fps / config.FRAME_RATE))

        raw_frames: List[dict] = []
        face_crops: List[Optional[np.ndarray]] = []
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            feature_data, crop = self._process_frame(frame)
            raw_frames.append(feature_data)
            face_crops.append(crop)
            frame_index += 1

        cap.release()

        if len(raw_frames) == 0:
            return np.zeros((0, config.VIDEO_FEAT_DIM), dtype=np.float32), []

        feature_array = self._build_feature_matrix(raw_frames)
        normalized = self._normalize_features(feature_array)
        return normalized, face_crops

    def _build_feature_matrix(self, raw_frames: List[dict]) -> np.ndarray:
        ear_series = np.array([f['ear_avg'] for f in raw_frames], dtype=np.float32)
        lip_series = np.array([f['lip_separation'] for f in raw_frames], dtype=np.float32)

        blink_rates = self._compute_blink_rate(ear_series, config.FRAME_RATE)
        micro_intensities = self._sliding_variance(ear_series, 5)
        sync_scores = self._speech_face_sync(lip_series)

        features = []
        for idx, frame_data in enumerate(raw_frames):
            features.append([
                frame_data['ear_avg'],
                frame_data['brow_raise'],
                frame_data['lip_compression'],
                frame_data['head_pitch'],
                frame_data['head_yaw'],
                frame_data['head_roll'],
                frame_data['gaze_direction'],
                frame_data['nose_wrinkle'],
                frame_data['asymmetry'],
                blink_rates[idx],
                micro_intensities[idx],
                sync_scores[idx],
            ])

        return np.vstack(features).astype(np.float32)

    def _process_frame(self, frame: np.ndarray) -> Tuple[dict, Optional[np.ndarray]]:
        if not self.use_face_detection or self.face_mesh is None:
            return {
                'ear_avg': 0.0,
                'brow_raise': 0.0,
                'lip_compression': 0.0,
                'head_pitch': 0.0,
                'head_yaw': 0.0,
                'head_roll': 0.0,
                'gaze_direction': 0.0,
                'nose_wrinkle': 0.0,
                'asymmetry': 0.0,
                'lip_separation': 0.0,
            }, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception as e:
            print(f"[WARN] Face detection failed: {e}")
            return {
                'ear_avg': 0.0,
                'brow_raise': 0.0,
                'lip_compression': 0.0,
                'head_pitch': 0.0,
                'head_yaw': 0.0,
                'head_roll': 0.0,
                'gaze_direction': 0.0,
                'nose_wrinkle': 0.0,
                'asymmetry': 0.0,
                'lip_separation': 0.0,
            }, None

        height, width = frame.shape[:2]
        if not results.multi_face_landmarks:
            return {
                'ear_avg': 0.0,
                'brow_raise': 0.0,
                'lip_compression': 0.0,
                'head_pitch': 0.0,
                'head_yaw': 0.0,
                'head_roll': 0.0,
                'gaze_direction': 0.0,
                'nose_wrinkle': 0.0,
                'asymmetry': 0.0,
                'lip_separation': 0.0,
            }, None

        landmarks = results.multi_face_landmarks[0].landmark
        ear_left = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, width, height)
        ear_right = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, width, height)
        ear_avg = (ear_left + ear_right) / 2.0
        brow_raise = self._brow_raise(landmarks, self.LEFT_BROW, self.RIGHT_BROW)
        lip_compression = self._lip_compression(landmarks, self.LIPS)
        head_pitch, head_yaw, head_roll = self._head_pose(landmarks, (height, width))
        gaze_direction = self._gaze_direction(landmarks, width)
        nose_wrinkle = self._nose_wrinkle(landmarks)
        asymmetry = self._asymmetry(ear_left, ear_right, landmarks)
        lip_separation = self._lip_separation(landmarks)
        crop = self._crop_face(frame, landmarks, width, height)

        return {
            'ear_avg': float(ear_avg),
            'brow_raise': float(brow_raise),
            'lip_compression': float(lip_compression),
            'head_pitch': float(head_pitch),
            'head_yaw': float(head_yaw),
            'head_roll': float(head_roll),
            'gaze_direction': float(gaze_direction),
            'nose_wrinkle': float(nose_wrinkle),
            'asymmetry': float(asymmetry),
            'lip_separation': float(lip_separation),
        }, crop

    def _gaze_direction(self, landmarks, width: int) -> float:
        left_eye_x = np.mean([landmarks[idx].x for idx in self.LEFT_EYE]) * width
        right_eye_x = np.mean([landmarks[idx].x for idx in self.RIGHT_EYE]) * width
        eye_center_x = (left_eye_x + right_eye_x) / 2.0
        nose_x = landmarks[self.NOSE[0]].x * width
        direction = (eye_center_x - nose_x) / max(width * 0.25, 1.0)
        return float(np.clip(direction, -1.0, 1.0))

    def _nose_wrinkle(self, landmarks) -> float:
        left_nose = landmarks[self.NOSE[1]]
        right_nose = landmarks[self.NOSE[2]]
        nose_tip = landmarks[self.NOSE[0]]
        nose_width = np.linalg.norm(
            np.array([left_nose.x - right_nose.x, left_nose.y - right_nose.y])
        )
        upper_lip_y = landmarks[self.LIPS[0]].y
        wrinkle = abs(nose_tip.y - upper_lip_y) / max(nose_width, 1e-6)
        return float(np.clip(wrinkle, 0.0, 2.0))

    def _asymmetry(self, left_ear: float, right_ear: float, landmarks) -> float:
        left_brow_y = np.mean([landmarks[i].y for i in self.LEFT_BROW])
        right_brow_y = np.mean([landmarks[i].y for i in self.RIGHT_BROW])
        brow_diff = abs(left_brow_y - right_brow_y)
        ear_diff = abs(left_ear - right_ear)
        return float(ear_diff + brow_diff)

    def _lip_separation(self, landmarks) -> float:
        upper_y = landmarks[self.LIPS[0]].y
        lower_y = landmarks[self.LIPS[1]].y
        return float(abs(lower_y - upper_y))

    def _compute_blink_rate(self, ear_series: np.ndarray, fps: int) -> np.ndarray:
        threshold = 0.22
        blink_flags = np.zeros_like(ear_series, dtype=np.float32)
        for i in range(1, len(ear_series)):
            if ear_series[i - 1] > threshold and ear_series[i] <= threshold:
                blink_flags[i] = 1.0

        window_frames = max(1, int(fps * 3))
        rates = np.convolve(blink_flags, np.ones(window_frames, dtype=np.float32), 'same')
        return rates / max(3.0, window_frames / fps)

    def _sliding_variance(self, series: np.ndarray, window: int) -> np.ndarray:
        padded = np.pad(series, (window - 1, 0), mode='edge')
        variances = []
        for i in range(len(series)):
            window_values = padded[i:i + window]
            variances.append(float(np.var(window_values)))
        return np.array(variances, dtype=np.float32)

    def _speech_face_sync(self, lip_series: np.ndarray) -> np.ndarray:
        if len(lip_series) == 0:
            return np.zeros((0,), dtype=np.float32)
        deltas = np.abs(np.diff(lip_series, prepend=lip_series[0]))
        normalized = deltas / np.maximum(np.max(deltas), 1e-6)
        return normalized.astype(np.float32)

    def _eye_aspect_ratio(
        self,
        landmarks,
        eye_indices: List[int],
        frame_width: int,
        frame_height: int,
    ) -> float:
        points = [landmarks[idx] for idx in eye_indices]
        pts = np.array(
            [[p.x * frame_width, p.y * frame_height] for p in points],
            dtype=np.float32,
        )

        p1, p2, p3, p4, p5, p6 = pts
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return float((vertical1 + vertical2) / (2.0 * horizontal))

    def _brow_raise(
        self,
        landmarks,
        left_brow_idx: List[int],
        right_brow_idx: List[int],
    ) -> float:
        left_brow_y = np.mean([landmarks[i].y for i in left_brow_idx])
        right_brow_y = np.mean([landmarks[i].y for i in right_brow_idx])
        left_eye_y = np.mean([landmarks[i].y for i in self.LEFT_EYE[1:5]])
        right_eye_y = np.mean([landmarks[i].y for i in self.RIGHT_EYE[1:5]])
        left_raise = left_eye_y - left_brow_y
        right_raise = right_eye_y - right_brow_y
        return float((left_raise + right_raise) / 2.0)

    def _lip_compression(self, landmarks, lip_indices: List[int]) -> float:
        if len(lip_indices) < 2:
            return 0.0
        upper_y = landmarks[lip_indices[0]].y
        lower_y = landmarks[lip_indices[1]].y
        return float(abs(lower_y - upper_y))

    def _head_pose(
        self,
        landmarks,
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        height, width = frame_shape
        image_points = np.array(
            [
                [landmarks[idx].x * width, landmarks[idx].y * height]
                for idx in self.HEAD_POSE_INDICES
            ],
            dtype=np.float64,
        )

        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -63.6, -12.5],
                [-43.3, 32.7, -26.0],
                [43.3, 32.7, -26.0],
                [-28.9, -28.9, -24.1],
                [28.9, -28.9, -24.1],
            ],
            dtype=np.float64,
        )

        focal_length = width
        camera_matrix = np.array(
            [[focal_length, 0, width / 2.0],
             [0, focal_length, height / 2.0],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vector, _ = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return 0.0, 0.0, 0.0

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))
        except cv2.error:
            return 0.0, 0.0, 0.0

    def _crop_face(
        self,
        frame: np.ndarray,
        landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:
        xs = [int(min(max(lm.x * width, 0), width - 1)) for lm in landmarks]
        ys = [int(min(max(lm.y * height, 0), height - 1)) for lm in landmarks]
        x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, width - 1)
        y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, height - 1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(frame, (224, 224))
        return cv2.resize(crop, (224, 224))

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        mask = np.any(features != 0.0, axis=1)
        normalized = features.copy()
        if np.any(mask):
            valid = features[mask]
            mean = valid.mean(axis=0)
            std = valid.std(axis=0)
            std[std < 1e-6] = 1.0
            normalized[mask] = (valid - mean) / std
        return normalized


if __name__ == '__main__':
    extractor = VideoExtractor()
    features, crops = extractor.extract('sample.mp4')
    print(f'Feature shape: {features.shape}')
    print(f'Face crops: {len(crops)}')
    if features.shape[0] > 0:
        print(f'Feature sample (frame 0): {features[0]}')
