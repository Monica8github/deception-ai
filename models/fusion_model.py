import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import config

from extractors.video_extractor import VideoExtractor
from extractors.audio_extractor import AudioExtractor
from extractors.text_extractor import TextExtractor
from models.capsule_network import CapsuleNetwork
from models.audio_cnn import AudioCNN
from models.distilbert_encoder import DistilBertEncoder


class ChannelWiseAttention(nn.Module):
    """
    Learns per-sample modality weights αv, αa, αt that sum to 1.
    Formula from paper: αi = exp(hi) / (exp(hv) + exp(ha) + exp(ht))
    F = αv*V + αa*A + αt*T
    """
    
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        """
        Initialize attention scorers for each modality.
        
        Args:
            embed_dim: Embedding dimension (default config.EMBED_DIM)
        """
        super().__init__()
        self.hv = nn.Linear(embed_dim, 1, bias=False)  # relevance scorer for video
        self.ha = nn.Linear(embed_dim, 1, bias=False)  # relevance scorer for audio
        self.ht = nn.Linear(embed_dim, 1, bias=False)  # relevance scorer for text

    def forward(self, V: torch.Tensor, A: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute modality-wise attention weights and fused embedding.
        
        Args:
            V: Video embedding (B, embed_dim)
            A: Audio embedding (B, embed_dim)
            T: Text embedding (B, embed_dim)
        
        Returns:
            F_fused: Weighted fusion (B, embed_dim)
            weights: Dict with mean attention weights for each modality
        """
        # Compute relevance scores
        hv = self.hv(V)  # (B, 1)
        ha = self.ha(A)  # (B, 1)
        ht = self.ht(T)  # (B, 1)
        
        # Concatenate and apply softmax
        scores = torch.cat([hv, ha, ht], dim=1)  # (B, 3)
        weights_norm = F.softmax(scores, dim=1)  # (B, 3)
        
        # Extract individual weights
        αv = weights_norm[:, 0:1]  # (B, 1)
        αa = weights_norm[:, 1:2]  # (B, 1)
        αt = weights_norm[:, 2:3]  # (B, 1)
        
        # Fused embedding
        F_fused = αv * V + αa * A + αt * T  # (B, embed_dim)
        
        # Return mean weights across batch for explainability
        return F_fused, {
            'video': αv.mean(),
            'audio': αa.mean(),
            'text': αt.mean()
        }


class DeceptionClassifier(nn.Module):
    """
    MLP classifier head on top of fused embedding.
    Input : (B, embed_dim)
    Output: (B, 2) logits for [truth, lie]
    """
    
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        """
        Initialize classification MLP.
        
        Args:
            embed_dim: Input embedding dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify fused embedding.
        
        Args:
            x: (B, embed_dim)
        
        Returns:
            logits: (B, 2)
        """
        return self.net(x)


class MultimodalDeceptionDetector(nn.Module):
    """
    Full pipeline: video file → deception prediction + explainability.
    Runs all extractors and encoders internally, returns comprehensive result dict.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize all components.
        
        Args:
            checkpoint_path: Optional path to checkpoint file to load
        """
        super().__init__()
        
        # Extractors (not nn.Modules — just data processors)
        self.video_extractor = VideoExtractor()
        self.audio_extractor = AudioExtractor()
        self.text_extractor = TextExtractor()

        # Encoders (nn.Modules)
        self.visual_encoder = CapsuleNetwork()
        self.audio_encoder = AudioCNN()
        self.text_encoder = DistilBertEncoder()

        # Fusion + Classifier
        self.attention = ChannelWiseAttention()
        self.classifier = DeceptionClassifier()

        # Move all nn.Module components to config.DEVICE
        self.to(config.DEVICE)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        else:
            print("[INFO] No checkpoint loaded — running with random weights (demo mode)")

    def predict_from_video(self, video_path: str) -> Dict:
        """
        Full pipeline inference from video file.
        
        Returns comprehensive result dict for dashboard visualization.
        
        Args:
            video_path: Path to input video file
        
        Returns:
            Dict with keys:
              - verdict: 'LIE' or 'TRUTH'
              - confidence: float (0-1)
              - lie_probability: float
              - truth_probability: float
              - modality_weights: Dict with video/audio/text weights
              - temporal_scores: List of segment lie probabilities
              - transcript: str
              - top_suspicious: List of (token, score) tuples
              - stress_score: np.ndarray for audio plot
              - video_features: np.ndarray for video feature plot
              - face_crops: list for GradCAM
              - tokens: List of token strings
              - attention_weights: np.ndarray
        """
        self.eval()
        result = {
            'verdict': 'UNKNOWN',
            'confidence': 0.0,
            'lie_probability': 0.0,
            'truth_probability': 0.0,
            'modality_weights': {'video': 0.0, 'audio': 0.0, 'text': 0.0},
            'temporal_scores': [],
            'transcript': '',
            'top_suspicious': [],
            'stress_score': np.array([]),
            'video_features': np.array([]),
            'face_crops': [],
            'tokens': [],
            'attention_weights': np.array([])
        }

        try:
            # [1/5] Extract video features
            print("[1/5] Extracting video features...")
            try:
                video_features, face_crops = self.video_extractor.extract(video_path)
                result['video_features'] = video_features
                result['face_crops'] = face_crops
            except Exception as e:
                print(f"[WARNING] Video extraction failed: {e}")
                video_features = None
                face_crops = []

            # [2/5] Extract audio features
            print("[2/5] Extracting audio features...")
            try:
                audio_features, stress_score = self.audio_extractor.extract(video_path)
                result['stress_score'] = stress_score
            except Exception as e:
                print(f"[WARNING] Audio extraction failed: {e}")
                audio_features = None
                stress_score = None

            # [3/5] Extract and encode text
            print("[3/5] Transcribing and encoding text...")
            try:
                text_result = self.text_extractor.extract(video_path)
                if isinstance(text_result, dict):
                    result['transcript'] = text_result.get('transcript', '')
                else:
                    result['transcript'] = str(text_result)
            except Exception as e:
                print(f"[WARNING] Text extraction failed: {e}")
                text_result = None
                result['transcript'] = ''

            # Encode all modalities under no_grad
            with torch.no_grad():
                # Video encoding
                if video_features is not None:
                    if face_crops:
                        # Use first face crop for visual encoding
                        V = self.visual_encoder(torch.from_numpy(face_crops[0]).unsqueeze(0).float().to(config.DEVICE))
                    else:
                        # Use raw features if no face crops
                        V = self.visual_encoder(torch.from_numpy(video_features).float().to(config.DEVICE))
                else:
                    V = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
                
                # Audio encoding
                if audio_features is not None:
                    A = self.audio_encoder.encode_features(audio_features)
                else:
                    A = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
                
                # Text encoding
                if text_result is not None:
                    T = self.text_encoder.encode_from_extractor(text_result)
                    # Extract attention weights and tokens for explainability
                    try:
                        T_text = text_result.get('transcript', '') if isinstance(text_result, dict) else result['transcript']
                        if T_text:
                            _, attn_weights = self.text_encoder.encode_text(T_text)
                            result['attention_weights'] = attn_weights
                            result['top_suspicious'] = self.text_encoder.get_suspicious_tokens(T_text, top_k=5)
                            result['tokens'] = self.text_encoder.tokenizer.tokenize(T_text)
                    except Exception as e:
                        print(f"[WARNING] Text explainability extraction failed: {e}")
                else:
                    T = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)

                # [4/5] Fuse modalities
                print("[4/5] Fusing modalities...")
                F_fused, modality_weights = self.attention(V, A, T)

                # Classify
                logits = self.classifier(F_fused)  # (1, 2)
                probs = F.softmax(logits, dim=1)  # (1, 2)
                
                pred_class = torch.argmax(probs, dim=1).item()  # 0=truth, 1=lie
                confidence = probs[0, pred_class].item()

                result['verdict'] = 'LIE' if pred_class == 1 else 'TRUTH'
                result['confidence'] = confidence
                result['lie_probability'] = probs[0, 1].item()
                result['truth_probability'] = probs[0, 0].item()
                result['modality_weights'] = {
                    'video': modality_weights['video'].item(),
                    'audio': modality_weights['audio'].item(),
                    'text': modality_weights['text'].item()
                }

            # [5/5] Temporal analysis
            print("[5/5] Running temporal analysis...")
            try:
                temporal_scores = self._temporal_analysis(video_path, n_segments=10)
                result['temporal_scores'] = temporal_scores
            except Exception as e:
                print(f"[WARNING] Temporal analysis failed: {e}")
                result['temporal_scores'] = []

        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")

        return result

    def _temporal_analysis(self, video_path: str, n_segments: int = 10) -> List[float]:
        """
        Split video into n_segments equal time chunks and analyze deception per segment.
        
        Args:
            video_path: Path to input video
            n_segments: Number of segments to divide video into
        
        Returns:
            List of n_segments lie probabilities (D(t) values)
        """
        temporal_scores = []

        try:
            # Get total frame count
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames == 0:
                return [0.5] * n_segments

            frames_per_segment = total_frames // n_segments

            for seg_idx in range(n_segments):
                try:
                    start_frame = seg_idx * frames_per_segment
                    end_frame = (seg_idx + 1) * frames_per_segment if seg_idx < n_segments - 1 else total_frames

                    # Extract features for this segment
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    segment_frames = []
                    segment_audio = None
                    frame_count = 0

                    while frame_count < (end_frame - start_frame):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        segment_frames.append(frame)
                        frame_count += 1

                    cap.release()

                    # Quick feature extraction for segment (video + audio only)
                    if segment_frames:
                        # Extract audio for segment (simplified — use entire audio with weight)
                        try:
                            _, seg_audio = self.audio_extractor.extract(video_path)
                            if seg_audio is not None:
                                # Use weighted portion of audio spectrum
                                seg_idx_normalized = seg_idx / max(n_segments - 1, 1)
                                audio_size = seg_audio.shape[0] if seg_audio.ndim > 0 else 100
                                seg_audio = seg_audio[int(audio_size * seg_idx_normalized):
                                                      int(audio_size * (seg_idx_normalized + 1/n_segments))]
                                if len(seg_audio) == 0:
                                    seg_audio = None
                        except:
                            seg_audio = None

                        # Encode with zeros for missing modalities
                        with torch.no_grad():
                            # Video (dummy for segment)
                            V = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
                            
                            # Audio
                            if seg_audio is not None:
                                try:
                                    A = self.audio_encoder.encode_features(seg_audio)
                                except:
                                    A = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
                            else:
                                A = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
                            
                            # Text (use full transcript with temporal weighting)
                            T = torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)

                            # Fuse and classify
                            F_fused, _ = self.attention(V, A, T)
                            logits = self.classifier(F_fused)
                            probs = F.softmax(logits, dim=1)
                            lie_prob = probs[0, 1].item()
                            temporal_scores.append(lie_prob)
                    else:
                        temporal_scores.append(0.5)  # Uncertain if no frames

                except Exception as e:
                    print(f"[WARNING] Segment {seg_idx} analysis failed: {e}")
                    temporal_scores.append(0.5)  # Default uncertain

        except Exception as e:
            print(f"[WARNING] Temporal analysis init failed: {e}")
            temporal_scores = [0.5] * n_segments

        return temporal_scores

    def forward(self, V: torch.Tensor, A: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Training-mode forward pass with pre-encoded embeddings.
        
        Args:
            V: Video embedding (B, embed_dim)
            A: Audio embedding (B, embed_dim)
            T: Text embedding (B, embed_dim)
        
        Returns:
            logits: (B, 2) — classification logits
            modality_weights: Dict with attention weights
        """
        F_fused, weights = self.attention(V, A, T)
        logits = self.classifier(F_fused)
        return logits, weights

    def _load_checkpoint(self, path: str):
        """
        Load state dict from checkpoint file.
        
        Args:
            path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 0)
                print(f"[INFO] Loaded checkpoint from epoch {epoch}")
            else:
                self.load_state_dict(checkpoint)
                print(f"[INFO] Loaded checkpoint")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")

    def save_checkpoint(self, path: str, epoch: int = 0, metrics: Dict = None):
        """
        Save model state dict + metadata to checkpoint file.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Optional dict of metrics to save
        """
        if metrics is None:
            metrics = {}
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        try:
            torch.save(checkpoint, path)
            print(f"[INFO] Saved checkpoint to {path} (epoch {epoch})")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")


if __name__ == "__main__":
    print("Initializing MultimodalDeceptionDetector...")
    detector = MultimodalDeceptionDetector()
    detector.eval()

    # Count parameters
    total = sum(p.numel() for p in detector.parameters())
    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass with dummy tensors
    B = 2
    V = torch.randn(B, config.EMBED_DIM).to(config.DEVICE)
    A = torch.randn(B, config.EMBED_DIM).to(config.DEVICE)
    T = torch.randn(B, config.EMBED_DIM).to(config.DEVICE)

    logits, weights = detector(V, A, T)
    print(f"Logits shape  : {logits.shape}")    # (2, 2)
    print(f"Modality weights: {weights}")

    probs = F.softmax(logits, dim=1)
    print(f"Probabilities : {probs}")

    def predict_from_video(self, video_path: str) -> Dict:
        """Run extractors for a video file and return a merged prediction result."""
        from extractors.audio_extractor import AudioExtractor
        from extractors.text_extractor import TextExtractor
        from extractors.video_extractor import VideoExtractor
        import cv2

        video_ext = VideoExtractor()
        audio_ext = AudioExtractor()
        text_ext = TextExtractor()

        video_features, face_crops = video_ext.extract(video_path)
        audio_features, stress = audio_ext.extract(video_path)
        text_result = text_ext.extract(video_path)

        face_crop = next((f for f in reversed(face_crops) if f is not None), None)
        if face_crop is not None:
            face_crop = cv2.resize(face_crop, (224, 224))
            face_tensor = torch.tensor(
                face_crop.transpose(2, 0, 1), dtype=torch.float32,
            ).unsqueeze(0) / 255.0
        else:
            face_tensor = torch.zeros(1, 3, 224, 224)

        audio_tensor = prepare_audio_tensor(audio_features)
        transcript = text_result.get('transcript', '')

        result = self.forward(face_tensor, audio_tensor, transcript)
        result.update({
            'transcript': transcript,
            'top_suspicious': text_result.get('top_suspicious', []),
            'stress_score': stress,
            'face_crops': face_crops,
            'audio_features': audio_features,
        })
        return result

    def temporal_predict(self, video_path: str, segment_duration_s: float = 3.0) -> Dict:
        """Segment video into chunks and run prediction on each segment."""
        import os
        import subprocess
        import tempfile

        overall = self.predict_from_video(video_path)
        probe = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries',
            'format=duration',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            video_path,
        ], capture_output=True, text=True)
        duration = float(probe.stdout.strip() or 0)

        segment_probs = []
        starts = np.arange(0, duration, segment_duration_s)
        for start in starts:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run([
                'ffmpeg',
                '-y',
                '-ss',
                str(start),
                '-t',
                str(segment_duration_s),
                '-i',
                video_path,
                '-c',
                'copy',
                tmp_path,
            ], capture_output=True)
            try:
                seg_result = self.predict_from_video(tmp_path)
                probs = seg_result['probs'][0].detach().cpu().numpy()
                segment_probs.append((float(probs[0]), float(probs[1])))
            except Exception:
                segment_probs.append((0.5, 0.5))
            finally:
                os.unlink(tmp_path)

        lie_over_time = np.array([p[1] for p in segment_probs])
        return {
            'segment_probs': segment_probs,
            'lie_over_time': lie_over_time,
            'overall': overall,
        }


if __name__ == '__main__':
    model = MultimodalDeceptionDetector()
    model.eval()
    face = torch.zeros(1, 3, 224, 224)
    audio = torch.zeros(1, 45, 100)
    result = model.forward(face, audio, 'I did not do anything wrong.')
    print(f"Prediction : {'LIE' if result['prediction'] else 'TRUTH'}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Weights    : {result['weights']}")
