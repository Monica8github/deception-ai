from dataset.simple_train import SimpleDeceptionModel
import torch
import numpy as np
import os
import config

class DeceptionPredictor:
    '''
    Loads the trained SimpleDeceptionModel and runs predictions.
    Used by dashboard.py instead of the heavy MultimodalDeceptionDetector.
    '''
    def __init__(self, checkpoint_path='checkpoints/best_model_v2.pt'):
        self.model = SimpleDeceptionModel()
        self.device = config.DEVICE
        
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f'✅ Loaded trained model from {checkpoint_path}')
            self.trained = True
        else:
            print('⚠️ No checkpoint - using random weights (demo mode)')
            self.trained = False
            
        self.model.to(self.device)
        self.model.eval()

    def predict_from_features(self, video_feat, audio_feat, text_feat):
        '''
        Args:
            video_feat: np.ndarray (T, 6) or (6,)
            audio_feat: np.ndarray (T, 45) or (45,)
            text_feat:  np.ndarray (256,)
        Returns result dict compatible with dashboard.py
        '''
        # Mean pool if temporal
        if video_feat.ndim > 1:
            video_feat = video_feat.mean(axis=0)
        if audio_feat.ndim > 1:
            audio_feat = audio_feat.mean(axis=0)

        # Normalize
        def norm(arr):
            std = arr.std()
            return (arr - arr.mean()) / (std + 1e-8) if std > 0 else arr

        v = torch.FloatTensor(norm(video_feat)).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(norm(audio_feat)).unsqueeze(0).to(self.device)
        t = torch.FloatTensor(norm(text_feat)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, weights = self.model(v, a, t)
            probs = torch.softmax(logits, dim=1)[0]
            pred = logits.argmax(1).item()

        lie_prob = probs[1].item()
        truth_prob = probs[0].item()
        confidence = max(lie_prob, truth_prob)

        return {
            'verdict': 'LIE' if pred == 1 else 'TRUTH',
            'confidence': confidence,
            'lie_probability': lie_prob,
            'truth_probability': truth_prob,
            'modality_weights': {
                'video': weights['video'],
                'audio': weights['audio'],
                'text': weights['text'],
            },
        }

    def predict_from_video(self, video_path: str) -> dict:
        '''
        Full pipeline: video file → prediction.
        Extracts features then runs predict_from_features.
        '''
        import sys
        sys.path.insert(0, '.')
        
        print('[1/4] Extracting video features...')
        try:
            from extractors.video_extractor import VideoExtractor
            ve = VideoExtractor()
            video_feat, face_crops = ve.extract(video_path)
        except Exception as e:
            print(f'  Video extraction failed: {e}')
            video_feat = np.zeros((1, 6), dtype=np.float32)
            face_crops = []

        print('[2/4] Extracting audio features...')
        try:
            from extractors.audio_extractor import AudioExtractor
            ae = AudioExtractor()
            audio_feat, stress_score = ae.extract(video_path)
        except Exception as e:
            print(f'  Audio extraction failed: {e}')
            audio_feat = np.zeros((1, 45), dtype=np.float32)
            stress_score = np.zeros(10)

        print('[3/4] Transcribing and encoding text...')
        try:
            from extractors.text_extractor import TextExtractor
            te = TextExtractor()
            text_result = te.extract(video_path)
            text_feat = text_result['embedding']
            transcript = text_result['transcript']
            top_suspicious = text_result['top_suspicious']
            tokens = text_result['tokens']
            attn_weights = text_result['attention_weights']
        except Exception as e:
            print(f'  Text extraction failed: {e}')
            text_feat = np.zeros(config.EMBED_DIM, dtype=np.float32)
            transcript = ''
            top_suspicious = []
            tokens = []
            attn_weights = np.zeros(1)

        print('[4/4] Running prediction...')
        result = self.predict_from_features(video_feat, audio_feat, text_feat)

        # Add temporal analysis
        temporal_scores = self._temporal_analysis(
            video_feat, audio_feat, text_feat
        )

        # Add stress index (0-100)
        stress_index = int(min(100, stress_score.mean() * 200))

        result.update({
            'temporal_scores': temporal_scores,
            'stress_score': stress_score,
            'video_features': video_feat,
            'face_crops': face_crops,
            'transcript': transcript,
            'top_suspicious': top_suspicious,
            'tokens': tokens,
            'attention_weights': attn_weights,
            'stress_index': stress_index,
        })
        return result

    def _temporal_analysis(self, video_feat, audio_feat,
                           text_feat, n_segments=10):
        '''Split features into segments, predict each segment lie prob.'''
        scores = []
        T = max(video_feat.shape[0] if video_feat.ndim > 1 else 1, n_segments)
        seg_size = max(1, T // n_segments)

        for i in range(n_segments):
            try:
                start = i * seg_size
                end = start + seg_size

                v_seg = video_feat[start:end] if video_feat.ndim > 1 else video_feat
                a_seg = audio_feat[start:end] if audio_feat.ndim > 1 else audio_feat

                v = v_seg.mean(axis=0) if v_seg.ndim > 1 else v_seg
                a = a_seg.mean(axis=0) if a_seg.ndim > 1 else a_seg

                res = self.predict_from_features(v, a, text_feat)
                scores.append(res['lie_probability'])
            except Exception:
                scores.append(0.5)
        return scores
