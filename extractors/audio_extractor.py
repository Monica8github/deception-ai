import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

import config


class AudioExtractor:
    """Extracts frame-level acoustic stress features from a video's audio track."""

    FRAME_MS = 25
    HOP_MS = config.AUDIO_HOP_MS

    def __init__(self) -> None:
        self.sr = 16000
        self.frame_len = int(self.sr * self.FRAME_MS / 1000)
        self.hop_len = int(self.sr * self.HOP_MS / 1000)

    def extract(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract audio features and stress score from a video file."""
        wav_path = self._extract_audio_from_video(video_path)

        try:
            y, sr = librosa.load(wav_path, sr=self.sr, mono=True)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        if y.size == 0:
            return np.zeros((1, config.AUDIO_FEAT_DIM), dtype=np.float32), np.zeros((1,), dtype=np.float32)

        mfcc = self._extract_mfcc(y)
        pitch, voiced = self._extract_pitch(y)
        energy = self._extract_energy(y)
        jitter = self._extract_jitter(y)
        shimmer = self._extract_shimmer(y)
        zcr = self._extract_zcr(y)
        centroid = self._extract_centroid(y)
        bandwidth = self._extract_bandwidth(y)
        rolloff = self._extract_rolloff(y)
        chroma = self._extract_chroma(y)
        mel_mean = self._extract_mel_mean(y)
        vad = self._extract_vad(y)

        pitch, voiced, energy, jitter, shimmer, zcr, centroid, bandwidth, rolloff, chroma, mel_mean, vad = self._align_lengths(
            pitch, voiced, energy, jitter, shimmer, zcr, centroid, bandwidth, rolloff, chroma, mel_mean, vad
        )

        f0_grad = np.gradient(pitch)
        e_grad = np.gradient(energy)
        stress_score = np.abs(f0_grad) + np.abs(e_grad)
        stress_score = np.where(voiced, stress_score, 0.0)

        acoustic_feats = np.stack([pitch, energy, jitter, shimmer], axis=1)
        spectral_feats = np.stack([zcr, centroid, bandwidth, rolloff], axis=1)
        features = np.concatenate([mfcc, acoustic_feats, spectral_feats, chroma, mel_mean, vad[:, None]], axis=1)
        features = self._zscore_normalize(features)

        if features.shape[0] == 0:
            return np.zeros((1, config.AUDIO_FEAT_DIM), dtype=np.float32), np.zeros((1,), dtype=np.float32)

        return features.astype(np.float32), stress_score.astype(np.float32)

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video into a temporary WAV file using ffmpeg."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f'Video path not found: {video_path}')

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_path = temp_wav.name
        except OSError as exc:
            raise RuntimeError('Could not create temporary audio file.') from exc

        command = [
            'ffmpeg',
            '-i',
            str(video_path),
            '-ac',
            '1',
            '-ar',
            str(self.sr),
            '-vn',
            wav_path,
            '-y',
            '-loglevel',
            'quiet',
        ]

        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise RuntimeError(
                    'ffmpeg is required but not found on PATH. Install ffmpeg and retry.'
                ) from exc
            raise RuntimeError('ffmpeg failed to extract audio from video.') from exc

        return wav_path

    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Compute 40 MFCCs for the audio signal."""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=40,
            n_fft=self.frame_len,
            hop_length=self.hop_len,
        )
        return mfcc.T

    def _extract_pitch(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pitch and voiced frames using librosa.pyin."""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                frame_length=self.frame_len,
                hop_length=self.hop_len,
            )
        except Exception:
            n_frames = 1 + (len(y) - self.frame_len) // self.hop_len
            return np.zeros((n_frames,), dtype=np.float32), np.zeros((n_frames,), dtype=bool)

        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        voiced = np.asarray(voiced_flag, dtype=bool)
        return f0, voiced

    def _extract_energy(self, y: np.ndarray) -> np.ndarray:
        """Compute RMS energy per frame."""
        rms = librosa.feature.rms(y=y, frame_length=self.frame_len, hop_length=self.hop_len)
        return rms.flatten().astype(np.float32)

    def _extract_jitter(self, y: np.ndarray) -> np.ndarray:
        """Approximate jitter from pitch period variation."""
        pitch, _ = self._extract_pitch(y)
        period = np.where(pitch > 0.0, 1.0 / (pitch + 1e-8), 0.0)
        diff = np.abs(np.diff(period, prepend=period[0]))
        denom = period + 1e-8
        return (diff / denom).astype(np.float32)

    def _extract_shimmer(self, y: np.ndarray) -> np.ndarray:
        """Approximate shimmer from frame-to-frame RMS energy variation."""
        energy = self._extract_energy(y)
        diff = np.abs(np.diff(energy, prepend=energy[0]))
        denom = energy + 1e-8
        return (diff / denom).astype(np.float32)

    def _extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Compute zero crossing rate per frame."""
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_len, hop_length=self.hop_len)
        return zcr.flatten().astype(np.float32)

    def _extract_centroid(self, y: np.ndarray) -> np.ndarray:
        """Compute spectral centroid per frame."""
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.frame_len, hop_length=self.hop_len)
        return centroid.flatten().astype(np.float32)

    def _extract_bandwidth(self, y: np.ndarray) -> np.ndarray:
        """Compute spectral bandwidth per frame."""
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.frame_len, hop_length=self.hop_len)
        return bandwidth.flatten().astype(np.float32)

    def _extract_rolloff(self, y: np.ndarray) -> np.ndarray:
        """Compute spectral rolloff per frame."""
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=self.frame_len, hop_length=self.hop_len)
        return rolloff.flatten().astype(np.float32)

    def _extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Compute 12-bin chroma features per frame."""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.frame_len, hop_length=self.hop_len)
        return chroma.T.astype(np.float32)

    def _extract_mel_mean(self, y: np.ndarray) -> np.ndarray:
        """Compute mean mel spectrogram values across 10 bands."""
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.frame_len, hop_length=self.hop_len, n_mels=10)
        return np.mean(mel, axis=0).T.astype(np.float32)

    def _extract_vad(self, y: np.ndarray) -> np.ndarray:
        """Estimate voice activity detection flags per frame."""
        energy = self._extract_energy(y)
        threshold = np.maximum(np.mean(energy) * 0.5, 1e-6)
        vad = (energy > threshold).astype(np.float32)
        return vad

    def _align_lengths(self, *arrays: np.ndarray) -> List[np.ndarray]:
        """Trim all arrays to the shortest length to align time frames."""
        lengths = [arr.shape[0] for arr in arrays]
        if len(lengths) == 0:
            return []
        min_len = min(lengths)
        return [arr[:min_len] for arr in arrays]

    def _zscore_normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature columns to zero mean and unit variance."""
        normalized = features.copy().astype(np.float32)
        for idx in range(normalized.shape[1]):
            col = normalized[:, idx]
            if np.allclose(col, 0.0):
                continue
            mean = np.mean(col)
            std = np.std(col)
            if std < 1e-6:
                continue
            normalized[:, idx] = (col - mean) / std
        return normalized


if __name__ == '__main__':
    extractor = AudioExtractor()
    features, stress = extractor.extract('sample.mp4')
    print(f'Feature shape: {features.shape}')
    print(f'Stress score shape: {stress.shape}')
    print(f'Peak stress frame: {int(np.argmax(stress))}')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(stress)
    plt.title('Vocal Stress Score Sa(t)')
    plt.xlabel('Frame')
    plt.ylabel('Stress')
    plt.savefig('stress_plot.png')
    print('Stress plot saved.')
