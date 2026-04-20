import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import DistilBertModel, DistilBertTokenizer

import config

try:
    # Use openai-whisper package
    import whisper
except (ImportError, TypeError) as exc:
    # If import fails due to conflict, try alternative import
    try:
        import openai_whisper as whisper
    except ImportError:
        whisper = None  # type: ignore
    _whisper_import_error = exc


class TextExtractor:
    """Transcribes video audio and encodes transcript with DistilBERT."""

    MAX_TOKENS = 512
    CHUNK_WORDS = 80
    OVERLAP = 10

    def __init__(self) -> None:
        if whisper is None:
            raise ImportError(
                'whisper is required for TextExtractor. Install it with `pip install whisper-openai`.'
            )

        print('Loading Whisper model, this may take a moment...')
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.DISTILBERT_MODEL)
        self.bert = DistilBertModel.from_pretrained(config.DISTILBERT_MODEL)
        self.bert.eval()
        self.bert.to(config.DEVICE)
        self.projector = torch.nn.Linear(768, config.EMBED_DIM).to(config.DEVICE)

    def extract(self, video_path: str) -> Dict:
        """Extract transcript embeddings and attention weights from a video file."""
        wav_path = self._extract_audio(video_path)
        try:
            result = self.whisper_model.transcribe(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        transcript = result.get('text', '').strip()
        if not transcript:
            return self._empty_result()

        chunks = self._chunk_text(transcript)
        chunk_embeddings: List[np.ndarray] = []
        token_embeddings: List[np.ndarray] = []
        attention_weights: List[np.ndarray] = []
        tokens_list: List[str] = []

        for idx, chunk in enumerate(chunks):
            cls_emb, token_embs, attn, tokens = self._encode_chunk(chunk)
            if idx > 0 and self.OVERLAP > 0:
                tokens = tokens[self.OVERLAP:]
                token_embs = token_embs[self.OVERLAP:, :]
                attn = attn[self.OVERLAP:]
            chunk_embeddings.append(cls_emb)
            token_embeddings.append(token_embs)
            attention_weights.append(attn)
            tokens_list.extend(tokens)

        if len(chunk_embeddings) == 0:
            return self._empty_result()

        cls_stack = np.stack(chunk_embeddings, axis=0)
        mean_cls = np.mean(cls_stack, axis=0)
        embedding = self._project_cls(mean_cls)
        token_embeddings_concat = np.concatenate(token_embeddings, axis=0) if token_embeddings else np.zeros((1, 768), dtype=np.float32)
        attention_weights_concat = np.concatenate(attention_weights, axis=0) if attention_weights else np.zeros((1,), dtype=np.float32)

        top_suspicious = self._compute_top_suspicious(tokens_list, attention_weights_concat)

        return {
            'embedding': embedding,
            'token_embeddings': token_embeddings_concat,
            'attention_weights': attention_weights_concat,
            'transcript': transcript,
            'tokens': tokens_list,
            'top_suspicious': top_suspicious,
        }

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio track from video using ffmpeg into a temporary WAV file."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f'Video path not found: {video_path}')

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_path = temp_wav.name
        except OSError as exc:
            raise RuntimeError('Could not create temporary WAV file for transcription.') from exc

        command = [
            'ffmpeg',
            '-i',
            str(video_path),
            '-ac',
            '1',
            '-ar',
            '16000',
            '-vn',
            wav_path,
            '-y',
            '-loglevel',
            'quiet',
        ]

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError('ffmpeg is required but not found on PATH. Install ffmpeg and retry.') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('ffmpeg failed to extract audio from the video.') from exc

        return wav_path

    def _chunk_text(self, transcript: str) -> List[str]:
        """Split transcript into overlapping word chunks for DistilBERT encoding."""
        words = transcript.split()
        if len(words) <= self.CHUNK_WORDS:
            return [transcript]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.CHUNK_WORDS, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            if end == len(words):
                break
            start = end - self.OVERLAP
        return chunks

    def _encode_chunk(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Encode one text chunk with DistilBERT and compute attention weights."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.MAX_TOKENS,
            padding=True,
        ).to(config.DEVICE)

        with torch.no_grad():
            outputs = self.bert(**inputs, output_attentions=True)

        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()
        cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().squeeze(0).numpy()
        attentions = outputs.attentions[-1][0].detach().cpu().numpy()
        attn_mean_heads = attentions.mean(axis=0)
        attn_weights = attn_mean_heads.mean(axis=0)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return cls_emb, hidden, attn_weights, tokens

    def _project_cls(self, cls_emb: np.ndarray) -> np.ndarray:
        """Project the 768-dim CLS embedding into the configured embedding dimension."""
        tensor = torch.from_numpy(cls_emb).to(config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            projected = self.projector(tensor)
        return projected.squeeze(0).cpu().numpy()

    def _compute_top_suspicious(self, tokens: List[str], attn: np.ndarray) -> List[Tuple[str, float]]:
        """Compute top-k suspicious tokens based on attention weights."""
        filtered: List[Tuple[str, float]] = []
        for token, score in zip(tokens, attn.tolist()):
            if token in ('[CLS]', '[SEP]'):
                continue
            if all(ch in '.,!?;:()[]{}"\'' for ch in token):
                continue
            filtered.append((token, float(score)))

        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered[:5]

    def _empty_result(self) -> Dict:
        """Return a zeroed-out result dictionary for empty transcript cases."""
        return {
            'embedding': np.zeros((config.EMBED_DIM,), dtype=np.float32),
            'token_embeddings': np.zeros((1, 768), dtype=np.float32),
            'attention_weights': np.zeros((1,), dtype=np.float32),
            'transcript': '',
            'tokens': [],
            'top_suspicious': [],
        }


if __name__ == '__main__':
    extractor = TextExtractor()
    result = extractor.extract('sample.mp4')
    print(f"Transcript: {result['transcript']}")
    print(f"Embedding shape: {result['embedding'].shape}")
    print(f"Token count: {len(result['tokens'])}")
    print('Top suspicious words:')
    for word, score in result['top_suspicious']:
        print(f"  '{word}' — attention score: {score:.4f}")
