import torch

"""Configuration settings for the deception detection system."""

FRAME_RATE = 10
AUDIO_HOP_MS = 10
MFCC_BINS = 40
VIDEO_FEAT_DIM = 12
AUDIO_FEAT_DIM = 71
TEXT_EMBED_DIM = 256
EMBED_DIM = 256
DISTILBERT_MODEL = 'distilbert-base-uncased'
WHISPER_MODEL = 'base'
CAPSULE_ROUTING_ITERS = 3
DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
