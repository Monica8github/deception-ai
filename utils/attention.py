import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import config

class ChannelWiseAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = None):
        super().__init__()
        embed_dim = embed_dim or config.EMBED_DIM
        self.score_v = nn.Linear(embed_dim, 1)
        self.score_a = nn.Linear(embed_dim, 1)
        self.score_t = nn.Linear(embed_dim, 1)

    def forward(self, V, A, T):
        scores = torch.cat([self.score_v(V), self.score_a(A), self.score_t(T)], dim=1)
        alpha = F.softmax(scores, dim=1)
        fused = alpha[:,0:1]*V + alpha[:,1:2]*A + alpha[:,2:3]*T
        weights = {'video': alpha[:,0:1], 'audio': alpha[:,1:2], 'text': alpha[:,2:3]}
        return fused, weights
