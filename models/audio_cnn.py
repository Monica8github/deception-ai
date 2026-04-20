import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class AudioCNN(nn.Module):
    """
    1D CNN audio encoder.
    Input : (B, 45, T) — batch of audio feature sequences
    Output: (B, config.EMBED_DIM) — fixed audio embedding
    """

    def __init__(self, input_dim: int = 45):
        """
        Build sequential layers for 1D CNN audio encoding.
        
        Args:
            input_dim: Number of audio features (default 45)
        """
        super().__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio CNN encoder.
        
        Args:
            x: (B, 45) or (B, T, 45) — audio feature vectors or batch of audio feature sequences
            
        Returns:
            (B, config.EMBED_DIM) — fixed audio embedding
        """
        # If 2D input, convert to 3D by adding time dimension
        if x.dim() == 2:  # (B, 45)
            x = x.unsqueeze(1)  # (B, 1, 45)
        
        # Now x should be (B, T, 45)
        # Transpose to (B, 45, T) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Head: AdaptiveAvgPool1d(1) → Flatten → Linear → ReLU → Dropout
        x = self.head(x)
        
        return x

    def encode_features(self, features) -> torch.Tensor:
        """
        Convenience method to encode raw audio features.
        
        Accepts raw numpy (T, 45) from AudioExtractor, converts to tensor,
        adds batch dimension, runs forward pass, and returns (1, EMBED_DIM) tensor.
        Moves tensor to config.DEVICE automatically.
        
        Args:
            features: numpy array of shape (T, 45)
            
        Returns:
            torch.Tensor of shape (1, config.EMBED_DIM)
        """
        import numpy as np
        
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Add batch dimension: (T, 45) → (1, T, 45)
        x = features.unsqueeze(0)
        
        # Move to device
        x = x.to(config.DEVICE)
        
        # Forward pass
        with torch.no_grad():
            embedding = self.forward(x)
        
        return embedding


if __name__ == "__main__":
    import numpy as np
    
    model = AudioCNN().to(config.DEVICE)
    model.eval()

    # Simulate AudioExtractor output
    dummy_features = np.random.randn(300, 45).astype(np.float32)
    embedding = model.encode_features(dummy_features)
    print(f"Embedding shape: {embedding.shape}")   # (1, 256)

    # Batch test
    x = torch.randn(4, 300, 45).to(config.DEVICE)
    out = model(x)
    print(f"Batch output: {out.shape}")            # (4, 256)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
