import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from transformers import DistilBertModel, DistilBertTokenizer
import config


class DistilBertEncoder(nn.Module):
    """
    Wraps DistilBERT and projects CLS embedding to config.EMBED_DIM.
    Accepts either raw text strings OR pre-computed dicts from TextExtractor.
    """

    def __init__(self):
        """
        Initialize DistilBERT encoder with frozen weights and trainable projection head.
        """
        super().__init__()
        
        # Load pretrained model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.DISTILBERT_MODEL)
        self.bert = DistilBertModel.from_pretrained(config.DISTILBERT_MODEL)
        self.bert.eval()
        
        # Freeze ALL DistilBERT weights (only train the projection head)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Projection head (trainable)
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(512, config.EMBED_DIM)
        )
        
        # Move everything to config.DEVICE
        self.to(config.DEVICE)

    def forward(self, text_input) -> torch.Tensor:
        """
        Forward pass accepting multiple input types.
        
        Args:
            text_input: Can be one of:
                - str: raw transcript string
                - dict: output dict from TextExtractor (has 'embedding' or 'token_embeddings')
                - torch.Tensor: shape (B, 768) — pre-computed CLS embeddings
        
        Returns:
            torch.Tensor of shape (B, config.EMBED_DIM) on config.DEVICE
        """
        # Handle string input
        if isinstance(text_input, str):
            if not text_input or text_input.isspace():
                return torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
            
            # Tokenize
            tokens = self.tokenizer(
                text_input,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            tokens = {k: v.to(config.DEVICE) for k, v in tokens.items()}
            
            # Forward through BERT with no_grad
            with torch.no_grad():
                outputs = self.bert(**tokens)
            
            # Extract CLS embedding: (1, 768)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Project to EMBED_DIM
            embedding = self.projector(cls_embedding)
            return embedding
        
        # Handle dict input (from TextExtractor)
        elif isinstance(text_input, dict):
            # If already projected embedding exists
            if 'embedding' in text_input:
                emb = text_input['embedding']
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).float().to(config.DEVICE)
                if emb.shape[-1] == config.EMBED_DIM:
                    return emb.unsqueeze(0) if emb.dim() == 1 else emb
            
            # If token embeddings exist, mean-pool and project
            if 'token_embeddings' in text_input:
                token_embs = text_input['token_embeddings']
                if isinstance(token_embs, np.ndarray):
                    token_embs = torch.from_numpy(token_embs).float().to(config.DEVICE)
                # Mean-pool: (seq_len, 768) → (1, 768)
                cls_embedding = token_embs.mean(dim=0, keepdim=True)
                embedding = self.projector(cls_embedding)
                return embedding
            
            # Fallback: return zeros
            return torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)
        
        # Handle tensor input (B, 768)
        elif isinstance(text_input, torch.Tensor):
            text_input = text_input.to(config.DEVICE)
            embedding = self.projector(text_input)
            return embedding
        
        # Default: return zeros
        else:
            return torch.zeros(1, config.EMBED_DIM, device=config.DEVICE)

    def encode_text(self, text: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Encode raw text and extract per-token attention weights.
        
        Args:
            text: raw transcript string
        
        Returns:
            embedding: torch.Tensor (1, config.EMBED_DIM) for fusion
            attn_weights: np.ndarray (seq_len,) per-token importance for explainability
        """
        if not text or text.isspace():
            return torch.zeros(1, config.EMBED_DIM, device=config.DEVICE), np.zeros(1)
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        tokens = {k: v.to(config.DEVICE) for k, v in tokens.items()}
        
        # Forward through BERT with attention output
        with torch.no_grad():
            outputs = self.bert(**tokens, output_attentions=True)
        
        # Extract CLS embedding and project
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embedding = self.projector(cls_embedding)
        
        # Extract attention: outputs.attentions[-1] → (B, num_heads, seq_len, seq_len)
        # Take last layer, first batch: (num_heads, seq_len, seq_len)
        attn = outputs.attentions[-1][0].cpu().numpy()
        
        # Mean over heads: (seq_len, seq_len)
        attn = attn.mean(axis=0)
        
        # Mean over query dimension (axis 0) → (seq_len,)
        attn_weights = attn.mean(axis=0)
        
        return embedding, attn_weights

    def encode_from_extractor(self, extractor_dict: Dict) -> torch.Tensor:
        """
        Clean interface for fusion_model.py to call.
        
        Args:
            extractor_dict: dict output from TextExtractor
        
        Returns:
            torch.Tensor of shape (1, config.EMBED_DIM) on config.DEVICE
        """
        embedding = self.forward(extractor_dict)
        return embedding.to(config.DEVICE)

    def get_suspicious_tokens(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Tokenize text, extract attention, and return top-k most attended tokens.
        
        Args:
            text: raw transcript string
            top_k: number of top tokens to return
        
        Returns:
            List of (token_str, attention_score) tuples for top_k tokens
        """
        if not text or text.isspace():
            return []
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        tokens = {k: v.to(config.DEVICE) for k, v in tokens.items()}
        
        # Forward through BERT with attention
        with torch.no_grad():
            outputs = self.bert(**tokens, output_attentions=True)
        
        # Extract attention from last layer
        attn = outputs.attentions[-1][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
        attn = attn.mean(axis=0)  # (seq_len, seq_len)
        attn_weights = attn.mean(axis=0)  # (seq_len,)
        
        # Get token IDs and strings
        token_ids = tokens['input_ids'][0].cpu().tolist()
        token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Filter out special tokens and punctuation
        filtered_tokens = []
        for idx, (tok_str, score) in enumerate(zip(token_strings, attn_weights)):
            # Skip special tokens
            if tok_str in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                continue
            # Skip pure punctuation (optional, but helps with explainability)
            if all(c in '.,!?;:\'"()[]{}' for c in tok_str.replace('##', '')):
                continue
            filtered_tokens.append((tok_str, float(score)))
        
        # Sort by attention score and return top_k
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        return filtered_tokens[:top_k]


if __name__ == "__main__":
    encoder = DistilBertEncoder().to(config.DEVICE)
    encoder.eval()
    
    # Test 1: raw string
    emb = encoder.forward("I did not take the money from the office yesterday.")
    print(f"String input → shape: {emb.shape}")   # (1, 256)
    
    # Test 2: encode_text with attention
    emb2, attn = encoder.encode_text("She was nervous and kept avoiding eye contact.")
    print(f"encode_text embedding: {emb2.shape}")  # (1, 256)
    print(f"Attention weights: {attn.shape}")       # (seq_len,)
    
    # Test 3: suspicious tokens
    tokens = encoder.get_suspicious_tokens(
        "I never said I was there on that particular evening.",
        top_k=5
    )
    print("Top suspicious tokens:")
    for tok, score in tokens:
        print(f"  '{tok}' → {score:.4f}")
    
    # Test 4: trainable params (only projector should be trainable)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")   # only projector ~400K
    print(f"Frozen params    : {frozen:,}")      # DistilBERT ~66M
