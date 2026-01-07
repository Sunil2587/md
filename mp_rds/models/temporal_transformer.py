"""
Temporal Transformer Model for Drift Detection
Captures direction, velocity, and stability of mental health patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict

from ..config import MODEL_CONFIG


class TemporalPositionalEncoding(nn.Module):
    """
    Positional encoding that captures temporal relationships.
    Enhanced to represent time-aware positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DriftAttention(nn.Module):
    """
    Drift-aware attention mechanism.
    Modified attention to emphasize temporal change patterns.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Drift bias: emphasize recent changes
        self.drift_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add temporal drift bias (emphasize recent positions)
        temporal_bias = torch.arange(seq_len, device=x.device).float()
        temporal_bias = temporal_bias.unsqueeze(0) - temporal_bias.unsqueeze(1)
        temporal_bias = temporal_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        scores = scores + self.drift_bias * temporal_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)


class DriftTransformerBlock(nn.Module):
    """Single transformer block with drift attention"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = DriftAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class DriftTransformer(nn.Module):
    """
    Main temporal transformer model for Risk Drift Score prediction.
    
    Architecture:
    - Feature embedding layer
    - Temporal positional encoding
    - N drift-aware transformer blocks
    - Temporal pooling
    - RDS prediction head
    """
    
    def __init__(
        self,
        d_features: int = MODEL_CONFIG.d_features,
        d_model: int = MODEL_CONFIG.d_model,
        n_heads: int = MODEL_CONFIG.n_heads,
        n_layers: int = MODEL_CONFIG.n_layers,
        dropout: float = MODEL_CONFIG.dropout,
        max_seq_len: int = MODEL_CONFIG.max_sequence_length
    ):
        super().__init__()
        
        self.d_features = d_features
        self.d_model = d_model
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(d_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = TemporalPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DriftTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Temporal pooling
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # RDS prediction head
        self.rds_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary heads for interpretability
        self.direction_head = nn.Linear(d_model, 1)  # Trend direction
        self.velocity_head = nn.Linear(d_model, 1)   # Rate of change
        self.stability_head = nn.Linear(d_model, 1)  # Fluctuation level
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature sequence (batch, seq_len, d_features)
            mask: Optional attention mask
            
        Returns:
            Dictionary with RDS and auxiliary outputs
        """
        batch_size = x.size(0)
        
        # Embed features
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        # Temporal pooling (attention-based)
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pool_keys = self.temporal_pool(x)
        pool_attn = F.softmax(
            torch.bmm(pool_query, pool_keys.transpose(1, 2)) / math.sqrt(self.d_model),
            dim=-1
        )
        pooled = torch.bmm(pool_attn, x).squeeze(1)  # (batch, d_model)
        
        # Predict RDS
        rds = self.rds_head(pooled)
        
        # Auxiliary outputs
        direction = torch.tanh(self.direction_head(pooled))
        velocity = torch.sigmoid(self.velocity_head(pooled))
        stability = torch.sigmoid(self.stability_head(pooled))
        
        return {
            'rds': rds.squeeze(-1),
            'direction': direction.squeeze(-1),
            'velocity': velocity.squeeze(-1),
            'stability': stability.squeeze(-1),
            'temporal_representation': pooled
        }
    
    def get_attention_weights(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Get attention weights for interpretability"""
        x = self.feature_embedding(x)
        x = self.pos_encoder(x)
        
        # Get attention from first block (simplified)
        # In practice, you'd want to extract from all blocks
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            block = self.transformer_blocks[0]
            
            q = block.attention.w_q(x)
            k = block.attention.w_k(x)
            
            q = q.view(batch_size, seq_len, block.attention.n_heads, -1).transpose(1, 2)
            k = k.view(batch_size, seq_len, block.attention.n_heads, -1).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(block.attention.d_k)
            attn = F.softmax(scores, dim=-1)
            
        return attn.mean(dim=1)  # Average over heads


def create_model(config: Optional[Dict] = None) -> DriftTransformer:
    """Factory function to create model with optional config override"""
    if config is None:
        return DriftTransformer()
    
    return DriftTransformer(
        d_features=config.get('d_features', MODEL_CONFIG.d_features),
        d_model=config.get('d_model', MODEL_CONFIG.d_model),
        n_heads=config.get('n_heads', MODEL_CONFIG.n_heads),
        n_layers=config.get('n_layers', MODEL_CONFIG.n_layers),
        dropout=config.get('dropout', MODEL_CONFIG.dropout),
        max_seq_len=config.get('max_seq_len', MODEL_CONFIG.max_sequence_length)
    )
