# core/forecasting.py
import torch
import torch.nn as nn
from typing import Dict, Tuple

class ClinicalTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer architecture for PD progression.
    Integrates Variable Selection Networks (VSN) and LSTM encoders.
    """
    def __init__(self, 
                 static_dim: int, 
                 dynamic_dim: int, 
                 hidden_size: int = 64, 
                 quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
        # 1. Variable Selection Networks (simplified as linear projections here)
        # Embeds static data (e.g., genetics APOL1/GBA)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.ELU()
        )
        
        # Embeds time-series (e.g., UPDRS, SBR history)
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_size),
            nn.ELU()
        )

        # 2. LSTM Encoder-Decoder Layer (Seq2Seq component)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            batch_first=True, 
            bidirectional=False
        )
        
        # 3. Gated Residual Network (GRN) & Attention headers would go here
        # Using MultiheadAttention for interpretability
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        
        # 4. Quantile Output Layer (Predicts 10th, 50th, 90th percentiles)
        # This provides the "Uncertainty Awareness" required.
        self.output_layer = nn.Linear(hidden_size, len(quantiles))

    def forward(self, 
                static_x: torch.Tensor, 
                dynamic_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            static_x: (Batch, Static_Feats)
            dynamic_x: (Batch, Time_Steps, Dynamic_Feats)
        Returns:
            predictions: (Batch, Time_Steps, Quantiles)
            attention_weights: (Batch, Time_Steps, Time_Steps) -> For Interpretability
        """
        # A. Encode Static Context
        c_static = self.static_encoder(static_x) # (Batch, Hidden)
        
        # B. Encode Dynamic Inputs
        x_dynamic = self.dynamic_encoder(dynamic_x) # (Batch, Time, Hidden)
        
        # C. Fuse Static context into Dynamic stream (Broadcasting addition)
        # In full TFT, this is done via Gated Linear Units
        combined_features = x_dynamic + c_static.unsqueeze(1)
        
        # D. Temporal Processing
        lstm_out, _ = self.lstm(combined_features)
        
        # E. Attention (Self-attention on history)
        # Permute for Torch Attention: (Time, Batch, Feat)
        lstm_permuted = lstm_out.permute(1, 0, 2)
        attn_out, attn_weights = self.attention(lstm_permuted, lstm_permuted, lstm_permuted)
        attn_out = attn_out.permute(1, 0, 2) # Back to (Batch, Time, Feat)
        
        # F. Residue connection + Norm
        out = torch.add(attn_out, lstm_out) # Simple skip connection
        
        # G. Forecast
        predictions = self.output_layer(out)
        
        return predictions, attn_weights

    def compute_loss(self, preds, target):
        """Quantile Loss function for uncertainty estimation."""
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        return torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))