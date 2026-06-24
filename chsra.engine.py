import torch
import torch.nn as nn
import torch.nn.functional as F


class CHSRAEngine(nn.Module):
    """
    CHSRA: Chunked Harmonic Structured Resonance Attention

    Production-ready module for harmonic modulation of neural features.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        feature_dim: int,
        chunk_size: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert feature_dim % chunk_size == 0, "feature_dim must be divisible by chunk_size"

        self.base_layer = base_layer
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.num_chunks = feature_dim // chunk_size

        # 🔹 Learnable harmonic phase matrices
        self.harmonic_phases = nn.Parameter(
            torch.randn(self.num_chunks, chunk_size, chunk_size) * 0.02
        )

        # 🔹 Learnable amplitudes (per chunk)
        self.harmonic_amplitudes = nn.Parameter(
            torch.ones(self.num_chunks, chunk_size) * 0.1
        )

        # 🔹 Dynamic phase generator (input-dependent)
        self.phase_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # 🔹 Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # 🔹 Normalization + Dropout
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

        # 🔹 Residual scaling (stabilizes training)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        x: [batch_size, feature_dim]
        """

        batch_size = x.size(0)

        # Base transformation (e.g., Linear or FFN)
        base_out = self.base_layer(x)

        # Reshape into chunks
        x_chunks = x.view(batch_size, self.num_chunks, self.chunk_size)

        # Generate dynamic phase (input-conditioned)
        dynamic_phase = self.phase_generator(x)
        dynamic_phase = dynamic_phase.view(batch_size, self.num_chunks, self.chunk_size)

        # Harmonic interaction (batched)
        harmonic = torch.matmul(
            x_chunks.unsqueeze(2),                      # [B, C, 1, chunk]
            self.harmonic_phases.unsqueeze(0)          # [1, C, chunk, chunk]
        ).squeeze(2)                                   # [B, C, chunk]

        # Add dynamic phase
        harmonic = harmonic + dynamic_phase

        # Stable modulation
        modulation = 1.0 + torch.tanh(
            self.harmonic_amplitudes.unsqueeze(0) * torch.cos(harmonic)
        )

        modulation = modulation.view(batch_size, -1)

        # Apply modulation
        modulated = base_out * modulation

        # Projection + dropout
        modulated = self.output_proj(modulated)
        modulated = self.dropout(modulated)

        # Residual connection
        out = base_out + self.res_scale * modulated

        # Normalize
        out = self.norm(out)

        return out
