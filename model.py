# model.py
import torch
import torch.nn as nn

class DecoderOnlyModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, pos_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.pos_emb_dim = pos_emb_dim
        self.total_dim = input_dim + target_dim

        # Positional embedding layer
        self.pos_emb = nn.Embedding(self.total_dim, pos_emb_dim)

        # Feature projection layer
        self.feature_proj = nn.Linear(1 + pos_emb_dim, 1)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, target_dim)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # Get positional embeddings
        pos_embeddings = self.pos_emb(positions)

        # Reshape input features
        x_reshaped = x.view(batch_size, seq_len, 1)
        x_with_pos = torch.cat([x_reshaped, pos_embeddings], dim=-1)

        # Feature projection
        projected = self.feature_proj(x_with_pos)
        projected_flat = projected.squeeze(-1)

        # Decoder processing
        return self.decoder(projected_flat)