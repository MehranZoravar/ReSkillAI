from __future__ import annotations
import torch
import torch.nn as nn

class SkillTransformer(nn.Module):
    def __init__(self, num_skills: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(num_skills, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)
        h = self.encoder(h)
        out = self.output_proj(h.squeeze(1))
        return self.sigmoid(out)
