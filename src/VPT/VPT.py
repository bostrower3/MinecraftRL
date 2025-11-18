import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from einops import rearrange
import math

# -----------------------------
# 1. Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x


# -----------------------------
# 2. Temporal Transformer
# -----------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth=4, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                ),
            ]))

    def forward(self, x):
        # x: (B, T, D)
        for norm1, attn, ff in self.layers:
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + ff(x)
        return self.norm(x)


# -----------------------------
# 3. EfficientNet Encoder
# -----------------------------
class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True, output_dim=1536):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.output_dim = output_dim

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, frames):
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        feats = self.backbone(frames)  # (B*T, D)
        feats = rearrange(feats, "(b t) d -> b t d", b=B, t=T)
        return feats


# -----------------------------
# 4. Latent Gaussian Head
# -----------------------------
class LatentHead(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, D) -> mean pool first
        x = x.mean(dim=1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization
        return z, mu, logvar


# -----------------------------
# 5. Full GROOT-style Encoder
# -----------------------------
class GROOTEncoder(nn.Module):
    def __init__(self, latent_dim=512, freeze_backbone=True):
        super().__init__()
        self.visual_encoder = EfficientNetEncoder(pretrained=True, freeze=freeze_backbone)
        self.temporal_encoder = TemporalTransformer(dim=1536, depth=4, heads=8, mlp_dim=2048)
        self.pos_encoding = PositionalEncoding(1536)
        self.latent_head = LatentHead(1536, latent_dim)

    def forward(self, video):
        # video: (B, T, 3, H, W)
        feats = self.visual_encoder(video)
        feats = self.pos_encoding(feats)
        feats = self.temporal_encoder(feats)
        z, mu, logvar = self.latent_head(feats)
        return z, mu, logvar


# -----------------------------
# 6. Testing forward/backward
# -----------------------------
if __name__ == "__main__":
    model = GROOTEncoder(latent_dim=512)
    video = torch.randn(2, 8, 3, 224, 224)  # batch=2, seq_len=8
    z, mu, logvar = model(video)
    print("z:", z.shape, "mu:", mu.shape, "logvar:", logvar.shape)

    loss = (mu ** 2 + logvar.exp() - logvar - 1).mean()  # simple KL example
    loss.backward()
    print("Backward pass success ✅")
