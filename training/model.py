import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussians:
    def __init__(self, means, scales, rotations, opacities, harmonics):
        self.means = means
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.harmonics = harmonics

class FreeTimeGSModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=4, nhead=4):
        super().__init__()
        
        self.feat_proj = nn.Linear(input_dim, hidden_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Static Base
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 + 3 + 4 + 1 + 3)
        )
        
        # Dynamic Shift
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dynamic_head = nn.Linear(hidden_dim, 3 + 3 + 4 + 1)
        nn.init.zeros_(self.dynamic_head.weight)
        nn.init.zeros_(self.dynamic_head.bias)

    def forward(self, concerto_tokens, concerto_coords, t):
        x = self.feat_proj(concerto_tokens) 
        
        # Base
        base_params = self.base_head(x)
        base_xyz = concerto_coords + torch.tanh(base_params[..., :3]) * 0.1 
        base_scale = base_params[..., 3:6]
        base_rot = F.normalize(base_params[..., 6:10], dim=-1)
        base_opac = base_params[..., 10]
        
        # SH [B, N, 3, 1] for gsplat
        base_sh = base_params[..., 11:].unsqueeze(-1)
        
        # Dynamic
        time_emb = self.time_mlp(t).unsqueeze(1) 
        x_dyn = x + time_emb
        x_dyn = self.transformer(x_dyn)
        deltas = self.dynamic_head(x_dyn)
        
        # === Fusion & Aggressive Initialization ===
        
        # 1. Position: Allow large movement
        final_xyz = base_xyz + deltas[..., :3]
        
        # 2. Scale: Bias +2.0 -> sigmoid(2.0) ~= 0.88
        # Result ~ 0.005 + 0.3 * 0.88 ~= 0.26 (Big enough to see!)
        raw_scale = base_scale + deltas[..., 3:6] + 2.0 
        scale_min, scale_max = 0.005, 0.3
        final_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(raw_scale)
        
        # 3. Rotation
        final_rot = F.normalize(base_rot + deltas[..., 6:10], dim=-1)
        
        # 4. Opacity: Bias +3.0 -> sigmoid(3.0) ~= 0.95 (Solid!)
        raw_opac = base_opac + deltas[..., 10] + 3.0
        final_opac = torch.sigmoid(raw_opac)
        
        return Gaussians(final_xyz, final_scale, final_rot, final_opac, base_sh)