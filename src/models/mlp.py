import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion_factor: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor * 2),
            GEGLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * expansion_factor, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int, expansion_factor: int = 4, dropout_rate: float = 0.5):

        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(dim=hidden_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )
        
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.output_projection(x)
        return x
