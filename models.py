import torch
import torch.nn as nn

class CanonicalMLP(nn.Module):
    """
    Canonise rows and pass it through the MLP
    """

    def __init__(
        self,
        num_points: int = 256,
        n_classes: int = 40,
        hidden_dims=(512, 256),
    ):
        super().__init__()
        self.num_points = num_points
        in_features = num_points * 3

        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _canonise(pc: torch.Tensor) -> torch.Tensor:
        """
        First sort according to the 'x' axis, if equal, then try 'y' axis and so on.
        """
        key = pc[:, :, 0] * 1e8 + pc[:, :, 1] * 1e4 + pc[:, :, 2]

        # argsort along point dimension
        idx = key.argsort(dim=1)        

        # Expand idx to (B, N, 3) so we can gather the xyz triplets
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)

        pc_sorted = torch.gather(pc, 1, idx_exp)   
        return pc_sorted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, 3)
        """
        x = self._canonise(x)               # map to the orbit representative
        x = x.reshape(x.size(0), -1)        # flatten
        return self.mlp(x)                  # forward pass