import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math


class CanonizationMLP(nn.Module):
    """
    Canonise rows and pass it through the MLP
    """

    def __init__(
        self,
        num_points=256,
        n_classes=40,
        hidden_dims=(256, 128),
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
    


class SymmetrizationMLP(nn.Module):
    """
    Subsample K points, iterate over all permutations of the points and average the MLP output.
    """

    def __init__(
        self,
        num_points=256,
        sampled_points=15,     # number of subsampled points
        n_classes=40,
        hidden_dims=(64, 32),
    ):
        
        super().__init__()
        self.num_points = num_points
        self.K = sampled_points

        # simple (vanilla) MLP
        in_features = self.K * 3
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))          # single‑class output
        self.mlp = nn.Sequential(*layers)

        # save the permutations in a buffer
        perm_list = list(itertools.permutations(range(self.K)))
        self.register_buffer(
            "perms",
            torch.tensor(perm_list, dtype=torch.long),  
            persistent=False,  # don't save in state_dict
        )
        self.group_size = len(perm_list)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, 3)                  
        """

        B, N, _ = x.shape
        device = x.device
        G = self.group_size    
        K = self.K

        # subsample K points from each cloud in the batch
        idx = torch.stack([torch.randperm(N, device=device)[:K] for _ in range(B)])
        pts    = torch.gather(
                    x, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
                )                                             # (B,K,3)

        # permute each cloud in all possible ways
        perms = self.perms.to(device)                        # (G,K)

        pts_exp = pts.unsqueeze(1).expand(B, G, K, 3)       # (B,G,K,3)
        perms_exp = perms.unsqueeze(0).unsqueeze(-1)          # (1,G,K,1)
        perms_exp = perms_exp.expand(B, G, K, 3)              # (B,G,K,3)

        permuted = torch.gather(pts_exp, 2, perms_exp)       # (B,G,K,3)

        # pass through MLP and pool (mean) over the group
        flat = permuted.reshape(B * G, K * 3)                 # (B·G, K*3)
        y = self.mlp(flat).view(B, G, -1)                  # (B,G,n_classes)

        return y.mean(dim=1)                                  # (B,n_classes)


class SampledSymmetrizationMLP(nn.Module):
    """
    
    """

    def __init__(
        self,
        num_points=256,
        sampled_pts=256, 
        n_classes=40,      
        hidden_dims=(128, 64)
    ):
        super().__init__()
        self.N = num_points
        self.K = sampled_pts

        # num of permuations to sample
        full_size = math.factorial(self.K)
        self.P = math.ceil(0.05 * full_size)
        assert self.P >= 1, "need at least one permutation sample"

        # vanilla MLP
        layers, prev = [], self.K * 3
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))    
        self.mlp = nn.Sequential(*layers)


    def _subsample_rows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Subsample K points from the cloud
        """
        B, N, _ = x.shape
        idx = torch.randperm(N, device=x.device)[: self.K]  # (K,)
        return x[:, idx]                               


    def _sample_permutations(self, device: torch.device) -> torch.Tensor:
        """
        Subsample P permutations
        """
        perms = [torch.randperm(self.K, device=device) for _ in range(self.P)]
        return torch.stack(perms)                           # (P,K)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,N,3) 
        """
        B = x.size(0)
        device = x.device

        pts = self._subsample_rows(x)                       # (B,K,3)
        perms = torch.stack(                            # (B*P, K)
        [torch.randperm(self.K, device=device) for _ in range(B * self.P)]
        ).view(B, self.P, self.K)           # (P,K)

        # create (B,P,K,3) tensor of permuted copies
        pts_exp  = pts[:, None].expand(B, self.P, self.K, 3)
        perms_exp = perms[None, :, :, None].expand_as(pts_exp)
        pts_perm = torch.gather(pts_exp, 2, perms_exp)      # (B,P,K,3)

        # flatten  and pass through MLP
        flat = pts_perm.reshape(B * self.P, -1)             # (B·P, K*3)
        y = self.mlp(flat).view(B, self.P, 1)               # (B,P,n_classes)

        # average over sampled permutations 
        return y.mean(dim=1)                                # (B,n_classes)