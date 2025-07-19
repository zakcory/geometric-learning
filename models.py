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
        sampled_points=8,     # number of subsampled points
        n_classes=40,
        hidden_dims=(256, 128),
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
        out = []

        perms = self.perms.to(device)         # (G,K)

        for b in range(B):                    # loop over clouds
            pts  = x[b, torch.randperm(N, device=device)[:K]]   # (K,3)
            perm = pts[perms]                                    # (G,K,3)
            flat = perm.reshape(G, -1)                           # (G,K*3)
            y    = self.mlp(flat).mean(dim=0)                    # (n_classes,)
            out.append(y)

        return torch.stack(out, dim=0)                           # (B,n_classes)


class SampledSymmetrizationMLP(nn.Module):
    """
    Subsample 5% of the group elements (row permutations) and average the output of the MLP over the sampled permutations
    """

    def __init__(
        self,
        num_points=256,
        sampled_pts=8, 
        n_classes=40,      
        hidden_dims=(256, 128)
    ):
        super().__init__()
        self.N = num_points
        self.K = sampled_pts

        # num of permuations to sample
        full_size = math.factorial(self.K)
        self.P = (full_size * 5 + 99) // 100
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
        pts_exp   = pts.unsqueeze(1).expand(-1, self.P, -1, 3)      # (B, P, K, 3)
        perms_exp = perms.unsqueeze(-1).expand_as(pts_exp)          # (B, P, K, 3)
        pts_perm = torch.gather(pts_exp, 2, perms_exp)      # (B,P,K,3)

        # flatten  and pass through MLP
        flat = pts_perm.reshape(B * self.P, -1)             # (B·P, K*3)
        y = self.mlp(flat).view(B, self.P, -1)               # (B,P,n_classes)

        # average over sampled permutations 
        return y.mean(dim=1)                                # (B,n_classes)


class EqvLayer(nn.Module):
    """
    Equivariant layer
    """

    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=bias)
        self.W_agg  = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        """
        x : (B, N, d_in)
        return : (B, N, d_out)
        """
        mean = x.mean(dim=1, keepdim=True)         
        y = self.W_self(x) + self.W_agg(mean)         
        return y


class EquivariantLinearMLP(nn.Module):
    """
    Linear equivariant layers + point-wise non-linearities + invariant layer
    """

    def __init__(
        self,
        d_in=3,
        hidden_dims=(256, 128),  
        n_classes=40,
    ):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_dims:
            layers += [EqvLayer(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.phi = nn.Sequential(*layers)              # equivariant
        self.head = nn.Linear(prev, n_classes)         # invariant 

    def forward(self, x):
        """
        x : (B, N, 3)
        """
        x = self.phi(x)                # equivariant (B,N,H)
        x = x.mean(dim=1)              # invariant  (B,H)
        return self.head(x)            # logits (B,n_classes)


class DataAugmentationMLP(nn.Module):
    """
    Standard MLP + permutation data augmentation
    """

    def __init__(
        self,
        num_points: int = 256,
        n_classes: int = 40,
        hidden_dims=(256, 128),
    ):
        super().__init__()
        self.num_points = num_points

        layers, prev = [], num_points * 3
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _random_permute(x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, 3)  randomly permuted rows, independently per sample.
        """
        B, N, _ = x.shape
        idx = torch.stack([torch.randperm(N, device=x.device) for _ in range(B)])
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, 3)
        """
        if self.training:
            x = self._random_permute(x)      # data augmentation only in train

        x = x.reshape(x.size(0), -1)         # flatten 
        return self.mlp(x)                   # logits (B, n_classes)
