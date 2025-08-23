import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassBalancedFocal(nn.Module):
    def __init__(self, samples_per_class, beta: float = 0.9999, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        eff_num = (1 - np.power(beta, samples_per_class)) / (1 - beta)
        weights = (1.0 / np.maximum(eff_num, 1e-12))
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # aafety
        weight = self.weight
        if weight.device != logits.device:
            weight = weight.to(logits.device)

        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class LogitAdjustedCE(nn.Module):
    def __init__(self, priors, la_tau: float = 1.0, reduction: str = "mean"):
        super().__init__()
        adj = np.log(np.maximum(priors, 1e-12)) * la_tau
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # safety
        adj = self.adj
        if adj.device != logits.device:
            adj = adj.to(logits.device)

        logits = logits - adj
        return F.cross_entropy(logits, target, reduction=self.reduction)