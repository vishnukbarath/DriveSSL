import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [batch_size, feature_dim]
        """
        batch_size = z1.size(0)
        device = z1.device

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Cosine similarity matrix
        sim = F.cosine_similarity(
            z.unsqueeze(1),
            z.unsqueeze(0),
            dim=2
        ) / self.temperature  # [2B, 2B]

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim.masked_fill_(mask, -9e15)

        # Positive pairs
        pos_sim = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)

        # Loss
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()
