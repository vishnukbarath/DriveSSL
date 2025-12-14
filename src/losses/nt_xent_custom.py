import torch
import torch.nn.functional as F


def nt_xent_custom(z1, z2, temperature=0.5):
    """
    Custom NT-Xent Loss for SimCLR embeddings

    Args:
        z1: [B, D] embeddings of first augmented view
        z2: [B, D] embeddings of second augmented view
        temperature: scaling factor for cosine similarity
    Returns:
        scalar loss
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2*B, D]
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature  # [2*B, 2*B]

    # Mask to remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # Positive pairs: i-th with i+B and i+B with i
    positives = torch.cat([torch.arange(batch_size, 2*batch_size),
                           torch.arange(0, batch_size)]).to(z.device)

    loss = F.cross_entropy(sim_matrix, positives)
    return loss
