import random
import numpy as np

import torch
import torch.nn.functional as F


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_knn_3d(xyz, kernel_size=5, knn=20):
    """ Use 2D Conv to compute neighbour distance and find k nearest neighbour
          xyz: (B, 3, H, W)

      Returns:
        idx: (B, H*W, k)
    """
    batch_size, _, height, width = xyz.shape
    assert (kernel_size % 2 == 1)
    hk = (kernel_size // 2)
    k2 = kernel_size ** 2

    t = np.zeros((kernel_size, kernel_size, 1, kernel_size ** 2))
    ind = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
                t[i, j, 0, ind] -= 1.0
                t[hk, hk, 0, ind] += 1.0
                ind += 1
    weight = np.zeros((kernel_size, kernel_size, 3, 3 * k2))
    weight[:, :, 0:1, :k2] = t
    weight[:, :, 1:2, k2:2*k2] = t
    weight[:, :, 2:3, 2*k2:3*k2] = t
    weight = torch.tensor(weight).float()

    weights_torch = torch.Tensor(weight.permute((3, 2, 0, 1))).to(xyz.device)
    dist = F.conv2d(xyz, weights_torch, padding=hk)

    dist_flat = dist.contiguous().view(batch_size, 3, k2, -1)
    dist2 = torch.sum(dist_flat ** 2, dim=1)

    _, nn_idx = torch.topk(-dist2, k=knn, dim=1)
    nn_idx = nn_idx.permute(0, 2, 1)
    h_offset = (nn_idx % k2) // kernel_size - hk
    w_offset = nn_idx % kernel_size - hk

    idx = torch.arange( height * width).to(xyz.device)
    idx = idx.view(1, -1, 1).expand(batch_size, -1, knn)
    idx = idx + (h_offset * width) + w_offset

    idx = torch.clamp(idx, 0, height * width - 1)

    return idx
