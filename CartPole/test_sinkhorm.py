import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ot
def calculate_w2(supportA: torch.Tensor, massA: torch.Tensor, supportB: torch.Tensor, massB: torch.Tensor) -> float:
    r"""
    calculate the 2-wasserstein distance between distributionA and distributionB

    Args:
        supportA (Tensor): a mini-batch tensor of shape (B x D)
        massA (Tensor): a mini-batch tensor of shape (B)
        supportB (Tensor): a mini-batch tensor of shape (B x D)
        massB (Tensor): a mini-batch tensor of shape (B)

    Returns:
        result (float): the 2-wasserstein distance between distributionA and distributionB
    """
    print(supportA.size(),massA.size(),supportB.size(),massB.size())
    if __debug__ and ((not isinstance(supportA, torch.Tensor)) or (not isinstance(massA, torch.Tensor)) or (not isinstance(supportB, torch.Tensor)) or (not isinstance(massB, torch.Tensor))):
        raise RuntimeError('input should be of type torch.Tensor')
    massA = massA.cpu().numpy().astype(np.float64)
    massB = massB.cpu().numpy().astype(np.float64)
    massA = massA / massA.sum()
    massB = massB / massB.sum()

    
    cost_matrix = torch.cdist(supportA, supportB, p = 2).pow(2).cpu().numpy()
    trans_plan = ot.emd(massA, massB, cost_matrix)
    result = np.sqrt((trans_plan * cost_matrix).sum().item()).astype(np.float32)
    return result

wd_0 = calculate_w2(supportA = torch.FloatTensor([[0,0],[1,1],[2,2],[3,3],[4,4]])
                    ,massA= torch.ones(5)
                    ,supportB = torch.FloatTensor([[4,4]])
                    ,massB= torch.ones(1))

print(wd_0)
wd_1 = calculate_w2(supportA = torch.FloatTensor([[0,0],[1,1],[2,2],[3,3],[4,4]])
                    ,massA= torch.ones(5)
                    ,supportB = torch.FloatTensor([[4,2]])
                    ,massB= torch.ones(1))

print(wd_1)