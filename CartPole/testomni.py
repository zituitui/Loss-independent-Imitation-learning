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

def to_space(i):
    return torch.cat((i[0],i[1].reshape(1)))

dataset = torch.load('./dataset100.exp')
idx = 45
 
expdata = [to_space(i) for i in dataset]
expdata = [i.cpu().numpy() for i in expdata]
expdata = np.array(expdata)
mean = np.mean(np.absolute(expdata), axis=0)
mean_4 = mean[4]
print(mean)
mean[4] = 1/(2*mean_4)
print('mean is ',mean)
print('expdata[idx] is ',expdata[idx])
expdata = expdata / mean
print('dataset[idx] is ',dataset[idx])
print('after expdata[idx] is',expdata[idx])
print('expdata[idx][0:4]',expdata[idx][0:4])
r = 0
for idx in range(0,99):
    wd_0 = calculate_w2(torch.cat((torch.tensor(expdata[idx][0:4]),torch.zeros(1))).unsqueeze(0),torch.ones(1),
                        torch.FloatTensor(expdata)[0:100,:]
                        ,torch.ones(100))
    wd_1 = calculate_w2(torch.cat((torch.tensor(expdata[idx][0:4]),torch.FloatTensor([2*mean_4]))).unsqueeze(0),torch.ones(1),
                        torch.FloatTensor(expdata)[0:100,:]
                        ,torch.ones(100))
    if(wd_0<wd_1 and dataset[idx][1] > 0.5):
        r +=1
print(r)
# print(wd_0)
# print(wd_1)
