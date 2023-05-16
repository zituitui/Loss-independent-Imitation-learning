
# library
# standard library
import os
import gym

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
from itertools import count
import torchvision
import ot
import random
import numpy as np

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
    #print(supportA.size(),massA.size(),supportB.size(),massB.size())
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


dataset = torch.load('./dataset1068.exp')

#expdata = dataset[:,[0,0]]
print(type(dataset))
expdata = [to_space(i) for i in dataset]
expdata = [i.cpu().numpy() for i in expdata]
expdata = np.array(expdata)
#mean = np.mean(np.absolute(expdata), axis=0)
#print('test',expdata[1])
#expdata = expdata / mean
print('expdata[1]',expdata[1])
#print('mean',mean)
print(type(expdata))
print('expdata[0]:',expdata[0])
print(type(torch.FloatTensor(expdata)))
print('expdata[0]:',torch.FloatTensor(expdata)[0])

#print(torch.tensor()dataset[0:1, 0:1 ,:])
def select_action(state,expdata,sample_num1 = 200,sample_start = 0):
    #print('staring select action')
    #print('state',state)
    #print(torch.cat((state,torch.zeros(1))).unsqueeze(0)) #tensor([-0.0169,  0.0050,  0.0312,  0.0267,  0.0000])
    perm = torch.randperm(torch.FloatTensor(expdata).size(0))
    idx = perm[:200]
    #print(idx)
    #print(torch.FloatTensor(expdata)[idx,:][0])
    #sample_num1 = 200
    sample_num2 = 200
    wd_0 = calculate_w2(torch.concat((torch.cat((state,torch.zeros(1))).unsqueeze(0),torch.FloatTensor(expdata)[sample_start:sample_start+sample_num1,:]),0)
                        ,torch.cat((torch.tensor([100]),torch.ones(sample_num1)))
                        ,torch.FloatTensor(expdata)[0:1068,:]
                        ,torch.ones(1068))
    wd_1 = calculate_w2(torch.concat((torch.cat((state,torch.ones(1))).unsqueeze(0),torch.FloatTensor(expdata)[sample_start:sample_start+sample_num1,:]),0) 
                        ,torch.cat((torch.tensor([100]),torch.ones(sample_num1)))
                        ,torch.FloatTensor(expdata)[0:1068,:]
                        ,torch.ones(1068))
    result = None
    #if(random.random()<0.5):
    if(wd_0 > wd_1):
        result = 1
    else:
        result = 0
    return torch.tensor(result)

#test env
sample_num1 = 200
#sample_num2 = 200
sample_start = 200
best_ava_reward = 0
bestnum1=0
bestnum2=0

env = gym.make("CartPole-v1")
num_episodes = 30

for i in range(50):
    sample_num1 = i*10
    for j in range(10):
        sample_start = j*50


        reward_list = []
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            total_reward = 0
            state,a = env.reset()
            #print('starting ',i_episode)
            for t in count():
                #print(state)
                action = select_action(torch.tensor(state),expdata,sample_num1=sample_num1,sample_start=sample_start)
                observation, reward, done, _ ,a= env.step(action.item())
                #print(reward)
                total_reward += reward
                if done:
                    state = None
                    #print('total r:',total_reward)
                    reward_list.append(total_reward)
                    break
                else:
                    state = torch.tensor(observation, dtype=torch.float32, device='cpu')
                    #print(state)
            #print(i_episode,'total_reward',total_reward)
        ava_r = sum(reward_list)/len(reward_list)
        print('### i,j=',i,',',j, 'avar = ',ava_r)
        if(ava_r>best_ava_reward):
            best_ava_reward = ava_r
            bestnum1 = sample_num1
            bestnum2 = sample_start

print('bestnum1',bestnum1)
print('bestnum2',bestnum2)
print('best reward',best_ava_reward)

#print(reward_list)
