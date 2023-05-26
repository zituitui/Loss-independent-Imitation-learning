import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import torch
import torch.nn as nn
import torch.utils.data as Data
from itertools import count
import torchvision
from mse_network import Network_mse, select_action_mse, mse_dataset
from omninetwork import Network_softmax, select_action_softmax ,omni_dataset,select_action_softmax2

BATCH_SIZE = 50
LR = 0.001              # learning rate



mse_train_data = mse_dataset()
mse_train_loader = Data.DataLoader(dataset = mse_train_data, batch_size=BATCH_SIZE, shuffle=True)

omni_train_data = omni_dataset()
omni_train_loader = Data.DataLoader(dataset = omni_train_data, batch_size=BATCH_SIZE, shuffle=True)

network_softmax = Network_softmax()
optimizer_soft = torch.optim.Adam(network_softmax.parameters(), lr=LR) 
loss_cross = nn.CrossEntropyLoss()

network_mse = Network_mse()
optimizer_mse = torch.optim.Adam(network_mse.parameters(), lr=LR)   # optimize all logistic parameters
loss_mse = nn.MSELoss() 
step_size = 1
for EPOCH in range(10,150,step_size):
    print(EPOCH)

    

    for epoch in range(step_size):
        for step, (b_x, b_y) in enumerate(omni_train_loader):   # gives batch data, normalize x when iterate train_loader
            # print(b_x.size())
            #b_x = b_x.view(-1, 28*28)
            # print(b_x.size())

            output = network_softmax(b_x)[0]               # logistic output
            #print('####output b_y',output,b_y)
            #print('####output b_y',output,b_y)
            loss = loss_cross(output, b_y)   # cross entropy loss
            optimizer_soft.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer_soft.step()                # apply gradients
        print("softmax traing epoch##",EPOCH)
    
    env_soft = gym.make("gym_robot_arm:robot-arm-v0")
    num_episodes = 200
    reward_list = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        print('env#',i_episode)
        total_reward = 0
        state = env_soft.reset()
        for t in count():
            #print(state)
            action = select_action_softmax(torch.tensor(state),network_softmax)
            observation, reward, done, _= env_soft.step(action.item())
            #done = done or _
            #print(reward)
            total_reward += reward
            if done or t>200:
                state = None
                #print('total r:',total_reward)
                reward_list.append(total_reward)
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device='cpu')
                #print(state)
    
    print(reward_list)
    ava_r = sum(reward_list)/len(reward_list)
    x1 = np.full((1),EPOCH)
    y1 = [ava_r]
    if EPOCH == 10:
        plt.scatter(x1, y1, color = 'red',alpha=0.5,label = 'OmniLayer with infinity loss')
    else:
        plt.scatter(x1, y1, color = 'red',alpha=0.5)

    env_soft2 = gym.make("gym_robot_arm:robot-arm-v0")
    num_episodes = 200
    reward_list = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        print('env#',i_episode)
        total_reward = 0
        state = env_soft2.reset()
        for t in count():
            #print(state)
            action = select_action_softmax2(torch.tensor(state),network_softmax)
            observation, reward, done, _= env_soft2.step(action.item())
            #done = done or _
            #print(reward)
            total_reward += reward
            if done or t>200:
                state = None
                #print('total r:',total_reward)
                reward_list.append(total_reward)
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device='cpu')
                #print(state)
    
    print(reward_list)
    ava_r = sum(reward_list)/len(reward_list)
    x1 = np.full((1),EPOCH)
    y1 = [ava_r]
    if EPOCH == 10:
        plt.scatter(x1, y1, color = 'green',alpha=0.5,label = 'OmniLayer with l2 loss')
    else:
        plt.scatter(x1, y1, color = 'green',alpha=0.5)
    
    for epoch in range(step_size):
        for step, (b_x, b_y) in enumerate(mse_train_loader):   # gives batch data, normalize x when iterate train_loader
            # print(b_x.size())
            #b_x = b_x.view(-1, 28*28)
            # print(b_x.size())
            
            output = network_mse(b_x)[0]               # logistic output
            #print('output ,b_y',output ,b_y)
            loss = loss_mse(output, b_y.float())   # cross entropy loss
            optimizer_mse.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer_mse.step()                # apply gradients
        print("mse traing epoch##",EPOCH)  
    env_mse = gym.make("gym_robot_arm:robot-arm-v0")
    num_episodes = 200
    reward_list = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        #print('env#',i_episode)
        total_reward = 0
        state = env_mse.reset()
        #print('reset done')
        for t in count():
            #print(state)
            #print('t=',t)
            action = select_action_mse(torch.tensor(state),network_mse)
            observation, reward, done, _ = env_mse.step(action.item())
            #done = done or _
            #print(reward)
            total_reward += reward
            if done or t>200:
                state = None
                #print('total r:',total_reward)
                reward_list.append(total_reward)
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device='cpu')
                #print(state)
        #print(total_reward) 
    print(reward_list)
    ava_r = sum(reward_list)/len(reward_list)
    x1 = np.full((1),EPOCH)
    y1 = [ava_r]
    if EPOCH == 10:
        plt.scatter(x1, y1, color = 'blue',alpha=0.5,label = 'No OmniLayer')
    else:
        plt.scatter(x1, y1, color = 'blue',alpha=0.5)


plt.xlabel("Training Epoch")
plt.ylabel("Environment Reward")
plt.legend(loc='upper left')
plt.savefig('./scatter3.png')
                             


