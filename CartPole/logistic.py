
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
# import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 25               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

class exp_dataset(Data.Dataset):
    def __init__(self):
        self.data = x = torch.load('./dataset1068.exp')
        return
    def __len__(self):
        return len(self.data)
    def  __getitem__(self, idx):
        return self.data[idx][0].to("cpu"), self.data[idx][1].item()
    

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',
#     train=True,                                     # this is training data
#     transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                                     # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#     download=DOWNLOAD_MNIST,
# )
train_data = exp_dataset()

# plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
print(type(train_data))
print('$$',train_data.__len__())
print('&&',type(train_data.__getitem__(3)))#<class 'tuple'>
print('&&',type(train_data.__getitem__(3)[0]))#<class 'torch.Tensor'>
print('&&',type(train_data.__getitem__(3)[1]))#<class 'int'>

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]


class logisticRg(nn.Module):
    def __init__(self):
        super(logisticRg, self).__init__()
        self.lr = nn.Sequential(
            nn.Linear(4,2)
        )

    def forward(self, x):
        output = self.lr(x)
        return output, x    # return x for visualization


lor = logisticRg()
print(lor)  # net architecture

optimizer = torch.optim.Adam(lor.parameters(), lr=LR)   # optimize all logistic parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        # print(b_x.size())
        #b_x = b_x.view(-1, 28*28)
        # print(b_x.size())

        output = lor(b_x)[0]               # logistic output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
    print("traing epoch##",epoch)



#print(lor(torch.tensor([ 0.0232,  0.5563,  0.0625, -0.7203])))

def select_action(state,lor):
    #print(state)
    output = lor(state)
    #print(output,'@@@@')
    result = 1
    if output[0][0].item()>output[0][1].item():
        result = 0
    return torch.tensor(result)

#test env
env = gym.make("CartPole-v1")
num_episodes = 30
reward_list = []
for i_episode in range(num_episodes):
    print('env#',i_episode)
    # Initialize the environment and get it's state
    total_reward = 0
    state,a = env.reset()
    for t in count():
        #print(state)
        action = select_action(torch.tensor(state),lor)
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

print(reward_list)
