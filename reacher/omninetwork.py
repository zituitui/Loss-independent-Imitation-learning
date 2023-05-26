
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
def set_seed(seed: int = 42) -> None:
    #np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    #os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


#set_seed(5)
# Hyper Parameters
EPOCH = 5              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

one_hot_dict=[
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
]


class omni_dataset(Data.Dataset):
    def __init__(self):
        self.data = torch.load('./dataset2.exp')
        return
    def __len__(self):
        return len(self.data)
    def  __getitem__(self, idx):
        result1 = torch.FloatTensor(self.data[idx][0:4])
        result2 = torch.FloatTensor(one_hot_dict[self.data[idx][4]])
        # if self.data[idx][4] :
        #     result2 = torch.Tensor([1,0])
        # else:
        #     result2 = torch.Tensor([0,1])
        return result1, result2
    
class Network_softmax(nn.Module):
    def __init__(self):
        super(Network_softmax, self).__init__()
        self.fc1 = nn.Linear(in_features=4 , out_features=4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4 , out_features=7)

    def forward(self, x):
        #print('$$$$$$$$',self.fc1(x))
        #print(x.dtype)
        #print(self.fc1.weight.dtype)
        output1 = self.relu1(self.fc1(x))  
        output = self.fc2(output1)
        return output, x    # return x for visualization


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
train_data = omni_dataset()

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

network_softmax = Network_softmax()
print(network_softmax)  # net architecture

optimizer_soft = torch.optim.Adam(network_softmax.parameters(), lr=LR)   # optimize all logistic parameters
loss_cross = nn.CrossEntropyLoss()                       # the target label is not one-hotted

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

        output = network_softmax(b_x)[0]               # logistic output
        #print('####output b_y',output,b_y)
        #print('####output b_y',output,b_y)
        loss = loss_cross(output, b_y)   # cross entropy loss
        optimizer_soft.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer_soft.step()                # apply gradients
    print("traing epoch##",epoch)



#print(network_softmax(torch.tensor([ 0.0232,  0.5563,  0.0625, -0.7203])))
import math
import random
def select_action_softmax(state,lor):
    #print('select_action_softmax state',state)
    #print(state.double())
    #print(type(state.double()))
    output = lor(state.to(torch.float32))
    #print(type(output))
    #print('output',output)
    result = torch.argmax(output[0])
    return torch.tensor(result)

def select_action_softmax2(state,lor):
    #print('select_action_softmax state',state)
    #print(state.double())
    #print(type(state.double()))
    output = lor(state.to(torch.float32))
    #print(type(output))
    #print('output',output)
    m = nn.Softmax()
    #print(output[0])
    weight = m(output[0])
    #print(weight)
    ava = torch.sqrt(torch.inner(weight,torch.FloatTensor([0,1,2,3,4,5,6])))
    result = ava.to(torch.int)
    return result

#test env
env_soft = gym.make("gym_robot_arm:robot-arm-v0")
num_episodes = 20
reward_list = []
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    print('env#',i_episode)
    total_reward = 0
    state = env_soft.reset()
    for t in count():
        #print(state)
        action = select_action_softmax2(torch.tensor(state),network_softmax)
        observation, reward, done, _ = env_soft.step(action.item())
        #done = done or _
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
print(sum(reward_list) / len(reward_list))
