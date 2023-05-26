
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


x = torch.load('./dataset2.exp')
idx = 1
print(type(x))          #<class 'list'>
print(type(x[idx]))     #<class 'list'>
print(x[idx][0])        #154.0009230900539
print(x[idx][1])        #119.48021492992245
print(x[idx])           #[154.0009230900539, 119.48021492992245, 0.38514320835637633, 0.9553195247369468, 3]
print(len(x))