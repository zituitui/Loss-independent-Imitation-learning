

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


x = torch.load('./dataset100.exp')
idx = 1
print(type(x[idx]))           #[tensor([ 0.0232,  0.5563,  0.0625, -0.7203], device='cuda:0'), tensor(0, device='cuda:0')]
print(x[idx][0])        #tensor([ 0.0232,  0.5563,  0.0625, -0.7203], device='cuda:0')
print(x[idx][0][0])     #tensor(0.0232, device='cuda:0')
print(x[idx][1])        #tensor(0, device='cuda:0')
# x1 = torch.zeros(1,1).to("cuda:0")
# x2 = torch.zeros(1,1).to("cuda:1")
# print(x1,x2)
# x3 = x1+x2.to("cuda:0")
# print(x1,x2,x3)

r = [68.0, 64.0, 82.0, 63.0, 63.0, 58.0, 125.0, 65.0, 71.0, 91.0, 63.0, 58.0, 156.0, 51.0, 74.0, 75.0, 114.0, 141.0, 49.0, 62.0, 73.0, 77.0, 83.0, 110.0, 228.0, 56.0, 54.0, 66.0, 91.0, 54.0, 79.0, 180.0, 68.0, 82.0, 53.0, 81.0, 74.0, 96.0, 67.0, 137.0, 76.0, 74.0, 71.0, 72.0, 56.0, 51.0, 154.0, 69.0, 51.0, 146.0, 81.0, 69.0, 98.0, 53.0, 57.0, 59.0, 54.0, 57.0, 75.0, 99.0, 64.0, 63.0, 59.0, 85.0, 49.0, 67.0, 82.0, 65.0, 60.0, 53.0, 72.0, 172.0, 132.0, 87.0, 142.0, 85.0, 77.0, 55.0, 86.0, 194.0, 51.0, 56.0, 79.0, 52.0, 81.0, 61.0, 134.0, 70.0, 71.0, 54.0, 65.0, 61.0, 67.0, 85.0, 72.0, 77.0, 83.0, 225.0, 50.0, 152.0]
r2=[94.0, 65.0, 93.0, 92.0, 80.0, 121.0, 96.0, 218.0, 68.0, 95.0, 83.0, 348.0, 146.0, 82.0, 72.0, 167.0, 76.0, 64.0, 79.0, 90.0, 81.0, 83.0, 88.0, 96.0, 83.0, 165.0, 102.0, 99.0, 146.0, 94.0, 89.0, 105.0, 1559.0, 78.0, 76.0, 91.0, 87.0, 68.0, 101.0, 113.0, 84.0, 120.0, 71.0, 86.0, 344.0, 78.0, 119.0, 84.0, 107.0, 113.0, 678.0, 102.0, 86.0, 107.0, 92.0, 110.0, 102.0, 84.0, 97.0, 109.0, 76.0, 318.0, 112.0, 85.0, 69.0, 88.0, 904.0, 102.0, 102.0, 210.0, 154.0, 330.0, 83.0, 72.0, 88.0, 124.0, 72.0, 57.0, 90.0, 74.0, 87.0, 83.0, 63.0, 68.0, 73.0, 100.0, 95.0, 80.0, 103.0, 466.0, 67.0, 88.0, 102.0, 92.0, 81.0, 204.0, 62.0, 78.0, 106.0, 107.0]
r3 = [27.0, 79.0, 37.0, 23.0, 15.0, 46.0, 29.0, 42.0, 26.0, 45.0, 52.0, 60.0, 53.0, 93.0, 37.0, 40.0, 33.0, 26.0, 41.0, 67.0, 46.0, 24.0, 31.0, 32.0, 65.0, 72.0, 30.0, 54.0, 36.0, 48.0, 17.0, 17.0, 41.0, 50.0, 73.0, 53.0, 20.0, 10.0, 47.0, 67.0, 69.0, 93.0, 27.0, 88.0, 24.0, 95.0, 45.0, 46.0, 26.0, 58.0, 48.0, 30.0, 35.0, 50.0, 67.0, 30.0, 31.0, 29.0, 30.0, 21.0, 35.0, 66.0, 91.0, 41.0, 39.0, 40.0, 48.0, 51.0, 19.0, 50.0, 34.0, 32.0, 39.0, 32.0, 15.0, 112.0, 47.0, 38.0, 55.0, 75.0, 19.0, 33.0, 36.0, 92.0, 28.0, 33.0, 36.0, 25.0, 57.0, 52.0, 60.0, 49.0, 43.0, 21.0, 32.0, 67.0, 35.0, 43.0, 32.0, 92.0]
r4 = [81.0, 90.0, 85.0, 97.0, 79.0, 92.0, 87.0, 102.0, 75.0, 71.0, 76.0, 97.0, 68.0, 127.0, 90.0, 102.0, 100.0, 73.0, 75.0, 75.0, 65.0, 77.0, 69.0, 101.0, 63.0, 80.0, 68.0, 65.0, 190.0, 111.0, 69.0, 70.0, 77.0, 101.0, 85.0, 70.0, 69.0, 93.0, 97.0, 80.0, 77.0, 93.0, 83.0, 100.0, 194.0, 100.0, 77.0, 164.0, 124.0, 71.0, 71.0, 70.0, 83.0, 65.0, 73.0, 64.0, 91.0, 84.0, 101.0, 67.0, 95.0, 76.0, 72.0, 124.0, 87.0, 105.0, 98.0, 130.0, 85.0, 85.0, 66.0, 89.0, 98.0, 93.0, 77.0, 77.0, 113.0, 75.0, 109.0, 71.0, 77.0, 64.0, 122.0, 93.0, 75.0, 89.0, 155.0, 108.0, 84.0, 83.0, 85.0, 72.0, 74.0, 80.0, 79.0, 75.0, 89.0, 96.0, 73.0, 108.0]
print(sum(r)/len(r))
print(sum(r2)/len(r2))
print(sum(r3)/len(r3))
print(sum(r4)/len(r4))