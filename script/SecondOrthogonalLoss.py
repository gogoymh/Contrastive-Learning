import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SecondOrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
        
    def forward(self, fake, reals):
        
        distance_need_short = self.L1_distance(fake, reals[:,0,:])
        distance_need_long = self.L1_distance(fake, reals[:,1,:])
        
        ratio = distance_need_short/distance_need_long
        #print(ratio)
        return ratio.mean()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,2)
        )
        
    def forward(self,x):
        return self.fc(x)
    
class matcher:
    def __init__(self):
        self.a = None
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
        
    def match(self, z0, x0):
        
        similar_matrix = self._make_matrix(z0, x0)
        
        index = torch.topk(similar_matrix, 2, dim=1, largest=False, sorted=True)[1]
        print(index)
        x1 = x0[index] 
        return z0, x1

colors = {0:'black', 1:'lightcoral', 2:'darkorange', 3:'springgreen', 4:'aqua',
          5:'magenta', 6:'red', 7:'chartreuse', 8:'deepskyblue', 9:'gray',
          10:'gold', 11:'peachpuff', 12:'yellow', 13:'dodgerblue', 14:'crimson',
          15:'slategray', 16:'blueviolet', 17:'fuchsia', 18:'tomato', 19:'limegreen',
          20:'peru', 21:'forestgreen', 22:'deeppink', 23:'olive', 24:'palegreen'}

device = torch.device("cuda:0")

newloss = SecondOrthogonalLoss()
oldloss = nn.L1Loss()

generator1 = Generator().to(device)
optim1 = optim.SGD(generator1.parameters(), lr=0.001)

generator2 = Generator().to(device)
generator2.load_state_dict(generator1.state_dict())
optim2 = optim.SGD(generator2.parameters(), lr=0.001)

pair = matcher()

plt.scatter([1,0,-1], [1,1,1], s=100, c='blue')
log_interval = 1000
for i in range(3000):
    #data = torch.from_numpy(np.random.uniform(-0.5, 0.5, (64,2))).float()
    #data = torch.from_numpy(np.random.normal(0, 1, (64,2))).float()
    data = torch.Tensor([[0.6,0],[0.4,0],[-0.4,0],[-0.6,0]]).float()
    reals = torch.Tensor([[1,1],[0,1],[-1,1]]).float()

    z, x = pair.match(data, reals)
    z = z.to(device)
    x = x.to(device)
    
    optim1.zero_grad()
    fake1 = generator1(z)
    loss1 = newloss(fake1, x)
    loss1.backward()
    optim1.step()
    
    optim2.zero_grad()
    fake2 = generator2(z)
    loss2 = oldloss(fake2, x[:,0,:])
    loss2.backward()
    optim2.step()
    
    
    print((i+1), loss1.item(), loss2.item())
    if i == 0:
        fake1_start = fake1.detach().cpu().numpy()
        fake2_start = fake2.detach().cpu().numpy()
        plt.scatter(fake1_start[0,0], fake1_start[0,1], s=100, c='orange', marker="^")
        plt.scatter(fake1_start[1,0], fake1_start[1,1], s=100, c='orange', marker="^")
        plt.scatter(fake1_start[2,0], fake1_start[2,1], s=100, c='orange', marker="^")
        plt.scatter(fake1_start[3,0], fake1_start[3,1], s=100, c='orange', marker="^")
        plt.scatter(fake2_start[0,0], fake2_start[0,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[1,0], fake2_start[1,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[2,0], fake2_start[2,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[3,0], fake2_start[3,1], s=100, c='green', marker="^")
        continue
    else:
        fake1_finish = fake1.detach().cpu().numpy()
        x = [fake1_start[0,0], fake1_finish[0,0]]
        y = [fake1_start[0,1], fake1_finish[0,1]]
        plt.plot(x, y, c='orange')
        x = [fake1_start[1,0], fake1_finish[1,0]]
        y = [fake1_start[1,1], fake1_finish[1,1]]
        plt.plot(x, y, c='orange')
        x = [fake1_start[2,0], fake1_finish[2,0]]
        y = [fake1_start[2,1], fake1_finish[2,1]]
        plt.plot(x, y, c='orange')
        x = [fake1_start[3,0], fake1_finish[3,0]]
        y = [fake1_start[3,1], fake1_finish[3,1]]
        plt.plot(x, y, c='orange')
        fake1_start = fake1.detach().cpu().numpy()
        
        fake2_finish = fake2.detach().cpu().numpy()
        x = [fake2_start[0,0], fake2_finish[0,0]]
        y = [fake2_start[0,1], fake2_finish[0,1]]
        plt.plot(x, y, c='green')
        x = [fake2_start[1,0], fake2_finish[1,0]]
        y = [fake2_start[1,1], fake2_finish[1,1]]
        plt.plot(x, y, c='green')
        x = [fake2_start[2,0], fake2_finish[2,0]]
        y = [fake2_start[2,1], fake2_finish[2,1]]
        plt.plot(x, y, c='green')
        x = [fake2_start[3,0], fake2_finish[3,0]]
        y = [fake2_start[3,1], fake2_finish[3,1]]
        plt.plot(x, y, c='green')
        fake2_start = fake2.detach().cpu().numpy()
    '''
    if i % log_interval == 0:
        print(loss1.item(), loss2.item())
        plt.scatter([1,-1], [1,1], s=100, c='blue')
        sample = torch.from_numpy(np.random.uniform(-0.5, 0.5, (100,2))).float().to(device)
        #sample = torch.from_numpy(np.random.normal(0, 1, (64,2))).float().to(device)
        sample1 = generator1(sample)
        sample1 = sample1.detach().cpu().numpy()
        sample2 = generator2(sample)
        sample2 = sample2.detach().cpu().numpy()
        plt.scatter(sample1[:,0], sample1[:,1], s = 10, c='orange')
        plt.scatter(sample2[:,0], sample2[:,1], s = 10, c='green')
        plt.xlim(-2,2)
        plt.ylim(-1,3)
        plt.show()
        plt.close()
'''
fake1 = generator1(data.to(device))
fake1 = fake1.detach().cpu().numpy()
plt.scatter(fake1[0,0], fake1[0,1], s=100, c='orange', marker="x")
plt.scatter(fake1[1,0], fake1[1,1], s=100, c='orange', marker="x")
plt.scatter(fake1[2,0], fake1[2,1], s=100, c='orange', marker="x")
plt.scatter(fake1[3,0], fake1[3,1], s=100, c='orange', marker="x")

fake2 = generator2(data.to(device))
fake2 = fake2.detach().cpu().numpy()
plt.scatter(fake2[0,0], fake2[0,1], s=100, c='green', marker="x")
plt.scatter(fake2[1,0], fake2[1,1], s=100, c='green', marker="x")
plt.scatter(fake2[2,0], fake2[2,1], s=100, c='green', marker="x")
plt.scatter(fake2[3,0], fake2[3,1], s=100, c='green', marker="x")
plt.show()
plt.close()












