import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class sample_distribution:
    def __init__(self, x_centers=None, y_centers=None, var=0.1, plot=False):
        self.var = var
        
        if x_centers is  None:
            x_centers = [-1,-1,-1,-1,-1,
                         0,0,0,0,0,
                         1,1,1,1,1]
        if y_centers is None:
            y_centers = [1,0,-1,
                         1,0,-1,
                         1,0,-1,
                         1,0,-1,
                         1,0,-1]
        if plot:
            plt.scatter(x_centers, y_centers, s=100)
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.show()
            plt.close()
        
        self.length = len(x_centers)
        
        point = []
        for i in range(self.length):
            point.append([x_centers[i],y_centers[i]])
        self.point = np.array(point)
        #print(point)

    def sample(self, num, plot=False):
        minibatch = self.point[np.random.choice(self.length, num)]
        noise = np.random.normal(0,self.var,(num,2))
        
        dist = minibatch + noise
        
        if plot:
            plt.scatter(dist[:,0], dist[:,1], s=100)
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.show()
            plt.close()
            
        return dist
    
    def point_sample(self):
        plt.scatter(self.point[:,0], self.point[:,1], s=100)
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        
        return self.point

class ContrastiveSupervisionLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        
        self.alpha = alpha
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
        
    def equal(self, a, b):
        return (a == b).sum(dim=-1)
    
    def _equal_matrix(self, a, b):
        matrix = self.equal(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
    
    def _prepare_data_mask(self, z0, x0):
        
        distance_matrix = self._make_matrix(z0, x0)
        index = distance_matrix.argmin(dim=1).long()
        
        fake_len = z0.shape[0]
        x_len = x0.shape[0]#torch.unique(index).shape[0]

        mask = torch.zeros((fake_len+x_len, fake_len+x_len))
        
        position = distance_matrix == distance_matrix.min(dim=1, keepdim=True)[0]
        #position = position[:,torch.unique(index)]
        
        mask[:fake_len, fake_len:] = position        
        mask[fake_len:, :fake_len] = position.transpose(0,1)
        
        index = index.unsqueeze(1)
        equal_matrix = self._equal_matrix(index, index) - torch.eye(fake_len)
        
        mask[:fake_len,:fake_len] = equal_matrix * self.alpha
        
        pos_mask = mask
        #neg_mask = 1 - mask - torch.eye(fake_len+x_len)
        
        return pos_mask #, neg_mask
    
    def forward(self, fake, x, pos_mask):#, neg_mask):
        representation = torch.cat((fake, x), dim=0)
        
        distance_matrix = self._make_matrix(representation, representation)
        
        pos = distance_matrix * pos_mask
        #neg = distance_matrix * neg_mask
        
        ratio = pos.sum()#/neg.sum()
        
        return ratio
'''
newloss = ContrastiveSupervisionLoss()
data = torch.Tensor([[0.6,0],[0.4,0],[-0.4,0],[-0.6,0]]).float()
reals = torch.Tensor([[1,1],[0,1],[-1,1]]).float()

z, x, pos_mask, neg_mask = newloss._prepare_data_mask(data, reals)

'''
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.LeakyReLU(),
            nn.Linear(1000,2),
            nn.Tanh()
        )
        
    def forward(self,x):
        return self.fc(x)
    
colors = {0:'black', 1:'lightcoral', 2:'darkorange', 3:'springgreen', 4:'aqua',
          5:'magenta', 6:'red', 7:'chartreuse', 8:'deepskyblue', 9:'gray',
          10:'gold', 11:'peachpuff', 12:'yellow', 13:'dodgerblue', 14:'crimson',
          15:'slategray', 16:'blueviolet', 17:'fuchsia', 18:'tomato', 19:'limegreen',
          20:'peru', 21:'forestgreen', 22:'deeppink', 23:'olive', 24:'palegreen'}

device = torch.device("cuda:0")

newloss = ContrastiveSupervisionLoss(1)
oldloss = nn.L1Loss()

generator1 = Generator().to(device)
optim1 = optim.SGD(generator1.parameters(), lr=0.001)

generator2 = Generator().to(device)
generator2.load_state_dict(generator1.state_dict())
optim2 = optim.SGD(generator2.parameters(), lr=0.001)

dist = sample_distribution()

#plt.scatter([1,0,-1], [1,1,1], s=100, c='blue')
#plt.xlim(-2,2)
#plt.ylim(0,1.5)
log_interval = 500
batch_size = 3
for i in range(1000000):
    data = torch.from_numpy(np.random.uniform(-1.25, 1.25, (batch_size,2))).float()
    #data = torch.from_numpy(np.random.normal(0, 1, (64,2))).float()
    #data = torch.Tensor([[0.6,0],[0.4,0],[-0.4,0],[-0.6,0]]).float()
    #reals = torch.Tensor([[1,1],[0,1],[-1,1]]).float()
    reals = dist.point_sample()
    reals = torch.from_numpy(reals).float()

    #z, x, pos_mask, neg_mask = newloss._prepare_data_mask(data, reals)
    pos_mask = newloss._prepare_data_mask(data, reals)
    pos_mask = pos_mask.long().to(device)
    #neg_mask = neg_mask.long().to(device)
    
    optim1.zero_grad()
    fake1 = generator1(data.to(device))
    #loss1 = newloss(fake1, x, pos_mask, neg_mask)
    loss1 = newloss(fake1, reals.to(device), pos_mask)
    loss1.backward()
    optim1.step()
    '''
    optim2.zero_grad()
    fake2 = generator2(z)
    loss2 = oldloss(fake2, x)
    loss2.backward()
    optim2.step()
    '''
    
    '''
    print((i+1), loss1.item())#, loss2.item())
    if i == 0:
        fake1_start = fake1.detach().cpu().numpy()
        #fake2_start = fake2.detach().cpu().numpy()
        plt.scatter(fake1_start[0,0], fake1_start[0,1], s=100, c=colors[0], marker="^")
        plt.scatter(fake1_start[1,0], fake1_start[1,1], s=100, c=colors[1], marker="^")
        plt.scatter(fake1_start[2,0], fake1_start[2,1], s=100, c=colors[2], marker="^")
        plt.scatter(fake1_start[3,0], fake1_start[3,1], s=100, c=colors[3], marker="^")
        
        plt.scatter(fake2_start[0,0], fake2_start[0,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[1,0], fake2_start[1,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[2,0], fake2_start[2,1], s=100, c='green', marker="^")
        plt.scatter(fake2_start[3,0], fake2_start[3,1], s=100, c='green', marker="^")
        
        continue
    else:
        fake1_finish = fake1.detach().cpu().numpy()
        x = [fake1_start[0,0], fake1_finish[0,0]]
        y = [fake1_start[0,1], fake1_finish[0,1]]
        plt.plot(x, y, c=colors[0])
        x = [fake1_start[1,0], fake1_finish[1,0]]
        y = [fake1_start[1,1], fake1_finish[1,1]]
        plt.plot(x, y, c=colors[1])
        x = [fake1_start[2,0], fake1_finish[2,0]]
        y = [fake1_start[2,1], fake1_finish[2,1]]
        plt.plot(x, y, c=colors[2])
        x = [fake1_start[3,0], fake1_finish[3,0]]
        y = [fake1_start[3,1], fake1_finish[3,1]]
        plt.plot(x, y, c=colors[3])
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
        print((i+1), loss1.item())#, loss2.item())
        #plt.scatter([1,0,-1], [1,1,1], s=100, c='blue')
        #plt.xlim(-2,2)
        #plt.ylim(0,1.5)
        sample = torch.from_numpy(np.random.uniform(-1.25, 1.25, (10000,2))).float().to(device)
        #sample = torch.from_numpy(np.random.normal(0, 1, (64,2))).float().to(device)
        #sample = torch.Tensor([[0.6,0],[0.4,0],[-0.4,0],[-0.6,0]]).float().to(device)
        sample1 = generator1(sample)
        sample1 = sample1.detach().cpu().numpy()
        #sample2 = generator2(sample)
        #sample2 = sample2.detach().cpu().numpy()
        plt.scatter(sample1[:,0], sample1[:,1], s = 10, c='orange')
        #plt.scatter(sample2[:,0], sample2[:,1], s = 10, c='green')
        #plt.xlim(-2,2)
        #plt.ylim(-1,1.5)
        plt.show()
        plt.close()
    '''
plt.show()
plt.close()

fake1 = generator1(sample.to(device))
fake1 = fake1.detach().cpu().numpy()
plt.scatter([1,0,-1], [1,1,1], s=100, c='blue')
#plt.xlim(-2,2)
#plt.ylim(0,1.5)
plt.scatter(fake1[0,0], fake1[0,1], s=10, c='orange')#, marker="x")
plt.scatter(fake1[1,0], fake1[1,1], s=10, c='orange')#, marker="x")
plt.scatter(fake1[2,0], fake1[2,1], s=10, c='orange')#, marker="x")
plt.scatter(fake1[3,0], fake1[3,1], s=10, c='orange')#, marker="x")

fake2 = generator2(data.to(device))
fake2 = fake2.detach().cpu().numpy()
plt.scatter(fake2[0,0], fake2[0,1], s=100, c='green', marker="x")
plt.scatter(fake2[1,0], fake2[1,1], s=100, c='green', marker="x")
plt.scatter(fake2[2,0], fake2[2,1], s=100, c='green', marker="x")
plt.scatter(fake2[3,0], fake2[3,1], s=100, c='green', marker="x")
plt.show()
plt.close()
'''











