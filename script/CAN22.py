import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

class sample_distribution:
    def __init__(self, x_centers=None, y_centers=None, var=0.1, plot=False):
        self.var = var
        
        if x_centers is  None:
            x_centers = [-4,-4,-4,-4,-4,
                         -2,-2,-2,-2,-2,
                         0,0,0,0,0,
                         2,2,2,2,2,
                         4,4,4,4,4]
        if y_centers is None:
            y_centers = [4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4]
        if plot:
            plt.scatter(x_centers, y_centers, s=100)
            plt.xlim(-5,5)
            plt.ylim(-5,5)
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
            plt.scatter(dist[:,0], dist[:,1], s=10)
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.show()
            plt.close()
            
        return dist

class rep1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out

class rep2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,2)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out

class matcher:
    def __init__(self, root1, root2, device, smooth=0.1):
        
        self.device = device
        self.smooth = smooth
        
        checkpoint_repnet1 = torch.load(root1)
        self.repnet1 = rep1()
        self.repnet1.load_state_dict(checkpoint_repnet1['model_state_dict'])
        self.repnet1.to(device)

        checkpoint_repnet2 = torch.load(root2)
        self.repnet2 = rep2()
        self.repnet2.load_state_dict(checkpoint_repnet2['model_state_dict'])
        self.repnet2.to(device)
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
        
    def match(self, z0, x0, prior=False, plot=False):
        
        z = torch.from_numpy(z0).float().to(self.device)
        x = torch.from_numpy(x0).float().to(self.device)
        
        embed_x = self.repnet1(x)
        embed_z = self.repnet2(z)
        
        if plot:
            plot_x = embed_x.detach().cpu().numpy()
            plot_z = embed_z.detach().cpu().numpy()
            plt.scatter(plot_x[:,0], plot_x[:,1], c='blue')
            plt.scatter(plot_z[:,0], plot_z[:,1], c='orange')
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
        
        similar_matrix = self._make_matrix(embed_z, embed_x)
        z1 = z[similar_matrix.argmin(dim=0).long()].detach().cpu().numpy()
        
        if prior:
            priority = 1/(similar_matrix.min(dim=0)[0] + self.smooth)
            priority = priority.detach().cpu().numpy()
            return z1, x0, priority
        else:
            return z1, x0

device = torch.device("cuda:0")
batch_size = 10
log_interval = 10

data = sample_distribution()
pair = matcher("C://유민형//개인 연구//Constrastive learning//CAN21_repnet1.pth", "C://유민형//개인 연구//Constrastive learning//CAN21_repnet2.pth", device)

z = np.random.normal(0, 1, (batch_size, 100))
x = data.sample(batch_size)

z, x = pair.match(z, x, plot=True)

'''
class Buffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.z = np.zeros((self.capacity, 100))
        self.x = np.zeros((self.capacity, 2))
        
        self.idx = 0
        self.n_entries = 0
        
    def add(self, z, x, priority=None):
        self.z[self.idx] = z
        self.x[self.idx] = x
        
        self.idx += 1
        
        if self.idx >= self.capacity: # First in, First out
            self.idx = 0            
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def sample(self, batch_size, priority=None):
        sample = np.random.choice(self.n_entries, batch_size, replace=False)        
        return self.z[sample], self.x[sample]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,2)
        )
        
    def forward(self,x):
        return self.fc(x)

device = torch.device("cuda:0")
batch_size = 128
log_interval = 10

data = sample_distribution()
pair = matcher("C://유민형//개인 연구//Constrastive learning//CAN21_repnet1.pth", "C://유민형//개인 연구//Constrastive learning//CAN21_repnet2.pth", device)
buffer = Buffer()
generator = Generator().to(device)
optim = optim.Adam(generator.parameters(), lr=0.0001)
criterion = nn.MSELoss() #nn.L1Loss()

for iteration in range(10000):
    z = np.random.normal(0, 1, (batch_size, 100))
    x = data.sample(batch_size)

    z, x = pair.match(z, x)
    for i in range(batch_size):
        buffer.add(z[i],x[i])
    
    if buffer.n_entries >= 10000:
        generator.train()
        
        z, x = buffer.sample(batch_size)
        z = torch.from_numpy(z).float().to(device)
        x = torch.from_numpy(x).float().to(device)
    
        optim.zero_grad()
        fake = generator(z)
        loss = criterion(fake, x)
        loss.backward()
        optim.step()
        
        if (iteration+1) % log_interval == 0:
            generator.eval()
            print("[Iteration:%d] [Loss:%f]" % ((iteration+1), loss.item()))
            z_sample = torch.from_numpy(np.random.normal(0, 1, (10000, 100))).float().to(device)
            z_rep = generator(z_sample).detach().cpu().numpy()
            x_centers = [-4,-4,-4,-4,-4,
                         -2,-2,-2,-2,-2,
                         0,0,0,0,0,
                         2,2,2,2,2,
                         4,4,4,4,4]
            y_centers = [4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4,
                         4,2,0,-2,-4]
            plt.scatter(x_centers, y_centers, s=100, c='blue')
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.scatter(z_rep[:,0], z_rep[:,1], s=10, c='orange')
            #plt.savefig("/home/compu/ymh/contrastive/CAN21/iter_%d.png" % (iteration+1))
            plt.show()
            plt.close()
            
    else:
        print("Buffer is being warmed up. [%d/10000]" % buffer.n_entries)
'''


















