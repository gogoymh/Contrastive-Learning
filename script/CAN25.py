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
    
    def point_sample(self):
        return self.point

class rep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,2)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out

class matcher:
    def __init__(self, root1, device, threshold=0.1):
        
        self.device = device
        self.threshold = threshold
        
        checkpoint_repnet1 = torch.load(root1)
        self.repnet1 = rep()
        self.repnet1.load_state_dict(checkpoint_repnet1['model_state_dict'])
        self.repnet1.to(device)
        self.repnet1.eval()
        
        self.colors = {0:'black', 1:'lightcoral', 2:'darkorange', 3:'springgreen', 4:'aqua',
                       5:'magenta', 6:'red', 7:'chartreuse', 8:'deepskyblue', 9:'gray',
                       10:'gold', 11:'peachpuff', 12:'yellow', 13:'dodgerblue', 14:'crimson',
                       15:'slategray', 16:'blueviolet', 17:'fuchsia', 18:'tomato', 19:'limegreen',
                       20:'peru', 21:'forestgreen', 22:'deeppink', 23:'olive', 24:'palegreen'}
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
        
    def match(self, z0, x0, plot=False):
        
        z = torch.from_numpy(z0).float().to(self.device)
        x = torch.from_numpy(x0).float().to(self.device)
        
        embed_x = self.repnet1(x)
        similar_matrix = self._make_matrix(z, embed_x)
        
        index = torch.topk(similar_matrix, 2, dim=1, largest=False, sorted=True)[1]
        x1 = x[index].detach().cpu().numpy()

        if plot:
            plot_x = embed_x.detach().cpu().numpy()
            
            for i in range(25):
                plt.scatter(plot_x[i,0], plot_x[i,1], c=self.colors[i])
                plt.xlim(-1.5,1.5)
                plt.ylim(-1.5,1.5)
            
            #index = index.detach().cpu().numpy()
            #for i in range(len(z0)):
            #    plt.scatter(z0[i,0], z0[i,1], c=self.colors[index[i]])
            plt.show()
            plt.close()
        
        return z0, x1
'''
device = torch.device("cuda:0")
batch_size = 5
log_interval = 10

data = sample_distribution()
pair = matcher("C://유민형//개인 연구//Constrastive learning//CAN23_repnet1.pth", device)

z = np.random.uniform(-1,1, (batch_size, 2))
x = data.point_sample()

z, x = pair.match(z, x, plot=True)

'''
class Buffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.z = np.zeros((self.capacity, 2))
        self.x = np.zeros((self.capacity, 2, 2))
        
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
            nn.Linear(2,1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000,1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000,2)
        )
        
    def forward(self,x):
        return self.fc(x)

class SecondOrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
        
    def forward(self, fake, reals):
        
        distance_need_short = self.L1_distance(fake, reals[:,0,:])
        distance_need_long = self.L1_distance(fake, reals[:,1,:])
        
        ratio = distance_need_short/distance_need_long
        
        return ratio.mean()

device = torch.device("cuda:0")
batch_size = 128
log_interval = 500

data = sample_distribution()
pair = matcher("C://유민형//개인 연구//Constrastive learning//CAN23_repnet1.pth", device)
buffer = Buffer()
generator = Generator().to(device)
optim = optim.Adam(generator.parameters(), lr=0.0005)
criterion = SecondOrthogonalLoss() # nn.MSELoss()# nn.L1Loss() #

for iteration in range(100000):
    z = np.random.uniform(-1,1, (batch_size, 2))
    #x = data.sample(batch_size)
    x = data.point_sample()

    z, x = pair.match(z, x)
    for i in range(len(z)):
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
            z_sample = torch.from_numpy(np.random.uniform(-1,1, (10000, 2))).float().to(device)
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
            #plt.xlim(-5,5)
            #plt.ylim(-5,5)
            plt.scatter(z_rep[:,0], z_rep[:,1], s=10, c='orange')
            #plt.savefig("/home/compu/ymh/contrastive/CAN21/iter_%d.png" % (iteration+1))
            plt.show()
            plt.close()
            
    else:
        print("Buffer is being warmed up. [%d/10000]" % buffer.n_entries)



















