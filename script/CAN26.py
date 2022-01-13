import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchlars import LARS

'''
class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.labels = torch.zeros(2 * self.batch_size).long().to(self.device)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        #labels = torch.zeros(2 * self.batch_size).long().to(self.device)
        loss = self.criterion(logits, self.labels)

        return loss / (2 * self.batch_size)
'''
class sample_distribution:
    def __init__(self, x_centers=None, y_centers=None, var=0.1, plot=False):
        self.var = var
        
        if x_centers is  None:
            x_centers = [-1,-1,-1,
                         0,0,0,
                         1,1,1,]
        if y_centers is None:
            y_centers = [1,0,-1,
                         1,0,-1,
                         1,0,-1]
        
        self.length = len(x_centers)
        
        point = []
        for i in range(self.length):
            point.append([x_centers[i],y_centers[i]])
        self.point = np.array(point)
        #print(point)
        
        self.colors = {0:'black', 1:'lightcoral', 2:'darkorange', 3:'springgreen', 4:'aqua',
                       5:'magenta', 6:'red', 7:'chartreuse', 8:'deepskyblue', 9:'gray'}
        
        if plot:
            for i in range(9):
                plt.scatter(self.point[i,0], self.point[i,1], s=100, c=self.colors[i])
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.show()
            plt.close()

    def sample(self, num, plot=False):
        index = np.random.choice(self.length, num)
        minibatch = self.point[index]
        noise1 = np.random.normal(0,self.var,(num,2))
        noise2 = np.random.normal(0,self.var,(num,2))
        
        dist1 = minibatch + noise1
        dist2 = minibatch + noise2
        
        if plot:
            for i in range(num):
                plt.scatter(dist1[i,0], dist1[i,1], s=100, c=self.colors[index[i]], marker='x')
                plt.scatter(dist2[i,0], dist2[i,1], s=100, c=self.colors[index[i]], marker='x')
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.show()
            plt.close()
        
        return dist1, dist2
    
    def point_sample(self, plot=False):
        if plot:
            for i in range(9):
                plt.scatter(self.point[i,0], self.point[i,1], s=100, c=self.colors[i])
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
        
        return self.point
    
class Represent(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,2),
            #nn.Sigmoid()
            nn.Softmax(dim=1)
        )
        
    def forward(self,x):
        return self.fc(x)
'''
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,32),
            nn.LeakyReLU(),
            nn.Linear(32,64)
        )
        
    def forward(self,x):
        return self.fc(x)


log = 500
batch_size = 128
device = torch.device("cuda:0")
a = sample_distribution(var=0.1, plot=True)
represent = Represent().to(device)
encoder = Encoder().to(device)
parameters = list(represent.parameters()) + list(encoder.parameters())
optimizer = optim.Adam(parameters, lr=0.0001)
contrastive_loss = NTXentLoss(device, batch_size)

best = 1000
for i in range(100000):
    x1, x2 = a.sample(batch_size)
    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)
    
    optimizer.zero_grad()
    out1 = encoder(represent(x1))
    out2 = encoder(represent(x2))
    
    loss = contrastive_loss(out1, out2)
    loss.backward()
    optimizer.step()
    
    if loss.item() <= best:
        print(i, loss.item())
        represent.eval()
        torch.save({'model_state_dict': represent.state_dict()}, "C://유민형//개인 연구//Constrastive learning//can26.pth")
        x = a.point_sample()
        x = torch.from_numpy(x).float().to(device)
        rep = represent(x)
        rep = rep.detach().cpu().numpy()
        for j in range(9):
            plt.scatter(rep[j,0], rep[j,1], s=100, c=a.colors[j], marker='x')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.close()
        best = loss.item()
        represent.train()
'''
class Matcher:
    def __init__(self):
        super().__init__()
        
        self.similarity = nn.CosineSimilarity(dim=-1)
        
    def _make_matrix(self, a, b):
        matrix = self.similarity(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
    
    def _make_label(self, sample, representation):
        
        similarity_matrix = self._make_matrix(sample, representation)
        #print(similarity_matrix)
        label = similarity_matrix.argmax(dim=1, keepdim=True)
        #print(label)
        return label.squeeze()

class AgreeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.distance = nn.L1Loss()
        
    def equal(self, a, b):
        return (a == b).sum(dim=-1)
    
    def _equal_matrix(self, a, b):
        matrix = self.equal(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
        
    def _make_mask(self, label):
        mask = self._equal_matrix(label.unsqueeze(1), label.unsqueeze(1)).fill_diagonal_(0)
        return mask
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
    
    def forward(self, fake, real, mask, iteration):
        
        desire = self.distance(fake, real)
        loss = desire
        
        if iteration >= 10000:
            gather = self._make_matrix(fake, fake) * mask
            loss = desire + gather.sum()/2
        
        return loss

def dist(batch_size):
    a = np.random.uniform(0,1,(batch_size,1))
    b = 1 - a
    c = np.concatenate((a, b), axis=1)
    return c

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,2),
            nn.Tanh()
        )
        
    def forward(self,x):
        return self.fc(x)

device = torch.device("cuda:0")
real_dist = sample_distribution(var=0.1, plot=True)

represent = Represent()
#checkpoint = torch.load("C://유민형//개인 연구//Constrastive learning//can26.pth")
checkpoint = torch.load("/home/compu/ymh/contrastive/can26.pth")
represent.load_state_dict(checkpoint['model_state_dict'])
represent.to(device)

match = Matcher()

generator = Generator().to(device)
#optimizer = optim.Adam(generator.parameters(), lr=0.0001)
optimizer = LARS(optim.SGD(generator.parameters(), lr=0.1))

contrastive_loss = AgreeLoss() # nn.L1Loss() 

real = real_dist.point_sample()
real = torch.from_numpy(real).float().to(device)
representation = represent(real)
rep = representation.detach().cpu().numpy()
for j in range(9):
    plt.scatter(rep[j,0], rep[j,1], s=100, c=real_dist.colors[j], marker='x')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
plt.close()

log = 500
for i in range(100000):
    optimizer.zero_grad()
    #loss = 0
    #for k in range(64//3):
    sample = dist(12800)
    sample = torch.from_numpy(sample).float().to(device)
    label = match._make_label(sample, representation)
    mask = contrastive_loss._make_mask(label)

    fake = generator(sample)
    loss = contrastive_loss(fake, real[label], mask, i)
    
    loss.backward()
    optimizer.step()
    
    
    if i % 500 == 0:
        print(i, loss.item())
        generator.eval()
        #torch.save({'model_state_dict': represent.state_dict()}, "C://유민형//개인 연구//Constrastive learning//can26.pth")
        _ = real_dist.point_sample(True)
        sample = dist(1000)
        sample = torch.from_numpy(sample).float().to(device)
        label = match._make_label(sample, representation)
        fake = generator(sample)
        fake = fake.detach().cpu().numpy()
        for j in range(sample.shape[0]):
            plt.scatter(fake[j,0], fake[j,1], s=100, c=real_dist.colors[label[j].item()], marker='x')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.savefig("/home/compu/ymh/contrastive/CAN26/plot_%d.png" % i)
        #plt.show()
        plt.close()
        best = loss.item()
        generator.train()
    




























