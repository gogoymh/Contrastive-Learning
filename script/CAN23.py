import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchlars import LARS

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

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,32)
        self.fc2 = nn.Linear(32,64)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

device = torch.device("cuda:0")
batch_size = 1792
log_interval = 500

repnet1 = rep().to(device)
encode1 = encoder().to(device)

param1 = list(repnet1.parameters()) + list(encode1.parameters())

optim1 = LARS(optim.SGD(param1, lr=0.1))

#optim1 = optim.Adam(param1, lr=0.0001)

contrastive_loss = NTXentLoss(device, batch_size)

data = sample_distribution()

for iteration in range(10000000):
    repnet1.train()
    
    ## ---- real dist ---- ##
    optim1.zero_grad()
    
    x = data.sample(batch_size)
    x1 = x + np.random.normal(0,0.01,(batch_size,2))
    x2 = x + np.random.normal(0,0.01,(batch_size,2))
    
    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)

    out_real1 = encode1(repnet1(x1))
    out_real2 = encode1(repnet1(x2))

    real_loss = contrastive_loss(out_real1, out_real2)
    real_loss.backward()
    optim1.step()
    
    if (iteration+1) % log_interval == 0:
        repnet1.eval()
        print("[Iteration:%d]" % (iteration+1), end=" ")
        print("[Real contrastive loss:%f]" % real_loss.item())
        
        torch.save({'model_state_dict': repnet1.state_dict()}, "/home/compu/ymh/contrastive/CAN23_repnet1.pth")# "C://유민형//개인 연구//Constrastive learning//CAN23_repnet1.pth")
        
        x_sample = torch.from_numpy(data.sample(1000)).float().to(device)
        x_rep = repnet1(x_sample).detach().cpu().numpy()
        plt.scatter(x_rep[:,0], x_rep[:,1], s=10, c='orange')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.savefig("/home/compu/ymh/contrastive/CAN23/iter_%d_xrep.png" % (iteration+1))
        plt.show()
        plt.close()

        











