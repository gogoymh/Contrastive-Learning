import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Network(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        #return self.tanh(out)*10
        return out

def plot(points, title, example):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    for i in range(example):
        plt.plot([0, points[i,0]], [0, points[i,1]])
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()

device = torch.device("cuda:0")
latent_dim = 100
batch_size = 64
interval = 1
example = 10

net = Network(latent_dim).to(device)
optim = optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))

#contrastive = IndividualLoss(device, batch_size)
contrastive = NTXentLoss(device, batch_size)
z0 = torch.from_numpy(np.random.normal(0, 1, (example, latent_dim))).float().to(device)

for i in range(1000):
    net.train()
    z = np.random.normal(0, 1, (batch_size, latent_dim))
    z1 = z + np.random.normal(0, 0.01, (batch_size, latent_dim))
    z2 = z + np.random.normal(0, 0.01, (batch_size, latent_dim))
    z1 = torch.from_numpy(z1).float().to(device)
    z2 = torch.from_numpy(z2).float().to(device)
    optim.zero_grad()
    embed1 = net(z1)
    embed2 = net(z2)
    #embed = net(z0)
    loss = contrastive(embed1, embed2)
    loss.backward()
    optim.step()
    
    if (i+1) % interval == 0:
        print("[Iteration:%d] [Loss:%f]" % ((i+1), loss.item()))
        net.eval()
        #ex = net(z0)
        z = torch.from_numpy(z).float().to(device)
        ex = net(z)
        plot(ex.detach().cpu().numpy(), "Iteration %d" % (i+1), batch_size)
        #plot(embed.detach().cpu().numpy(), "Iteration %d" % (i+1), batch_size)










