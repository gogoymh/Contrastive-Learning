import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.utils import save_image
#from torchlars import LARS

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
'''
class Augmented_MNIST(Dataset):
    def __init__(self, root, aug_transform):
        super().__init__()

        self.aug_transform = aug_transform
        
        save_file = os.path.join(root, 'Augmented_MNIST.npy')
        
        if os.path.isfile(save_file):
            self.mnist = np.load(save_file)
            print("File is loaded.")
        
        else:
            train_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor()])
                        ),
                batch_size=1, shuffle=False)
            
            self.mnist = np.empty((60000,28,28,1))
            
            for idx, (x, _) in enumerate(train_loader):
                x = x*255
                x = x.numpy().reshape(28,28,1)
                self.mnist[idx] = x
                print("[%d/60000]" % (idx+1))
                
            save_file = os.path.join(root, 'Augmented_MNIST')
            np.save(save_file, self.mnist)
                
        self.len = 60000
        self.mnist = self.mnist.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.mnist[index]
                   
        aug1 = self.aug_transform(img)
        aug2 = self.aug_transform(img)
        
        return aug1, aug2
    
    def __len__(self):
        return self.len
'''
class Represent(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(32*32,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,2),
            #nn.Sigmoid()
            nn.Softmax(dim=1)
        )
        
    def forward(self,x):
        x = x.view(-1, 32*32)
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

colors = {0:'black', 1:'lightcoral', 2:'darkorange', 3:'springgreen', 4:'aqua',
          5:'magenta', 6:'red', 7:'chartreuse', 8:'deepskyblue', 9:'gray'}

log = 500
batch_size = 128
device = torch.device("cuda:0")
represent = Represent().to(device)
encoder = Encoder().to(device)
parameters = list(represent.parameters()) + list(encoder.parameters())
optimizer = optim.Adam(parameters, lr=0.0001)
contrastive_loss = NTXentLoss(device, batch_size)

real_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
                        ),
                batch_size=100, shuffle=True)#, pin_memory=True)
sample0, _ = real_loader.dataset.__getitem__(0)
sample1, _ = real_loader.dataset.__getitem__(1)
sample2, _ = real_loader.dataset.__getitem__(2)
sample3, _ = real_loader.dataset.__getitem__(3)
sample4, _ = real_loader.dataset.__getitem__(4)
sample5, _ = real_loader.dataset.__getitem__(5)
sample6, _ = real_loader.dataset.__getitem__(7)
sample7, _ = real_loader.dataset.__getitem__(13)
sample8, _ = real_loader.dataset.__getitem__(15)
sample9, _ = real_loader.dataset.__getitem__(17)
sample = torch.cat((sample0, sample1, sample2, sample3, sample4,
                    sample5, sample6, sample7, sample8, sample9), dim=0)
sample = sample.float().to(device)

color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform_pretrain = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize(32),
         transforms.RandomCrop(32),
         transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
         transforms.RandomApply([color_jitter], p=0.8),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)) #transforms.Normalize((mean,mean,mean), (std,std,std)) # 
     ])
aug_mnist = Augmented_MNIST("C://유민형//개인 연구//Constrastive learning//", transform_pretrain)
augment_loader = DataLoader(aug_mnist, batch_size=batch_size, shuffle=True)#, pin_memory=True)

best = 1000
for i in range(100000):
    x1, x2 = augment_loader.__iter__().next()
    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    
    optimizer.zero_grad()
    out1 = encoder(represent(x1))
    out2 = encoder(represent(x2))
    
    loss = contrastive_loss(out1, out2)
    loss.backward()
    optimizer.step()
    
    if loss.item() <= best:
        print(i, loss.item())
        represent.eval()
        torch.save({'model_state_dict': represent.state_dict()}, "C://유민형//개인 연구//Constrastive learning//can28_2.pth")
        
        #a = 20
        #while a:
        #    sample2, y = real_loader.__iter__().next()
        #    if y.item() == 0:
        #        rep = represent(sample2.float().to(device))
        #        rep = rep.detach().cpu().numpy()
        #        plt.scatter(rep[0,0], rep[0,1], s=10, c=colors[0], marker='x')                
        #        a -= 1
        sample2, y = real_loader.__iter__().next()
        rep = represent(sample2.float().to(device))
        rep = rep.detach().cpu().numpy()
        for j in range(100):
            plt.scatter(rep[j,0], rep[j,1], s=100, c=colors[y[j].item()], marker='x')
        plt.xlim(-0.25,1.25)
        plt.ylim(-0.25,1.25)
        plt.show()
        plt.close()
        best = loss.item()
        represent.train()
'''
real_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
                        ),
                batch_size=64, shuffle=True)#, pin_memory=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 32, 32)
        return img

def dist(batch_size):
    a = np.random.uniform(0,1,(batch_size,1))
    b = 1 - a
    c = np.concatenate((a, b), axis=1)
    return c

device = torch.device("cuda:0")
real_dist = sample_distribution(var=0.1, plot=True)

represent = Represent()
checkpoint = torch.load("C://유민형//개인 연구//Constrastive learning//can28_2.pth")
#checkpoint = torch.load("/home/compu/ymh/contrastive/can26.pth")
represent.load_state_dict(checkpoint['model_state_dict'])
represent.to(device)

generator = Generator().to(device)
optimizer = optim.Adam(generator.parameters(), lr=0.0001)
#optimizer = LARS(optim.SGD(generator.parameters(), lr=0.1))

contrastive_loss = nn.L1Loss() 

sample = dist(25)
sample = torch.from_numpy(sample).float().to(device)

for i in range(1000000):
    optimizer.zero_grad()
    
    real, _ = real_loader.__iter__().next()
    real = real.float().to(device)
    representation = represent(real)

    fake = generator(representation.detach())
    loss = contrastive_loss(fake, real)
    
    loss.backward()
    optimizer.step()
    
    
    if i % 1000 == 0:
        print(i, loss.item())
        generator.eval()
        sample_plot = generator(sample)
        save_image(sample_plot.data, "C://유민형//개인 연구//Constrastive learning//sample//%d.png" % i, nrow=5, normalize=True)
        #plt.show()
        #plt.close()
        generator.train()
    




























