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
            return self._make_matrix

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)
    
    def L2_distance(self, a, b):
        return ((a-b)**2).mean(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L2_distance(a.unsqueeze(1), b.unsqueeze(0))
        #matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
    
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
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 32, 32)
        return img

class Represent(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(32*32,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,32)
        )
        
    def forward(self, x, train=True):
        x = x.view(-1, 32*32)
        
        out = self.fc1(x)

        if train:
            return self.fc2(out)
        else:
            return out

class Represent2(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1 = nn.Conv2d(1, 6, 5) # 0
    self.conv3 = nn.Conv2d(6, 16, 5) # 2
    self.conv5 = nn.Conv2d(16, 120, 5) # 4
    
    self.fc6 = nn.Linear(120, 64) # 6
    self.fc7 = nn.Linear(64, 32) # 8
    
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(2)
    
  def forward(self, x, train=True):
      
      out = self.maxpool(self.relu(self.conv1(x)))
      out = self.maxpool(self.relu(self.conv3(out)))
      out = self.relu(self.conv5(out))
      out = out.view(-1, 120)
      
      if train:
          out = self.relu(self.fc6(out))
          out = self.fc7(out)
          return out
      else:
          return out

class Matching:
    def __init__(self):
        super().__init__()
        
    def L2_distance(self, a, b):
        return ((a-b)**2).mean(dim=-1)
    
    def L1_distance(self, a, b):
        return torch.abs(a - b).mean(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L2_distance(a.unsqueeze(1), b.unsqueeze(0))
        #matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix

    def match(self, rep_gz, rep_x):
        distance = self._make_matrix(rep_gz, rep_x)
        #print(distance)
        index = distance.argmin(dim=0).long()
        #print(index.unique().shape[0])
        return index, index.unique().shape[0]
'''
device = torch.device("cuda:0")
represent = Represent().to(device)
generator = Generator().to(device)
matcher = Matching()

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
                batch_size=3, shuffle=True)#, pin_memory=True)

z = torch.from_numpy(np.random.normal(0,1,(10, 100))).float().to(device)
x, _ = real_loader.__iter__().next()
x = x.float().to(device)

rep_z = represent(generator(z), False)
rep_x = represent(x, False)

index = matcher.match(rep_z, rep_x)

z = z[index]
print(z)
optim_G.zero_grad()
rep_z = represent(generator(z), False)
rep_x = represent(x, False)

loss_G = generate_loss(rep_z, rep_x)
loss_G.backward()
optim_G.step()

'''
batch_size = 128
device = torch.device("cuda:0")

represent = Represent2().to(device)
generator = Generator().to(device)

optim_G = optim.Adam(generator.parameters(), lr=0.0001)
optim_R = optim.Adam(represent.parameters(), lr=0.0001)

contrastive_loss = NTXentLoss(device, batch_size)
generate_loss = nn.MSELoss()# nn.L1Loss() ## #

matcher = Matching()

augment_loader = DataLoader(aug_mnist, batch_size=batch_size, shuffle=True)#, pin_memory=True)
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
                batch_size=batch_size, shuffle=True)#, pin_memory=True)

sample = torch.from_numpy(np.random.normal(0,1,(25, 100))).float().to(device)

for i in range(200000):
    '''
    ## ---- R ---- ##
    represent.train()
    generator.eval()
    
    optim_R.zero_grad()
    
    x1, x2 = augment_loader.__iter__().next()
    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    
    rep1 = represent(x1)
    rep2 = represent(x2)
    
    loss_R = contrastive_loss(rep1, rep2)
    loss_R.backward()
    optim_R.step()
    '''
    ## ---- G ---- ##
    represent.eval()
    generator.eval()
    
    z = torch.from_numpy(np.random.normal(0,1,(batch_size*4, 100))).float().to(device)
    x, _ = real_loader.__iter__().next()
    x = x.float().to(device)

    #rep_z = represent(generator(z), False)
    #rep_x = represent(x, False)

    #index, unique = matcher.match(rep_z, rep_x)
    index, unique = matcher.match(generator(z).view(-1,1024), x.view(-1,1024))
    z = z[index].detach()
    
    optim_G.zero_grad()
    generator.train()
    
    #rep_z = represent(generator(z))#, False)
    #rep_x = represent(x)#, False)
    
    loss_G = generate_loss(generator(z), x)
    #loss_G = generate_loss(rep_z,rep_x)
    #loss_G = contrastive_loss(generator(z).view(-1,1024), x.view(-1,1024))
    loss_G.backward()
    optim_G.step()
    
    if i % 100 == 0:
        #print(i, loss_R.item(), loss_G.item(), unique)
        print(i, loss_G.item(), unique)
        generator.eval()
        sample_plot = generator(sample)
        save_image(sample_plot.data, "C://유민형//개인 연구//Constrastive learning//sample5//%d.png" % i, nrow=5, normalize=True)
        #plt.show()
        #plt.close()
        generator.train()
    
























