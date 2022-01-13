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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        
    def forward(self, x, train=True):
        x = x.view(-1, 32*32)
        
        out = self.fc1(x)
        
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

#generator = Generator().to(device)
discriminator = Discriminator().to(device)

#optim_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

mle_loss = nn.MSELoss()
adv_loss = nn.BCELoss()

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
                batch_size=batch_size, shuffle=True)#, pin_memory=True)

sample = torch.from_numpy(np.random.normal(0,1,(25, 100))).float().to(device)

ones = torch.ones((batch_size, 1)).float().to(device)
zeros = torch.zeros((batch_size, 1)).float().to(device)
'''
for i in range(100000):
    z = torch.from_numpy(np.random.normal(0,1,(batch_size*4, 100))).float().to(device)
    x, _ = real_loader.__iter__().next()
    x = x.float().to(device)

    index, unique = matcher.match(generator(z).view(-1,1024), x.view(-1,1024))
    z = z[index].detach()
    
    optim_G.zero_grad()
    generator.train()
    
    loss_G = mle_loss(generator(z), x)
    loss_G.backward()
    optim_G.step()  
        
    if (i+1) % 1000 == 0:
        print(i+1, loss_G.item(), unique)
        
        generator.eval()
        sample_plot = generator(sample)
        save_image(sample_plot.data, "C://유민형//개인 연구//Constrastive learning//sample6//%d.png" % (i+1), nrow=5, normalize=True)
        #plt.show()
        #plt.close()
        generator.train()

torch.save({'model_state_dict': generator.state_dict()}, "C://유민형//개인 연구//Constrastive learning//generator.pth")
'''

checkpoint_generator = torch.load("C://유민형//개인 연구//Constrastive learning//generator.pth")
generator = Generator()
generator.load_state_dict(checkpoint_generator['model_state_dict'])
#generator = nn.DataParallel(generator)
generator.to(device)

optim_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))


sample_num = 1
for i in range(sample_num):
    sample = torch.from_numpy(np.random.normal(0,1,(25, 100))).float().to(device)
    generator.eval()
    sample_plot = generator(sample)
    save_image(sample_plot.data, "C://유민형//개인 연구//Constrastive learning//sample7//%d.png" % (i+1), nrow=5, normalize=True)

'''
for i in range(10000):
    ## ---- D ---- ##
    z = torch.from_numpy(np.random.normal(0,1,(batch_size, 100))).float().to(device)
    x, _ = real_loader.__iter__().next()
    x = x.float().to(device)
    
    optim_D.zero_grad()
    generator.eval()
    discriminator.train()
    
    fake = generator(z)
    
    real_loss = adv_loss(discriminator(x), ones)
    fake_loss = adv_loss(discriminator(fake.detach()), zeros)
            
    loss_D = (real_loss + fake_loss) / 2
    loss_D.backward()
    optim_D.step()
    
    if (i+1) % 10 == 0:
        generator.eval()
        discriminator.eval()
        correct = 0
        ## ---- Test ---- ##
        for j in range(10):
            z = torch.from_numpy(np.random.normal(0,1,(100, 100))).float().to(device)
            output = discriminator(generator(z))
            pred = (output < 0.5).view(-1).sum()
            correct += pred.item()
        
        accuracy = correct / 1000
        print(i+1 , accuracy)

'''
for i in range(1000):
    for j in range(1000):
        ## ---- D ---- ##
        z = torch.from_numpy(np.random.normal(0,1,(batch_size, 100))).float().to(device)
        x, _ = real_loader.__iter__().next()
        x = x.float().to(device)
        
        optim_D.zero_grad()
        generator.eval()
        discriminator.train()
    
        fake = generator(z)
        
        real_loss = adv_loss(discriminator(x), ones)
        fake_loss = adv_loss(discriminator(fake.detach()), zeros)
            
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optim_D.step()
        #print(loss_D.item())
    '''
    if loss_D.item() == 13.815510749816895:
        torch.save({'model_state_dict': discriminator.state_dict()}, "C://유민형//개인 연구//Constrastive learning//discriminator.pth")
        print("discriminator saved")
    '''
    for k in range(100):
        ## ---- G ---- ##
        z = torch.from_numpy(np.random.normal(0,1,(batch_size, 100))).float().to(device)
            
        optim_G.zero_grad()
        discriminator.eval()
        generator.train()
            
        fake = generator(z)
        
        gan_loss = adv_loss(discriminator(fake), ones)
        gan_loss.backward()
        optim_G.step()   
        
    if (i+1) % 1 == 0:
        print(i+1+sample_num, loss_D.item(), gan_loss.item())
        
        generator.eval()
        sample_plot = generator(sample)
        save_image(sample_plot.data, "C://유민형//개인 연구//Constrastive learning//sample7//%d.png" % (i+1+sample_num), nrow=5, normalize=True)
        #plt.show()
        #plt.close()
        generator.train()

'''
checkpoint_discriminator = torch.load("C://유민형//개인 연구//Constrastive learning//discriminator.pth")
discriminator = Discriminator()
discriminator.load_state_dict(checkpoint_discriminator['model_state_dict'])
#generator = nn.DataParallel(generator)
discriminator.to(device)

optim_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

z = torch.from_numpy(np.random.normal(0,1,(batch_size, 100))).float().to(device)
x, _ = real_loader.__iter__().next()
x = x.float().to(device)
        
generator.eval()
discriminator.eval()
    
fake = generator(z)
        
real_prob = discriminator(x)
fake_prob = discriminator(fake.detach())

print(real_prob)
print(fake_prob)

real_loss = adv_loss(discriminator(x), ones)
fake_loss = adv_loss(discriminator(fake.detach()), zeros)

print(real_loss.item())
print(fake_loss.item())
'''

















