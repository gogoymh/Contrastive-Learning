import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
#import random
import matplotlib.pyplot as plt
from network import resnet56

########################################################################################################################
latent_dim = 100
img_shape = (1, 32, 32)
#data_path = "C://유민형//개인 연구//Constrastive learning//"
data_path = "/home/compu/ymh/contrastive/"
#data_path = "/DATA/ymh/contrastive/"
batch_size = 2
lr = 5e-4
weight_decay = 10e-6
out_dim = 64
mean = 0.1307
std = 0.3081

print("[Batch size:%d] [Lr:%f] [Out dim:%d]" % (batch_size, lr, out_dim))
########################################################################################################################
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
            *block(latent_dim, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 1024, normalize=False),
            nn.Linear(1024, int(np.prod(img_shape)))
            #nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), int(np.prod(img_shape))),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_dim)
            )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

########################################################################################################################
class Real_MNIST(Dataset):
    def __init__(self, root, real_transform=None):
        super().__init__()
        
        self.real_transform = real_transform
        
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
        
        choice = np.random.choice(self.len,2)
        
        real1 = self.mnist[choice[0]]
        real2 = self.mnist[choice[1]]
        
        if self.real_transform is not None:
            real1 = self.real_transform(real1)
            real2 = self.real_transform(real2)
            
        else:  
            real1 = real1.transpose(2,0,1)
            real2 = real2.transpose(2,0,1)
        
        return real1, real2
    
    def __len__(self):
        return self.len
    
class Fake_MNIST(Dataset):
    def __init__(self, fake_transform=None):
        super().__init__()
        
        self.fake_transform = fake_transform
        
        self.len = 60000
        self.noise = np.random.choice(255,(60000,28,28,1)).astype('uint8')
        
    def __getitem__(self, index):
        
        choice = np.random.choice(self.len,2)
        
        fake1 = self.noise[choice[0]]
        fake2 = self.noise[choice[1]]
        
        if self.fake_transform is not None:
            fake1 = self.fake_transform(fake1)
            fake2 = self.fake_transform(fake2)
            
        else:  
            fake1 = fake1.transpose(2,0,1)
            fake2 = fake2.transpose(2,0,1)
        
        return fake1, fake2
    
    def __len__(self):
        return self.len

########################################################################################################################
class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.labels = torch.zeros(2 * self.batch_size).long().to(self.device)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
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

########################################################################################################################
transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

real_mnist = Real_MNIST(data_path, real_transform=transform1)
fake_mnist = Fake_MNIST(fake_transform=transform2)

########################################################################################################################
real_loader = DataLoader(dataset=real_mnist, batch_size=1, shuffle=True, pin_memory=True)
fake_loader = DataLoader(dataset=fake_mnist, batch_size=(batch_size-1), shuffle=True, pin_memory=True)
mnist_loader = DataLoader(
                datasets.MNIST(
                        "../data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (mean,), (std,))]
                                ),
                        ),
                batch_size=1, shuffle=True, pin_memory=True)

device = torch.device("cuda:0")
generator = Generator().to(device)
encoder = resnet56().to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_E = optim.Adam(encoder.parameters(), lr=lr)

contrastive = NTXentLoss(device, batch_size)
generate_loss = nn.L1Loss()

z_sample = torch.from_numpy(np.random.normal(0, 1, (6, latent_dim))).float().to(device)
generator.eval()
gen_sample = generator(z_sample).detach().cpu()

for idx_sample in range(6):
    img_sample = gen_sample[idx_sample]
    img_sample = img_sample * std + mean
    plt.imshow(img_sample.squeeze(), cmap='gray')
    #plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//epoch0_" + str(idx_sample) + ".png")
    plt.savefig("/home/compu/ymh/contrastive/sample/" + "epoch0_" + str(idx_sample) + ".png")
    #plt.savefig("/DATA/ymh/contrastive/sample6/" + "epoch0_" + str(idx_sample) + ".png")
    plt.close()
print("-"*10, end=" ")
print("Image is saved!", end=" ")
print("-"*10)

print("=" * 100)
for epoch in range(500):
    
    g_running_loss=0
    e_running_loss=0
    
    for i in range(60000//batch_size):
        real1, real2 = real_loader.__iter__().next()
        fake1, fake2 = fake_loader.__iter__().next()
        
        real1 = real1.float().to(device)
        real2 = real2.float().to(device)
        fake1 = fake1.float().to(device)
        fake2 = fake2.float().to(device)
        
        real3, _ = mnist_loader.__iter__().next()
        real3 = real3.float().to(device)
        z = torch.from_numpy(np.random.normal(0, 1, (1, latent_dim))).float().to(device)
        
        ###############################################################################################
        encoder.train()
        generator.eval()
        optim_E.zero_grad()
        
        set1 = torch.cat((real1, fake1), dim=0)
        set2 = torch.cat((real2, fake2), dim=0)
        
        rep_set1 = encoder(set1)
        rep_set2 = encoder(set2)
        
        rep_set1 = F.normalize(rep_set1, dim=1)
        rep_set2 = F.normalize(rep_set2, dim=1)
        
        e_loss = contrastive(rep_set1, rep_set2)
        #print(e_loss)
        e_loss.backward()
        optim_E.step()
        e_running_loss += e_loss.item()

        ###############################################################################################
        encoder.eval()
        generator.train()
        optim_G.zero_grad()
        
        gen_img = generator(z)
        
        set3 = torch.cat((real3, fake1))
        set4 = torch.cat((gen_img, fake2), dim=0)
        
        rep_set3 = encoder(set3)
        rep_set4 = encoder(set4)
        
        rep_set3 = F.normalize(rep_set3, dim=1)
        rep_set4 = F.normalize(rep_set4, dim=1)
        
        g_loss = contrastive(rep_set3, rep_set4)
        #print(g_loss)
        g_loss.backward()
        optim_G.step()
        g_running_loss += g_loss.item()
        
    e_running_loss /= (60000//batch_size)
    g_running_loss /= (60000//batch_size)
    print("[Epoch:%d] [E loss:%f] [G Loss:%f]" % ((epoch+1), e_running_loss, g_running_loss), end=" ")

    if (epoch+1) % 1 == 0:
        generator.eval()
        gen_sample = generator(z_sample).detach().cpu()
        
        for idx_sample in range(6):
            img_sample = gen_sample[idx_sample]
            img_sample = img_sample * std + mean
            plt.imshow(img_sample.squeeze(), cmap='gray')
            #plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            plt.savefig("/home/compu/ymh/contrastive/sample/epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            #plt.savefig("/DATA/ymh/contrastive/sample6/epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            plt.close()
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)    





