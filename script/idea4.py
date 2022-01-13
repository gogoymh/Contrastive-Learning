import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random

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
latent_dim = 100
img_shape = (1, 32, 32)
mean = 0.5 # 0.1307
std = 0.5 # 0.3081
batch_size = 1024
lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.01

########################################################################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

########################################################################################################################
device = torch.device("cuda:0")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

adversarial_loss = nn.BCELoss()
contrastive_loss = NTXentLoss(device, batch_size)

########################################################################################################################
z_sample = torch.from_numpy(np.random.normal(0, 1, (25, latent_dim))).float().to(device)
generator.eval()
fake_sample = generator(z_sample)
#save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample//epoch_0.png", nrow=5, normalize=True)
save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample2/epoch_0.png", nrow=5, normalize=True)
#save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_0.png", nrow=5, normalize=True)
print("-"*10, end=" ")
print("Image is saved!", end=" ")
print("-"*10)

########################################################################################################################
class Augmented_latent(Dataset):
    def __init__(self, transform, mean=0.5, std=0.5, num=1000000):
        super().__init__()
        
        self.transform = transform
        self.len = num
        
        self.latent = np.random.normal(0, 1, (self.len,10,10,1))
        self.latent = self.latent * std + mean
        self.latent = self.latent * 255
        self.latent = self.latent.astype('uint8')
        
    def __getitem__(self, index):
        
        latent = self.latent[index]
        
        latent1 = self.transform(latent)
        latent2 = self.transform(latent)
        
        return latent, latent1, latent2

    def __len__(self):
        return self.len
        
color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(0, scale=[0.8,1.2], shear=[-15, 15, -15, 15]),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

augmented_latent = Augmented_latent(transform=transform)
latent_loader = DataLoader(dataset=augmented_latent, batch_size=batch_size, shuffle=True, pin_memory=True)

mnist_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (mean,), (std,))]
                                ),
                        ),
                batch_size=batch_size, shuffle=True, pin_memory=True)

########################################################################################################################
ones = torch.ones((batch_size, 1)).float().to(device)
zeros = torch.zeros((batch_size, 1)).float().to(device)

batch_num = 60000//batch_size

print("=" * 100)
for epoch in range(200):
    
    g_running_loss = 0
    d_running_loss = 0
    e_running_loss = 0
        
    for i in range(batch_num):
        real, _ = mnist_loader.__iter__().next()
        z, z1, z2 = latent_loader.__iter__().next()
        
        real = real.float().to(device)
            
        z = z.float().to(device)
        z1 = z1.float().to(device)
        z2 = z2.float().to(device)
        
        ###############################################################################################
        discriminator.eval()
        generator.train()
        
        optim_G.zero_grad()
            
        fake = generator(z.view(batch_size,-1))
        
        adv_loss = adversarial_loss(discriminator(fake), ones)
        
        #_, rep1 = generator(z1.view(batch_size,-1))
        #_, rep2 = generator(z2.view(batch_size,-1))
        #contr_loss = contrastive_loss(rep1, rep2)
        g_loss = adv_loss# + alpha * contr_loss
        
        g_loss.backward()
        optim_G.step()
        
        #print(adv_loss.item(), end=" ")
        #print(contr_loss.item(), end=" ")
        g_running_loss += adv_loss.item()
        #e_running_loss += contr_loss.item()
        
        ###############################################################################################
        discriminator.train()
        generator.eval()
        
        optim_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(real), ones)
        fake_loss = adversarial_loss(discriminator(fake.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optim_D.step()
        
        #print(d_loss.item())
        d_running_loss += d_loss.item()
    
    g_running_loss /= batch_num
    d_running_loss /= batch_num
    e_running_loss /= batch_num
    print("[Epoch:%d] [G Loss:%f] [D loss:%f] [E loss:%f]" % ((epoch+1), g_running_loss, d_running_loss, e_running_loss))
    
    if (epoch+1) % 10 == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        #save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample//epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample2/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        #save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)






