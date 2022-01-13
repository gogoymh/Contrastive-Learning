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
#data_path = "/home/compu/ymh/contrastive/"
data_path = "/DATA/ymh/contrastive/"
batch_size = 512
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
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
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
class Augmented_MNIST(Dataset):
    def __init__(self, root, real_transform=None, aug_transform=None):
        super().__init__()
        
        self.real_transform = real_transform
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
        
        if self.real_transform is not None:
            real = self.real_transform(img)
        else:  
            real = img.transpose(2,0,1)
            
        if self.aug_transform is not None:
            aug1 = self.aug_transform(img)
            aug2 = self.aug_transform(img)
        else:  
            aug1 = real
            aug2 = real
        
        return real, aug1, aug2
    
    def __len__(self):
        return self.len

class Real_Fake_MNIST(Dataset):
    def __init__(self, root, real_transform=None, fake_transform=None):
        super().__init__()
        
        self.real_transform = real_transform
        self.fake_transform = fake_transform
        
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
        self.noise = np.random.choice(255,(60000,28,28,1))
        
    def __getitem__(self, index):
        
        choice = np.random.choice(self.len,2)
        
        real1 = self.mnist[choice[0]]
        real2 = self.mnist[choice[1]]
        
        fake1 = self.noise[choice[0]]
        fake2 = self.noise[choice[1]]
        
        if self.real_transform is not None:
            real1 = self.real_transform(real1)
            real2 = self.real_transform(real2)
            
        else:  
            real1 = real1.transpose(2,0,1)
            real2 = real2.transpose(2,0,1)
        
        if self.fake_transform is not None:
            fake1 = self.fake_transform(fake1)
            fake2 = self.fake_transform(fake2)
            
        else:  
            fake1 = fake1.transpose(2,0,1)
            fake2 = fake2.transpose(2,0,1)
        
        return real1, real2, fake1, fake2
    
    def __len__(self):
        return self.len


########################################################################################################################
class DiceCoefficient_one2one(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        
        self.smooth = smooth

    def forward(self, pred, target):
        
        intersection = torch.mul(pred, target)        
        coef = (2. * torch.sum(intersection) + self.smooth) / (torch.sum(pred) + torch.sum(target) + self.smooth)
        
        return coef

class DiceCoefficient_one2many(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        
        self.smooth = smooth

    def forward(self, pred, target):
        
        batch_size = target.shape[0]
        
        m1 = pred.expand(batch_size, -1)
        m2 = target.view(batch_size, -1)
        
        intersection = torch.mul(m1,m2)        
        coef = (2. * torch.sum(intersection,dim=1) + self.smooth) / (torch.sum(m1,dim=1) + torch.sum(m2,dim=1) + self.smooth)
        
        return coef

class make_logit(nn.Module):
    def __init__(self, temperature=2):
        super().__init__()
        
        self.one2one = DiceCoefficient_one2one()
        self.one2many = DiceCoefficient_one2many()
        
        self.temperature = temperature
        
    def forward(self, one, target, many):
        
        #numerator = torch.exp(self.one2one(one, target)/self.temperature)
        #denominator = torch.sum(torch.exp(self.one2many(one, many)/self.temperature))
        
        numerator = self.one2one(one, target)/self.temperature
        denominator = torch.sum(self.one2many(one, many)/self.temperature)
                
        logit = - torch.log(numerator/denominator)
        
        return logit

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.logit = make_logit()
        
    def vec_except(self, vec, idx):
        vec = torch.cat([vec[0:idx], vec[idx+1:]])
        return vec
        
    def forward(self, image1, image2):
        batch_size = image1.shape[0]
        
        loss = 0
        many1 = torch.cat((image1, image2), dim=0)
        many2 = torch.cat((image2, image1), dim=0)
        
        for i in range(batch_size):
            loss += self.logit(image1[i], many1[i+batch_size], self.vec_except(many1, i))
            
        for j in range(batch_size):
            loss += self.logit(image2[j], many2[j+batch_size], self.vec_except(many2, j))
            
        loss /= 2. * batch_size
        
        return loss
    

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

color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform2 = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomResizedCrop(32),
    transforms.Resize(32),
    transforms.RandomAffine(0, scale=[0.8,1.2], shear=[-15, 15, -15, 15]),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

aug_mnist = Augmented_MNIST(data_path, real_transform=transform1, aug_transform=transform2)

########################################################################################################################
train_loader = DataLoader(dataset=aug_mnist, batch_size=batch_size, shuffle=True)#, pin_memory=True)

device = torch.device("cuda:0")
generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator.to(device)

encoder = resnet56().to(device)
#encoder = nn.DataParallel(encoder)
#encoder.to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr)#, weight_decay=weight_decay)
optim_E = optim.Adam(encoder.parameters(), lr=lr)#, weight_decay=weight_decay)

#scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optim_G, T_max=(60000//batch_size))
#scheduler_E = optim.lr_scheduler.CosineAnnealingLR(optim_E, T_max=(60000//batch_size))

contrastive = NTXentLoss(device, batch_size)# ContrastiveLoss() # #  #
generate_loss = nn.L1Loss()

z_sample = torch.from_numpy(np.random.normal(0, 1, (6, latent_dim))).float().to(device)
generator.eval()
fake_sample = generator(z_sample).detach().cpu()


for idx_sample in range(6):
    img_sample = fake_sample[idx_sample]
    img_sample = img_sample * std + mean
    plt.imshow(img_sample.squeeze(), cmap='gray')
    #plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//epoch0_" + str(idx_sample) + ".png")
    #plt.savefig("/home/compu/ymh/contrastive/sample2/" + "epoch0_" + str(idx_sample) + ".png")
    plt.savefig("/DATA/ymh/contrastive/sample6/" + "epoch0_" + str(idx_sample) + ".png")
    plt.close()
print("-"*10, end=" ")
print("Image is saved!", end=" ")
print("-"*10)


print("=" * 100)
for epoch in range(500):
    
    g_running_loss=0
    e_running_loss=0
    
    #for idx, (real, aug1, aug2) in enumerate(train_loader):
        #print("&"*10)
    for i in range(60000//batch_size):
        real, aug1, aug2 = train_loader.__iter__().next()
        
        real = real.float().to(device)
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
        z = torch.from_numpy(np.random.normal(0, 1, (real.shape[0], latent_dim))).float().to(device)
        
        ###############################################################################################
        encoder.eval()
        generator.train()
        optim_G.zero_grad()
        
        fake = generator(z)
        
        rep_fake = encoder(fake)
        rep_real = encoder(real)
        
        #rep_fake = F.normalize(rep_fake, dim=1)
        #rep_real = F.normalize(rep_real, dim=1)
        
        g_loss = generate_loss(rep_fake, rep_real) # contrastive(rep_fake, rep_real)
        #print(g_loss)
        g_loss.backward()
        optim_G.step()
        g_running_loss += g_loss.item()
        
        ###############################################################################################
        encoder.train()
        generator.eval()
        optim_E.zero_grad()
        
        rep_aug1 = encoder(aug1)
        rep_aug2 = encoder(aug2)
        
        rep_aug1 = F.normalize(rep_aug1, dim=1)
        rep_aug2 = F.normalize(rep_aug2, dim=1)
        
        e_loss = contrastive(rep_aug1, rep_aug2)
        #print(e_loss)
        e_loss.backward()
        optim_E.step()
        e_running_loss += e_loss.item()
        
    g_running_loss /= (60000//batch_size)
    e_running_loss /= (60000//batch_size)
    print("[Epoch:%d] [G Loss:%f] [E loss:%f]" % ((epoch+1), g_running_loss, e_running_loss), end=" ")
    
    if (epoch+1) % 1 == 0:
        generator.eval()
        fake_sample = generator(z_sample).detach().cpu()
        
        for idx_sample in range(6):
            img_sample = fake_sample[idx_sample]
            img_sample = img_sample * std + mean
            plt.imshow(img_sample.squeeze(), cmap='gray')
            #plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            #plt.savefig("/home/compu/ymh/contrastive/sample2/epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            plt.savefig("/DATA/ymh/contrastive/sample6/epoch" + str(epoch+1) + "_" + str(idx_sample) + ".png")
            plt.close()
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)    

    #if (epoch+1) >= 10:
    #    scheduler_G.step()
    #    scheduler_E.step()



'''
real, aug1, aug2 = train_loader.__iter__().next()

real = real[0].numpy()
plt.imshow(real.squeeze(), cmap='gray')
plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//" + "real.png")
plt.close()

aug1 = aug1[0].numpy()
plt.imshow(aug1.squeeze(), cmap='gray')
plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//" + "aug1.png")
plt.close()

aug2 = aug2[0].numpy()
plt.imshow(aug2.squeeze(), cmap='gray')
plt.savefig("C://유민형//개인 연구//Constrastive learning//sample//" + "aug2.png")
plt.close()
'''






























