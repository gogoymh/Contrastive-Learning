import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from network import resnet56

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
img_shape = (1, 28, 28)
#data_path = "C://유민형//개인 연구//Constrastive learning//"
data_path = "/home/compu/ymh/contrastive/"
#data_path = "/DATA/ymh/contrastive/"
mean = 0.5 # 0.1307
std = 0.5 # 0.3081
batch_size = 512
lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.1

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
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

########################################################################################################################
mnist_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(28),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (mean,), (std,))]
                                ),
                        ),
                batch_size=batch_size, shuffle=True, pin_memory=True)

device = torch.device("cuda:0")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
encoder = resnet56().to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
optim_E = optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2))

adversarial_loss = nn.BCELoss()
contrastive_loss = NTXentLoss(device, batch_size)

########################################################################################################################
z_sample = torch.from_numpy(np.random.normal(0, 1, (25, latent_dim))).float().to(device)
generator.eval()
fake_sample = generator(z_sample)#.detach().cpu()
#save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample//epoch_0.png", nrow=5, normalize=True)
save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample/epoch_0.png", nrow=5, normalize=True)
#save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_0.png", nrow=5, normalize=True)
print("-"*10, end=" ")
print("Image is saved!", end=" ")
print("-"*10)

########################################################################################################################
class Augment(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(28,28))
        
    def forward(self, fake):
        
        x1, x2 = np.random.choice(7,2)
        y1, y2 = np.random.choice(7,2)
        
        h1, h2 = np.random.choice(list(range(21,28)), 2)
        w1, w2 = np.random.choice(list(range(21,28)), 2)
        
        aug1 = self.upsample(fake[:,:,x1:h1,y1:w1])
        aug2 = self.upsample(fake[:,:,x2:h2,y2:w2])
        
        aug1 = self.sigmoid(aug1 + torch.from_numpy(np.random.normal(0, 0.15, (1, 28, 28))).float().to(self.device))
        aug2 = self.sigmoid(aug2 + torch.from_numpy(np.random.normal(0, 0.15, (1, 28, 28))).float().to(self.device))
        
        return aug1.view(fake.shape[0],-1), aug2.view(fake.shape[0],-1)

########################################################################################################################
ones = torch.ones((batch_size, 1)).float().to(device)
zeros = torch.zeros((batch_size, 1)).float().to(device)

self_augment = Augment(device)

print("=" * 100)
for epoch in range(200):
    
    g_running_loss = 0
    d_running_loss = 0
    e_running_loss = 0
    
    for i in range(60000//batch_size):
        real, _ = mnist_loader.__iter__().next()
        
        real = real.float().to(device)
        z = torch.from_numpy(np.random.normal(0, 1, (real.shape[0], latent_dim))).float().to(device)
        
        ###############################################################################################
        discriminator.eval()
        generator.train()
        #encoder.train()
        
        optim_G.zero_grad()
        #optim_E.zero_grad()
        
        fake = generator(z)
        aug1_fake, aug2_fake = self_augment(fake)
        
        e_loss = contrastive_loss(aug1_fake, aug2_fake)
        g_loss = adversarial_loss(discriminator(fake), ones) + alpha * e_loss
        g_loss.backward()
        optim_G.step()
        #optim_E.step()
        g_running_loss += g_loss.item()
        e_running_loss += e_loss.item()
        
        ###############################################################################################
        discriminator.train()
        generator.eval()
        optim_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(real), ones)
        fake_loss = adversarial_loss(discriminator(fake.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D.step()
        d_running_loss += d_loss.item()
        
        ###############################################################################################
        '''
        encoder.train()
        optim_E.zero_grad()
        
        aug1_real, aug2_real = self_augment(real)
        
        rep_aug1_real = encoder(aug1_real)
        rep_aug2_real = encoder(aug2_real)
        
        rep_aug1_real = F.normalize(rep_aug1_real, dim=1)
        rep_aug2_real = F.normalize(rep_aug2_real, dim=1)
        
        e_loss = contrastive_loss(rep_aug1_real, rep_aug2_real)
        e_loss.backward()
        optim_E.step()
        e_running_loss += e_loss.item()
        '''
        
    g_running_loss /= (60000//batch_size)
    d_running_loss /= (60000//batch_size)
    e_running_loss /= (60000//batch_size)
    print("[Epoch:%d] [G Loss:%f] [D loss:%f] [E loss:%f]" % ((epoch+1), g_running_loss, d_running_loss, e_running_loss))
    
    if (epoch+1) % 1 == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        #save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample//epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        #save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)






