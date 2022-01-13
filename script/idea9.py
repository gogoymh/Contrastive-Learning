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
from skimage.io import imread
from torchlars import LARS

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
img_shape = (3, 64, 64)
mean = 0.5 # 0.1307
std = 0.5 # 0.3081
pre_batch_size = 260*7
batch_size = 128

lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.01

test_idx = 5
pre_train = False
pre_test_idx = 4
pretrained_path = "/home/compu/ymh/contrastive/test" + str(pre_test_idx) + "generator.pth"
batch_num = 202599//pre_batch_size if pre_train else 202599//batch_size

########################################################################################################################
class Generator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.sigmoid(self.conv5(x))

        return x.view(x.shape[0], 1)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64)
            )
        
    def forward(self, x):
        
        out = x.view(x.shape[0], -1)
        out = self.fc(out)
        
        return out

########################################################################################################################
device = torch.device("cuda:0")

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, latent_dim, 1, 1))).float().to(device)

########################################################################################################################
class CelebA_Image(Dataset):
    def __init__(self, root, image_path=None, transform=None):
        super().__init__()
        
        self.transform=transform
        self.len = 202599
        
        save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
        save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
        save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
        save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
        save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
        
        if os.path.isfile(save_file1):
            self.CelebA_part1 = np.load(save_file1)
            print("Part 1 is loaded.")
        else:
            self.CelebA_part1 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx+1)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part1[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
            np.save(save_file1, self.CelebA_part1)
            print("Part1 is saved.")
        
        if os.path.isfile(save_file2):
            self.CelebA_part2 = np.load(save_file2)
            print("Part 2 is loaded.")
        else:
            self.CelebA_part2 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 40000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part2[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
            np.save(save_file2, self.CelebA_part2)
            print("Part2 is saved.")
        
        if os.path.isfile(save_file3):
            self.CelebA_part3 = np.load(save_file3)
            print("Part 3 is loaded.")
        else:
            self.CelebA_part3 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 80000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part3[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
            np.save(save_file3, self.CelebA_part3)
            print("Part3 is saved.")
        
        if os.path.isfile(save_file4):
            self.CelebA_part4 = np.load(save_file4)
            print("Part 4 is loaded.")
        else:
            self.CelebA_part4 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 120000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part4[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
            np.save(save_file4, self.CelebA_part4)
            print("Part4 is saved.")
        
        if os.path.isfile(save_file5):
            self.CelebA_part5 = np.load(save_file5)
            print("Part 5 is loaded.")
        else:
            self.CelebA_part5 = np.empty((42599,218,178,3), dtype='uint8')
            
            for idx in range(42599):
                image_name = "%06d.jpg" % (idx + 1 + 160000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part5[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 42599))
                
            save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
            np.save(save_file5, self.CelebA_part5)
            print("Part5 is saved.")
        
        self.CelebA = np.concatenate((self.CelebA_part1, self.CelebA_part2, self.CelebA_part3, self.CelebA_part4, self.CelebA_part5), axis=0)
        print("Dataset is ready.")
        
        #self.CelebA = self.CelebA.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.CelebA[index]
        
        if self.transform is not None:
            img = self.transform(img)
        else:  
            img = img.transpose(2,0,1)
        
        return img
    
    def __len__(self):
        return self.len

########################################################################################################################
if pre_train:
    model_name_generator = "/DATA/ymh/contrastive/test" + str(4) + "generator.pth"
    checkpoint1 = torch.load(model_name_generator)
    
    generator = Generator()
    generator = nn.DataParallel(generator)
    generator.to(device)
    
    generator.load_state_dict(checkpoint1['model_state_dict'])
    
    model_name_encoder = "/DATA/ymh/contrastive/test" + str(4) + "encoder.pth"
    checkpoint2 = torch.load(model_name_encoder)
    
    encoder = Encoder()
    encoder = nn.DataParallel(encoder)
    encoder.to(device)
    
    encoder.load_state_dict(checkpoint2['model_state_dict'])

    optim_pre_G = LARS(optim.SGD(generator.parameters(), lr=0.1))
    optim_pre_E = LARS(optim.SGD(encoder.parameters(), lr=0.1))
    
    contrastive_loss = NTXentLoss(device, pre_batch_size)
    
    print("=" * 100)
    for epoch in range(1000):
        encoder.train()
        generator.train()
        
        e_running_loss = 0
    
        for i in range(batch_num):
            z = np.random.normal(0, 1, (pre_batch_size, latent_dim, 1, 1))
            z1 = np.random.normal(0, 0.01, (pre_batch_size, latent_dim, 1, 1))
            z2 = np.random.normal(0, 0.01, (pre_batch_size, latent_dim, 1, 1))
            
            z1 = z + z1
            z2 = z + z2
            
            z1 = torch.from_numpy(z1)
            z2 = torch.from_numpy(z2)
            
            z1 = z1.float().to(device)
            z2 = z2.float().to(device)
        
            optim_pre_E.zero_grad()
            optim_pre_G.zero_grad()
        
            rep1 = encoder(generator(z1))
            rep2 = encoder(generator(z2))
        
            e_loss = contrastive_loss(rep1, rep2)
            e_loss.backward()
            optim_pre_E.step()
            optim_pre_G.step()
        
            e_running_loss += e_loss.item()
        
        e_running_loss /= batch_num
        print("[Pre Epoch:%d] [E loss:%f]" % ((epoch+1), e_running_loss))
        
        if (epoch+1) % 10 == 0:
            #model_name_generator = "/home/compu/ymh/contrastive/test" + str(test_idx) + "generator.pth"
            model_name_generator = "/DATA/ymh/contrastive/test" + str(test_idx) + "generator.pth"
            torch.save({'model_state_dict': generator.state_dict()}, model_name_generator)
            
            #model_name_encoder = "/home/compu/ymh/contrastive/test" + str(test_idx) + "encoder.pth"
            model_name_encoder = "/DATA/ymh/contrastive/test" + str(test_idx) + "encoder.pth"
            torch.save({'model_state_dict': encoder.state_dict()}, model_name_encoder)
            print('model saved')

else:
    from collections import OrderedDict
    
    transform_celebA = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([64,64]),
    transforms.ToTensor(),
    transforms.Normalize((mean,mean,mean), (std,std,std))
    ])

    celebA = CelebA_Image(root="/home/compu/ymh/contrastive/data/CelebA",
                          transform=transform_celebA)

    celebA_loader = DataLoader(dataset=celebA, batch_size=batch_size, shuffle=True, pin_memory=True)
    ########################################################################################################################
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    checkpoint = torch.load(pretrained_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    generator.load_state_dict(new_state_dict)
    generator.train()
    
    optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    
    adversarial_loss = nn.BCELoss()

    generator.eval()
    fake_sample = generator(z_sample.view(25, latent_dim, 1, 1))
    #save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample3//epoch_0.png", nrow=5, normalize=True)
    save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample4/epoch_0.png", nrow=5, normalize=True)
    #save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_0.png", nrow=5, normalize=True)
    print("-"*10, end=" ")
    print("Image is saved!", end=" ")
    print("-"*10)

    ########################################################################################################################
    ones = torch.ones((batch_size, 1)).float().to(device)
    zeros = torch.zeros((batch_size, 1)).float().to(device)
    
    print("=" * 100)
    for epoch in range(200):
    
        g_running_loss = 0
        d_running_loss = 0
            
        for i in range(batch_num):
            ###############################################################################################
            real = celebA_loader.__iter__().next()
            real = real.float().to(device)
            z = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim, 1, 1)))
            z = z.float().to(device)    
        
            discriminator.train()
            generator.eval()
        
            optim_D.zero_grad()
            
            fake = generator(z.view(batch_size, latent_dim, 1, 1))
        
            real_loss = adversarial_loss(discriminator(real), ones)
            fake_loss = adversarial_loss(discriminator(fake.detach()), zeros)
            
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optim_D.step()
        
            d_running_loss += d_loss.item()
            #print(d_loss.item())
        
            ###############################################################################################
            z = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim, 1, 1)))
            z = z.float().to(device)
        
            discriminator.eval()
            generator.train()
        
            optim_G.zero_grad()
        
            fake = generator(z.view(batch_size, latent_dim, 1, 1))
        
            g_loss = adversarial_loss(discriminator(fake), ones)
            g_loss.backward()
            optim_G.step()
        
            g_running_loss += g_loss.item()
            #print(g_loss.item(), end=" ")
    
        g_running_loss /= batch_num
        d_running_loss /= batch_num
        print("[Epoch:%d] [G Loss:%f] [D loss:%f]" % ((epoch+1), g_running_loss, d_running_loss))
    
        if (epoch+1) % 1 == 0:
            generator.eval()
            fake_sample = generator(z_sample)
            
            #save_image(fake_sample.data, "C://유민형//개인 연구//Constrastive learning//sample3//epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
            save_image(fake_sample.data, "/home/compu/ymh/contrastive/sample4/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
            #save_image(fake_sample.data, "/DATA/ymh/contrastive/sample6/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
            print("-"*10, end=" ")
            print("Image is saved!", end=" ")
            print("-"*10)






