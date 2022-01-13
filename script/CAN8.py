import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

from torchlars import LARS
from network import Generator, Discriminator, NTXentLoss
import config as cfg

########################################################################################################################
aug_mnist = cfg.Augmented_MNIST(cfg.base_root, cfg.transform_pretrain)
augment_loader = DataLoader(aug_mnist, batch_size=cfg.pre_batch_size, shuffle=True, pin_memory=True)
real_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(cfg.img_shape[1]),
                                transforms.ToTensor(),
                                transforms.Normalize((cfg.mean,), (cfg.std,))
                                ])
                        ),
                batch_size=cfg.pre_batch_size, shuffle=False, pin_memory=True)
########################################################################################################################

device = torch.device("cuda:0")

generator = Generator()
#generator = nn.DataParallel(generator)
generator.to(device)
       
discriminator1 = Discriminator()
#discriminator1 = nn.DataParallel(discriminator1)
discriminator1.to(device)

discriminator2 = Discriminator()
#discriminator2 = nn.DataParallel(discriminator2)
discriminator2.to(device)

optim_pre_G = LARS(optim.SGD(generator.parameters(), lr=0.1))
optim_pre_D1 = LARS(optim.SGD(discriminator1.parameters(), lr=0.1))

contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)
cosine_similarity = nn.CosineSimilarity(dim=0)

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Pre batch size:%d]" % (cfg.test_idx, cfg.pre_batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")
for epoch in range(cfg.pre_epoch):
    Eg_running_loss = 0
    Ed_running_loss = 0
    
    for i in range(cfg.pre_batch_num):
        ########################################################################################################################
        discriminator1.train()
        discriminator2.eval()
        generator.eval()
        
        discriminator2.load_state_dict(discriminator1.state_dict())
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))            
        z = torch.from_numpy(z)
        z = z.float().to(device)
        
        optim_pre_D1.zero_grad()
        
        fake = generator(z)
        
        out1 = discriminator1(real,True)
        out2 = discriminator2(fake,True)
        
        out1 = out1.mean(dim=0)
        out2 = out2.mean(dim=0)
        
        d_loss = (cosine_similarity(out1, out2) + 1)/2
        d_loss.backward()
        optim_pre_D1.step()
            
        Ed_running_loss += d_loss.item()
            
        ########################################################################################################################
        discriminator1.eval()
        discriminator2.eval()
        generator.train()
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))            
        z = torch.from_numpy(z)
        z = z.float().to(device)

        optim_pre_G.zero_grad()
        
        fake = generator(z)
                        
        out1 = discriminator1(fake,True)
        out2 = discriminator1(real,True)
        
        smp1 = out1[0]
        smp2 = out2[0]
                
        out1 = out1.mean(dim=0)
        out2 = out2.mean(dim=0)
        
        g_loss = (2 - cosine_similarity(out1, out2) - cosine_similarity(smp1, smp2))/4
        g_loss.backward()
        optim_pre_G.step()
        
        Eg_running_loss += g_loss.item()
    
    ########################################################################################################################
    Eg_running_loss /= cfg.pre_batch_num
    Ed_running_loss /= cfg.pre_batch_num
    print("[Pre Epoch:%d] [E generator loss:%f] [E discriminator loss:%f]" % ((epoch+1), Eg_running_loss, Ed_running_loss))
        
    if (epoch+1) % cfg.pre_model_save == 0:
        torch.save({'model_state_dict': generator.state_dict()}, cfg.generator_pre_savefile)
        torch.save({'model_state_dict': discriminator1.state_dict()}, cfg.discriminator_pre_savefile)
        print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)