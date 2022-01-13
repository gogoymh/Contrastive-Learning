import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

from torchlars import LARS
from network import Generator, Discriminator, Encoder, NTXentLoss, Decoder2
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
       
discriminator = Discriminator()
#discriminator = nn.DataParallel(discriminator)
discriminator.to(device)

optim_pre_G = LARS(optim.SGD(generator.parameters(), lr=0.1))
optim_pre_D = LARS(optim.SGD(discriminator.parameters(), lr=0.1))

contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Pre batch size:%d]" % (cfg.test_idx, cfg.pre_batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")
for epoch in range(cfg.pre_epoch):
    Eg_running_loss = 0
    Ed_running_loss = 0
    
    for i in range(cfg.pre_batch_num):
        ########################################################################################################################
        discriminator.train()
        generator.eval()
        
        aug1, aug2 = augment_loader.__iter__().next()
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
        
        optim_pre_D.zero_grad()
        
        out1 = discriminator(aug1,True)
        out2 = discriminator(aug2,True)
        
        d_loss = contrastive_loss(out1, out2)
        d_loss.backward()
        optim_pre_D.step()
            
        Ed_running_loss += d_loss.item()
            
        ########################################################################################################################
        discriminator.eval()
        generator.train()
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))            
        z = torch.from_numpy(z)
        z = z.float().to(device)

        optim_pre_G.zero_grad()
        
        fake = generator(z)
                        
        out1 = discriminator(fake,True)
        out2 = discriminator(real,True)
        
        g_loss = contrastive_loss(out1, out2)
        g_loss.backward()
        optim_pre_G.step()
        
        Eg_running_loss += g_loss.item()
    
    ########################################################################################################################
    Eg_running_loss /= cfg.pre_batch_num
    Ed_running_loss /= cfg.pre_batch_num
    print("[Pre Epoch:%d] [E generator loss:%f] [E discriminator loss:%f]" % ((epoch+1), Eg_running_loss, Ed_running_loss))
        
    if (epoch+1) % cfg.pre_model_save == 0:
        torch.save({'model_state_dict': generator.state_dict()}, cfg.generator_pre_savefile)
        torch.save({'model_state_dict': discriminator.state_dict()}, cfg.discriminator_pre_savefile)
        print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)