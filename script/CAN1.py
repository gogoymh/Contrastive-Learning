import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader

from torchlars import LARS
from network import Generator, Discriminator, Encoder, NTXentLoss
import config as cfg

########################################################################################################################
aug_mnist = cfg.Augmented_MNIST(cfg.base_root, cfg.transform_pretrain)
augment_loader = DataLoader(aug_mnist, batch_size=cfg.pre_batch_size, shuffle=True, pin_memory=True)

########################################################################################################################

device = torch.device("cuda:0")

generator = Generator()
#generator = nn.DataParallel(generator)
generator.to(device)
       
discriminator = Discriminator()
#discriminator = nn.DataParallel(discriminator)
discriminator.to(device)

encoder = Encoder()
#encoder = nn.DataParallel(encoder)
encoder.to(device)

optim_pre_G = LARS(optim.SGD(generator.parameters(), lr=0.1))
optim_pre_D = LARS(optim.SGD(discriminator.parameters(), lr=0.1))
optim_pre_E = LARS(optim.SGD(encoder.parameters(), lr=0.1))

contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Pre batch size:%d]" % (cfg.test_idx, cfg.pre_batch_size))
for epoch in range(cfg.pre_epoch):
    Eg_running_loss = 0
    Ed_running_loss = 0
    
    for i in range(cfg.pre_batch_num):
        ########################################################################################################################
        encoder.train()
        discriminator.train()
        generator.eval()
        
        aug1, aug2 = augment_loader.__iter__().next()
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
                    
        optim_pre_D.zero_grad()
        optim_pre_E.zero_grad()
                        
        out1 = discriminator(aug1,True)
        out2 = discriminator(aug2,True)
            
        rep1 = encoder(out1)
        rep2 = encoder(out2)
            
        d_loss = contrastive_loss(rep1, rep2)
        d_loss.backward()
        optim_pre_E.step()
        optim_pre_D.step()
            
        Ed_running_loss += d_loss.item()
            
        ########################################################################################################################
        encoder.eval()
        discriminator.eval()
        generator.train()
        
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))
        z1 = np.random.normal(0, cfg.noise_var , (cfg.pre_batch_size, *cfg.latent_dim))
        z2 = np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        
        z1 = z + z1
        z2 = z + z2
            
        z1 = torch.from_numpy(z1)
        z2 = torch.from_numpy(z2)
            
        z1 = z1.float().to(device)
        z2 = z2.float().to(device) 
        
        optim_pre_G.zero_grad()
        
        fake1 = generator(z1)
        fake2 = generator(z2)
                        
        out1 = discriminator(fake1,True)
        out2 = discriminator(fake2,True)
            
        rep1 = encoder(out1)
        rep2 = encoder(out2)
        
        g_loss = contrastive_loss(rep1, rep2)
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
        torch.save({'model_state_dict': encoder.state_dict()}, cfg.encoder_pre_savefile)
        print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)