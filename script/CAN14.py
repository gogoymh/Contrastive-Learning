import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import Generator, Discriminator, Encoder, NTXentLoss
import config as cfg

########################################################################################################################
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
                batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

aug_mnist = cfg.Augmented_MNIST(cfg.base_root, cfg.transform_pretrain)
augment_loader = DataLoader(aug_mnist, batch_size=cfg.pre_batch_size, shuffle=True, pin_memory=True)
########################################################################################################################

device = torch.device("cuda:0")

g = Generator().to(device)
d = Discriminator().to(device)

optim_G = optim.Adam(g.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D = optim.Adam(d.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Batch size:%d]" % (cfg.test_idx, cfg.batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")

ones = torch.ones((cfg.batch_size, 1)).float().to(device)
zeros = torch.zeros((cfg.batch_size, 1)).float().to(device)

adversarial_loss = nn.BCELoss()
contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)

for epoch in range(cfg.epoch):
    
    for i in range(cfg.batch_num):        
        ########################################################################################################################
        d.train()
        g.eval()
        
        if (i+1) % 2 != 0:
            aug1, aug2 = augment_loader.__iter__().next()
            aug1 = aug1.float().to(device)
            aug2 = aug2.float().to(device)
        
            optim_D.zero_grad()
        
            rep1 = d(aug1)#,True)
            rep2 = d(aug2)#,True)
        
            e_loss = contrastive_loss(rep1, rep2)
            e_loss.backward()
            optim_D.step()
            
            print("[E:%f]" % e_loss.item(), end=" ")
            
        else:        
            real,_ = real_loader.__iter__().next()
            real = real.float().to(device)
            
            z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
            z = z.float().to(device)
            
            optim_D.zero_grad()
            
            fake = g(z)
            
            d_loss = -torch.mean(d(real)) + torch.mean(d(fake.detach()))
            d_loss.backward()
            optim_D.step()
            
            print("[D:%f]" % d_loss.item(), end=" ")
            
            #for p in d.parameters():
            #    p.data.clamp_(-0.01, 0.01)
        
        ########################################################################################################################
        if (i+1) % 4 == 0:
            d.eval()
            g.train()
        
            z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
            z = z.float().to(device)

            optim_G.zero_grad()
            
            fake = g(z)
            g_loss = -torch.mean(d(fake))
            g_loss.backward()
            optim_G.step()
                
            print("[G:%f]" % g_loss.item())
        
    ########################################################################################################################
        
    if (epoch+1) % cfg.pre_model_save == 0:
        torch.save({'model_state_dict': g.state_dict()}, cfg.generator_pre_savefile)
        torch.save({'model_state_dict': d.state_dict()}, cfg.discriminator_pre_savefile)
        #print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        g.eval()
        fake_sample = g(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        #print("-"*10, end=" ")
        #print("Image is saved!", end=" ")
        #print("-"*10)