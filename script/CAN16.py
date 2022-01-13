import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import Generator, Discriminator, NTXentLoss
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
d_keep = Discriminator().to(device)

optim_G = optim.Adam(g.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D = optim.Adam(d.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_E = optim.Adam(d_keep.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Batch size:%d]" % (cfg.test_idx, cfg.batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")

ones = torch.ones((cfg.batch_size, 1)).float().to(device)
zeros = torch.zeros((cfg.batch_size, 1)).float().to(device)

adversarial_loss = nn.BCELoss()
contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)

for epoch in range(cfg.epoch):
    g1_running_loss = 0
    d1_running_loss = 0
    g2_running_loss = 0
    d2_running_loss = 0
    
    for i in range(cfg.batch_num):
        ########################################################################################################################
        d.train()
        g.eval()
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
            
        z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
        z = z.float().to(device)
            
        optim_D.zero_grad()
        
        fake = g(z)
            
        real_loss = adversarial_loss(d(real), ones)
        fake_loss = adversarial_loss(d(fake.detach()), zeros)
    
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D.step()
            
        print("[D:%f]" % d_loss.item(), end=" ")
        d_keep.load_state_dict(d.state_dict())
        ########################################################################################################################
        d_keep.train()
        g.eval()
        
        aug1, aug2 = augment_loader.__iter__().next()
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
        '''
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))            
        z1 = np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        z2 = np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        
        z1 = z + z1
        z2 = z + z2
            
        z1 = torch.from_numpy(z1)
        z2 = torch.from_numpy(z2)
            
        z1 = z1.float().to(device)
        z2 = z2.float().to(device)        
        
        z = torch.from_numpy(z)
        z = z.float().to(device)
        '''
        optim_E.zero_grad()
        rep1 = d_keep(aug1,True)
        rep2 = d_keep(aug2,True)
        '''
        fake1 = g(z1)
        fake2 = g(z2)
        rep3 = d_keep(fake1.detach(),True)
        rep4 = d_keep(fake2.detach(),True)
        '''
        e_loss = contrastive_loss(rep1, rep2) #+ contrastive_loss(rep3, rep4))/2
        optim_E.step()
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
            
        z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
        z = z.float().to(device)
            
        optim_E.zero_grad()
        
        fake = g(z)
            
        real_loss = adversarial_loss(d_keep(real), ones)
        fake_loss = adversarial_loss(d_keep(fake.detach()), zeros)
    
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D.step()
        
        d_keep.eval()
        g.train()
        
        z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
        z = z.float().to(device)

        optim_G.zero_grad()
        fake = g(z)
        g_loss = adversarial_loss(d_keep(fake), ones)
        g_loss.backward()
        optim_G.step()
                
        print("[G:%f]" % g_loss.item())
        
        ########################################################################################################################
        
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