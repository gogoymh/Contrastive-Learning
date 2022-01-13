import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import G1, G2, D1, D2, NTXentLoss
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

g1 = G1().to(device)
g2 = G2().to(device)

d1 = D1().to(device)
d2 = D2().to(device)

#optim_G1 = LARS(optim.SGD(g1.parameters(), lr=0.1))
#optim_D1 = LARS(optim.SGD(d1.parameters(), lr=0.1))

optim_G1 = optim.Adam(g1.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D1 = optim.Adam(d1.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

optim_G2 = optim.Adam(g2.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D2 = optim.Adam(d2.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

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
    g2_running_loss = 0
    d_running_loss = 0
    
    for i in range(cfg.batch_num):
        ########################################################################################################################
        '''
        d1.train()
        
        aug1, aug2 = augment_loader.__iter__().next()
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
        
        optim_D1.zero_grad()
        
        rep1 = d1(aug1)
        rep2 = d1(aug2)
        
        d_loss = contrastive_loss(rep1, rep2)
        d_loss.backward()
        optim_D1.step()
        
        d1_running_loss += d_loss.item()
        print("[D1:%f]" % d_loss.item(), end=" ")
        '''
        ########################################################################################################################
        d1.eval()
        g1.train()
        g2.eval()
        
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

        optim_G1.zero_grad()
        
        fake1 = g2(g1(z1))
        fake2 = g2(g1(z2))
                        
        rep1 = d1(fake1)
        rep2 = d1(fake2)
        
        g_loss = contrastive_loss(rep1, rep2)
        g_loss.backward()
        optim_G1.step()
        
        g1_running_loss += g_loss.item()
        print("[G1:%f]" % g_loss.item(), end=" ")
        
        ########################################################################################################################
        d1.train()
        d2.train()
        g1.eval()
        g2.eval()
                
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim))            
        z = torch.from_numpy(z)
        z = z.float().to(device)
        
        optim_D1.zero_grad()
        optim_D2.zero_grad()
        
        fake = g2(g1(z))
        
        real_loss = adversarial_loss(d2(d1(real)), ones)
        fake_loss = adversarial_loss(d2(d1(fake)), zeros)
            
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D1.step()
        optim_D2.step()
            
        d_running_loss += d_loss.item()
        print("[D:%f]" % d_loss.item(), end=" ")
        
        ########################################################################################################################
        d1.eval()
        d2.eval()
        g1.eval()
        g2.train()
        
        z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
        z = z.float().to(device)
        
        optim_G2.zero_grad()
        
        fake = g2(g1(z))
        
        g_loss = adversarial_loss(d2(d1(fake)), ones)
        g_loss.backward()
        optim_G2.step()
        
        g2_running_loss += g_loss.item()
        print("[G2:%f]" % g_loss.item())
        
    ########################################################################################################################
    g1_running_loss /= cfg.batch_num
    g2_running_loss /= cfg.batch_num
    d_running_loss /= cfg.batch_num
    print("[Epoch:%d] [G1 loss:%f] [G2 loss:%f] [D loss:%f]" % ((epoch+1), g1_running_loss, g2_running_loss, d_running_loss))
        
    if (epoch+1) % cfg.pre_model_save == 0:
        torch.save({'model_state_dict': g1.state_dict()}, cfg.g1_pre_savefile)
        torch.save({'model_state_dict': g2.state_dict()}, cfg.g2_pre_savefile)
        torch.save({'model_state_dict': d1.state_dict()}, cfg.d1_pre_savefile)
        torch.save({'model_state_dict': d2.state_dict()}, cfg.d2_pre_savefile)
        print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        g1.eval()
        g2.eval()
        fake_sample = g2(g1(z_sample))
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)