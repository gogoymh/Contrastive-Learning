import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import Generator, Discriminator, NTXentLoss, Encoder, IndividualLoss
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

########################################################################################################################

device = torch.device("cuda:0")

g = Generator().to(device)
d = Discriminator().to(device)
e = Encoder().to(device)

optim_G = optim.Adam(g.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D = optim.Adam(d.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_E = optim.Adam(e.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Batch size:%d]" % (cfg.test_idx, cfg.batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")

ones = torch.ones((cfg.batch_size, 1)).float().to(device)
zeros = torch.zeros((cfg.batch_size, 1)).float().to(device)

adversarial_loss = nn.BCELoss()
contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)
contrastive_loss2 = IndividualLoss(device, cfg.batch_size)

for iteration in range(20000):
    if (iteration+1) % 100 == 0:
        print("[Iteration:%d]" % (iteration+1), end=" ")
    
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
    
    if (iteration+1) % 100 == 0:
        print("[D real:%f] [D fake:%f]" % (real_loss.item(), fake_loss.item()), end=" ")

    ########################################################################################################################
    d.eval()
    g.train()
        
    z = np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim))
    
    z = torch.from_numpy(z).float().to(device)
    
    optim_G.zero_grad()
    optim_E.zero_grad()
    
    fake = g(z)
    
    adv_loss = adversarial_loss(d(fake), ones)
    
    if (iteration+1) <= 10000:
        #z0 = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))
        #z1 = z0 + np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        #z2 = z0 + np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        
        #z0 = torch.from_numpy(z0).float().to(device)
        #z1 = torch.from_numpy(z1).float().to(device)
        #z2 = torch.from_numpy(z2).float().to(device)
        
        #fake0 = g(z0)
        #fake1 = g(z1)
        #fake2 = g(z2)
        
        rep = e(fake)
        #rep1 = e(fake1)
        #rep2 = e(fake2)
        
        #cont_loss = contrastive_loss(rep1, rep2)
        cont_loss2 = contrastive_loss2(rep)

        #e_loss = adv_loss + cont_loss
        e_loss = adv_loss + cont_loss2
        e_loss.backward()
        optim_E.step()
        optim_G.step()
    else:
        adv_loss.backward()
        optim_G.step()
        
    if (iteration+1) % 100 == 0:
        print("[G adv:%f] [G cont:%f]" % (adv_loss.item(), cont_loss2.item()))
        
    ########################################################################################################################
    if (iteration+1) % 100 == 0:
        torch.save({'model_state_dict': g.state_dict()}, cfg.generator_pre_savefile)
        torch.save({'model_state_dict': d.state_dict()}, cfg.discriminator_pre_savefile)
        #print('model saved')
        
    if (iteration+1) % 100 == 0:
        g.eval()
        fake_sample = g(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(iteration+1) + ".png", nrow=5, normalize=True)
        #print("-"*10, end=" ")
        #print("Image is saved!", end=" ")
        #print("-"*10)