import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import Generator, Discriminator
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

checkpoint_generator = torch.load(cfg.generator_load)
generator = Generator()
generator.load_state_dict(checkpoint_generator['model_state_dict'])
#generator = nn.DataParallel(generator)
generator.to(device)

checkpoint_discriminator = torch.load(cfg.discriminator_load)
discriminator = Discriminator()
discriminator.load_state_dict(checkpoint_discriminator['model_state_dict'])
#discriminator = nn.DataParallel(discriminator)
discriminator.to(device)

optim_G = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
optim_D = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Batch size:%d]" % (cfg.test_idx, cfg.batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")

ones = torch.ones((cfg.batch_size, 1)).float().to(device)
zeros = torch.zeros((cfg.batch_size, 1)).float().to(device)

adversarial_loss = nn.BCELoss()

for epoch in range(cfg.epoch):
    Eg_running_loss = 0
    Ed_running_loss = 0
    
    for i in range(cfg.batch_num):
        ########################################################################################################################
        discriminator.train()
        generator.eval()
        
        real,_ = real_loader.__iter__().next()
        real = real.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim))            
        z = torch.from_numpy(z)
        z = z.float().to(device)
        
        optim_D.zero_grad()
        
        fake = generator(z)
        
        real_loss = adversarial_loss(discriminator(real), ones)
        fake_loss = adversarial_loss(discriminator(fake.detach()), zeros)
            
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D.step()
            
        Ed_running_loss += d_loss.item()
            
        ########################################################################################################################
        discriminator.eval()
        generator.train()
        
        z = torch.from_numpy(np.random.normal(0, 1, (cfg.batch_size, *cfg.latent_dim)))
        z = z.float().to(device)
        
        optim_G.zero_grad()
        
        fake = generator(z)
        
        g_loss = adversarial_loss(discriminator(fake), ones)
        g_loss.backward()
        optim_G.step()
        
        Eg_running_loss += g_loss.item()
    
    ########################################################################################################################
    Eg_running_loss /= cfg.batch_num
    Ed_running_loss /= cfg.batch_num
    print("[Epoch:%d] [Generator loss:%f] [Discriminator loss:%f]" % ((epoch+1), Eg_running_loss, Ed_running_loss))
        
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