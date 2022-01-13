import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader

from torchlars import LARS
from network import Generator, Discriminator, Encoder, NTXentLoss, Decoder
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

encoder1 = Encoder()
#encoder1 = nn.DataParallel(encoder1)
encoder1.to(device)

encoder2 = Encoder()
#encoder2 = nn.DataParallel(encoder2)
encoder2.to(device)

decoder = Decoder()
#decoder = nn.DataParallel(decoder)
decoder.to(device)

optim_pre_G = LARS(optim.SGD(generator.parameters(), lr=0.1))
optim_pre_D = LARS(optim.SGD(discriminator.parameters(), lr=0.1))
optim_pre_E0 = LARS(optim.SGD(decoder.parameters(), lr=0.1))
optim_pre_E1 = LARS(optim.SGD(encoder1.parameters(), lr=0.1))
optim_pre_E2 = LARS(optim.SGD(encoder2.parameters(), lr=0.1))

contrastive_loss = NTXentLoss(device, cfg.pre_batch_size)

z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)

print("=" * 100)
print("[Test:%d] [Pre batch size:%d] " % (cfg.test_idx, cfg.pre_batch_size), end=" ")
print("[Image Directory:" + cfg.pre_image_savefile +"]")
for epoch in range(cfg.pre_epoch):
    Eg_running_loss = 0
    Ed_running_loss = 0
    
    for i in range(cfg.pre_batch_num):
        ########################################################################################################################
        decoder.train()
        encoder1.train()
        encoder2.train()
        discriminator.train()
        generator.eval()
        
        aug1, aug2 = augment_loader.__iter__().next()
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)
        
        z = np.random.normal(0, 1, (cfg.pre_batch_size, *cfg.latent_dim))
        z1 = np.random.normal(0, cfg.noise_var , (cfg.pre_batch_size, *cfg.latent_dim))
        z2 = np.random.normal(0, cfg.noise_var, (cfg.pre_batch_size, *cfg.latent_dim))
        
        z1 = z + z1
        z2 = z + z2
            
        z1 = torch.from_numpy(z1)
        z2 = torch.from_numpy(z2)
            
        z1 = z1.float().to(device)
        z2 = z2.float().to(device) 
        
        optim_pre_D.zero_grad()
        optim_pre_E0.zero_grad()
        optim_pre_E1.zero_grad()
        optim_pre_E2.zero_grad()
        
        out1 = discriminator(aug1,True)
        out2 = discriminator(aug2,True)
            
        rep1 = encoder1(out1)
        rep2 = encoder1(out2)
        
        fake1 = generator(z1)
        fake2 = generator(z2)
                        
        out3 = discriminator(fake1,True)
        out4 = discriminator(fake2,True)
        
        rep3 = encoder2(out3)
        rep4 = encoder2(out4)
        
        fin1 = decoder(rep1)
        fin2 = decoder(rep2)
        fin3 = decoder(rep3)
        fin4 = decoder(rep4)
        
        d_loss = (contrastive_loss(fin1, fin2) + contrastive_loss(fin3, fin4))/2
        d_loss.backward()
        optim_pre_E0.step()
        optim_pre_E1.step()
        optim_pre_E2.step()
        optim_pre_D.step()
            
        Ed_running_loss += d_loss.item()
            
        ########################################################################################################################
        decoder.eval()
        encoder1.eval()
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
            
        rep1 = encoder1(out1)
        rep2 = encoder1(out2)
        
        fin1 = decoder(rep1)
        fin2 = decoder(rep2)
        
        g_loss = contrastive_loss(fin1, fin2)
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
        torch.save({'model_state_dict': encoder1.state_dict()}, cfg.encoder1_pre_savefile)
        torch.save({'model_state_dict': encoder2.state_dict()}, cfg.encoder2_pre_savefile)
        torch.save({'model_state_dict': decoder.state_dict()}, cfg.decoder_pre_savefile)
        print('model saved')
        
    if (epoch+1) % cfg.pre_image_save == 0:
        generator.eval()
        fake_sample = generator(z_sample)
        
        save_image(fake_sample.data, cfg.pre_image_savefile + str(epoch+1) + ".png", nrow=5, normalize=True)
        print("-"*10, end=" ")
        print("Image is saved!", end=" ")
        print("-"*10)