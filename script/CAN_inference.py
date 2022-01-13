import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

#from torchlars import LARS
from network import G1, G2, D1, D2 #, NTXentLoss
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

checkpoint_g1 = torch.load(cfg.g1_load)
g1 = G1().to(device)
g1.load_state_dict(checkpoint_g1['model_state_dict'])

checkpoint_g2 = torch.load(cfg.g2_load)
g2 = G2().to(device)
g2.load_state_dict(checkpoint_g2['model_state_dict'])

g1.eval()
g2.eval()

for i in range(6*6):
    z_sample = torch.from_numpy(np.random.normal(0, 1, (25, *cfg.latent_dim))).float().to(device)
    fake_sample = g2(g1(z_sample))        
    save_image(fake_sample.data, cfg.pre_image_savefile + str(i+1) + ".png", nrow=5, normalize=True)
    print(i)
