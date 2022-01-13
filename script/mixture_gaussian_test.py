import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn
#from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mixture_gaussian import data_generator, NTXentLoss, IndividualLoss
plt.style.use('ggplot')

########################################################################################################################
cuda = True
device = torch.device("cuda:0")

########################################################################################################################
## choose uniform mixture gaussian or weighted mixture gaussian
dset = data_generator()
dset.random_distribution()
#dset.uniform_distribution()

plt.plot(dset.p)
plt.title('Weight of each gaussian')
plt.show()
plt.close()

########################################################################################################################
def plot(points, title):

    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()
    
sample_points = dset.sample(512)
plot(sample_points, 'Sampled data points')

########################################################################################################################
# Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
z_dim = 256
g_inp = z_dim
g_hid = 128
g_out = dset.size

d_inp = g_out
d_hid = 128
d_out = 1

e_inp = g_out
e_hid = 128
e_out = 64

minibatch_size = 512

unrolled_steps = -10
e_learning_rate = 1e-4
d_learning_rate = 1e-4
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 20001
half_iter = 10000
log_interval = 100
d_steps = 1
g_steps = 1

prefix = "unrolled_steps-{}-prior_std-{:.2f}".format(unrolled_steps, np.std(dset.p))
print("Save file with prefix", prefix)

########################################################################################################################
def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')

########################################################################################################################
###### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, hidden_size)
        self.map5 = nn.Linear(hidden_size, output_size)
        #self.activation_fn = nn.Tanh()
        self.activation_fn = nn.ReLU()
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        x = self.activation_fn(self.map3(x))
        x = self.activation_fn(self.map4(x))
        return self.map5(x)
        #return self.tanh(self.map5(x)) * 4
    

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, hidden_size)
        self.map5 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        x = self.activation_fn(self.map3(x))
        x = self.activation_fn(self.map4(x))
        return self.sigmoid(self.map5(x))
        #return self.map3(x)
    
    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
                    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.map1(x))
        return self.map2(out)

########################################################################################################################
G = Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out)
D = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out)
E = Encoder(input_size=e_inp, hidden_size=e_hid, output_size=e_out)
if cuda:
    G.to(device)
    D.to(device)
    E.to(device)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
e_optimizer = optim.Adam(E.parameters(), lr=e_learning_rate, betas=optim_betas)

contrastive_loss = NTXentLoss(device, minibatch_size)
contrastive_loss2 = IndividualLoss(device, minibatch_size)

########################################################################################################################
def d_loop(iteration):
    d_optimizer.zero_grad()
    #noise = torch.from_numpy(np.random.normal(0, 0.01, (minibatch_size, d_inp))).float().to(device)
    
    z = dset.sample(minibatch_size)
    d_real_data = torch.from_numpy(z).float()
    if cuda:
        d_real_data = d_real_data.to(device)
        
    #if iteration > half_iter:
    #    d_real_data = d_real_data + noise
    
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)
    #d_real_error = torch.mean(d_real_decision)

    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp)).float()
    if cuda:
        d_gen_input = d_gen_input.to(device)
        
    with torch.no_grad():
        d_fake_data = G(d_gen_input)
        #if iteration > half_iter:
        #    d_fake_data = d_fake_data + noise
    
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)
    #d_fake_error = torch.mean(d_fake_decision)
    
    d_loss = (d_real_error + d_fake_error)/2
    #d_loss = -d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()

########################################################################################################################
def d_unrolled_loop(d_gen_input=None):
    d_optimizer.zero_grad()

    z = dset.sample(minibatch_size)
    d_real_data = torch.from_numpy(z).float()
    if cuda:
        d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)

    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp)).float()
    if cuda:
        d_gen_input = d_gen_input.to(device)
    
    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)
    
    d_loss = (d_real_error + d_fake_error)/2
    d_loss.backward(create_graph=True)
    d_optimizer.step()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()

########################################################################################################################
def g_loop(iteration):
    g_optimizer.zero_grad()
    e_optimizer.zero_grad()
    
    z = noise_sampler(minibatch_size, g_inp)
    
    gen_input = torch.from_numpy(z).float()
    if cuda: 
        gen_input = gen_input.to(device)
        
    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            d_unrolled_loop(d_gen_input=gen_input)
    
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.to(device)
    g_adv_error = criterion(dg_fake_decision, target)
        

    if iteration <= half_iter:
        z1 = z + np.random.normal(0, 0.01, (minibatch_size, g_inp))
        z2 = z + np.random.normal(0, 0.01, (minibatch_size, g_inp))
    
        z1 = torch.from_numpy(z1).float().to(device)
        z2 = torch.from_numpy(z2).float().to(device)
    
        fake1 = G(z1)
        fake2 = G(z2)
        
        rep1 = E(fake1)
        rep2 = E(fake2)
    
        #g_error = g_error + contrastive_loss(rep1, rep2)
        g_cont_error = contrastive_loss(rep1, rep2)
        g_error = g_adv_error + g_cont_error
        #g_cont_error.backward()
        g_error.backward()
        g_optimizer.step()
        e_optimizer.step()
        
        return g_adv_error.cpu().item(), g_cont_error.cpu().item()
    
    else:
        g_adv_error.backward()
        g_optimizer.step()
        
        return g_adv_error.cpu().item(), 0
        
    if unrolled_steps > 0:
        D.load(backup)    
        del backup

########################################################################################################################
def g_sample():
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        if cuda:
            gen_input = gen_input.to(device)
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()

########################################################################################################################
print("="*50)
samples = []
for it in range(num_iterations): #tqdm_notebook(range(num_iterations)):
    d_infos = []
    for d_index in range(d_steps):
        d_info = d_loop((it+1))
        d_infos.append(d_info)
    d_infos = np.mean(d_infos, 0)
    d_real_loss, d_fake_loss = d_infos
    
    #for p in D.parameters():
    #    p.data.clamp_(-0.01, 0.01)
    
    g_infos = []
    for g_index in range(g_steps):
        g_info = g_loop((it+1))
        g_infos.append(g_info)
    g_infos = np.mean(g_infos, 0)
    g_adv_loss, g_cont_loss = g_infos
    
    if (it+1) % log_interval == 0:
        g_fake_data = g_sample()
        samples.append(g_fake_data)
        plot(g_fake_data, title='[{}] Iteration {}'.format(prefix, (it+1)))
        print("[Iteration:%d]" % (it+1), end=" ")
        print(d_real_loss, d_fake_loss, g_adv_loss, g_cont_loss)

########################################################################################################################
# plot the samples through iterations
def plot_samples(samples):
    xmax = 5
    cols = len(samples)
    bg_color  = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2*cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i+1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap='Greens', n_levels=20, clip=[[-xmax,xmax]]*2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d'%(i*log_interval))
    
    ax.set_ylabel('%d unrolling steps'% unrolled_steps)
    plt.gcf().tight_layout()
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()

########################################################################################################################
plot_samples(samples)






























