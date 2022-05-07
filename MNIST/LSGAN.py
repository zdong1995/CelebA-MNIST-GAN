import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
from torchvision.utils import save_image

sample_dir = 'LSGAN_samples'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0

class ChunkSampler(sampler.Sampler): # define chunksampler
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

train_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)

train_data = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0))

val_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)

val_data = DataLoader(val_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

def discriminator():
    net = nn.Sequential(        
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    return net

def generator(noise_dim=NOISE_DIM):   
    net = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net

# use least square loss to replace bce_loss as loss function

def ls_discriminator_loss(scores_real, scores_fake):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss

def ls_generator_loss(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss

# use adam optimizer，learning rate 3e-4, beta1 = 0.5, beta2 = 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=1000, 
                noise_size=96, num_epochs=1000):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # Discriminator
            real_data = Variable(x).view(bs, -1).cuda() # real data
            logits_real = D_net(real_data) # D score
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 uniform distribution
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # generate fake data
            logits_fake = D_net(fake_images) # D score

            d_total_error = discriminator_loss(logits_real, logits_fake) # D loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # optimize Discriminator
            
            # Generator
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # generate fake data

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # G loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # optimize Generator

            if (iter_count % show_every == 0):
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}' 
                  .format(epoch, num_epochs, iter_count+1, 5000, d_total_error.data.item(), g_error.data.item()))
            writer.add_scalar('Loss/D', d_total_error.data.item())
            iter_count += 1

        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(fake_images, os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, ls_discriminator_loss, ls_generator_loss)