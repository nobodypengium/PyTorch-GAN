import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter # 用于记录训练信息
import sys # 用于引入上一层的py文件，如果用pycharm执行没有问题，但是如果直接python *.py会找不到文件
sys.path.append("../..")
from implementations import global_config

# 根据上层定义好的全局数据构建结果文件夹，所有GAN都使用这种结构
os.makedirs(global_config.generated_image_root, exist_ok=True)
os.makedirs(global_config.checkpoint_root, exist_ok=True)
os.makedirs(global_config.pretrained_generator_root, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# 添加预读取模型相关参数
parser.add_argument("--generator_interval",type=int,default=20,help="interval between saving generators, epoch")
# 若启用多卡训练
parser.add_argument("--gpus",type=str,default=None,help="gpus you want to use, e.g. \"0,1\"")
opt = parser.parse_args()
print(opt)

if opt.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 设置可见GPU
    print("Now using gpu " + opt.gpus +" for training...")

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，而不需要处理损失，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    discriminator = nn.DataParallel(discriminator)

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        global_config.data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化log记录器
writer = SummaryWriter(log_dir=os.path.join(global_config.log_root,"log"))
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z =Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sampled_imgs = decoder(fixed_z)  # 固定tensor采样出的随着训练进行的图片变化，方便观察
            save_image(sampled_imgs.data[:25],
                       os.path.join(global_config.generated_image_root, "%d.png" % batches_done), nrow=5,
                       normalize=True)
            writer.add_scalar("loss/G_loss", g_loss.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
            writer.add_scalar("loss/D_loss", d_loss.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
            writer.add_scalars("loss/loss", {"g_loss": g_loss.item(), "d_loss": d_loss.item()},
                               global_step=batches_done)  # 两个loss画在一张图里

    if epoch % opt.generator_interval == 0:
        # 保存生成器（解码器）
        torch.save(decoder.state_dict(),
                   os.path.join(global_config.pretrained_generator_root, "%05d_ckpt_decoder.pth" % epoch))  # 只保存一个生成器

# 最后再保存一遍所有信息
# 定义所有需要保存并加载的参数，以字典的形式
state = {
    'epoch': epoch,
    'encoder_state_dict': encoder.module.state_dict(),
    'decoder_state_dict': decoder.module.state_dict(),
    'discriminator_state_dict': discriminator.module.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_D': optimizer_D.state_dict(),
}
torch.save(state, os.path.join(global_config.checkpoint_root, "%05d_ckpt.pth" % epoch))  # 保存checkpoint的时候用字典，方便恢复
torch.save(decoder.state_dict(), os.path.join(global_config.pretrained_generator_root,
                                                "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片