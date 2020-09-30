"""
EBGAN生成器与传统GAN相同(本代码采用改进版的DCGAN)。判别器变为一个AutoEncoder(AE)，首先降采样图片，再升采样图片，最后比较升采样后的图片与输入图片的差距，或者说，判别器采用瓶颈结构(结构图长得像花瓶，两头粗中间细)。
"""
import argparse
import os
import numpy as np
import math

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
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
# 添加预读取模型相关参数
parser.add_argument("--generator_interval",type=int,default=20,help="interval between saving generators, epoch")
# 若启用多卡训练
parser.add_argument("--gpus",type=str,default=None,help="gpus you want to use, e.g. \"0,1\"")
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)

if opt.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 设置可见GPU
    print("Now using gpu " + opt.gpus +" for training...")

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding


# Reconstruction loss of AE
pixelwise_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，而不需要处理损失，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Configure data loader
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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化log记录器
writer = SummaryWriter(log_dir=os.path.join(global_config.log_root,"log"))
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z =Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt


# ----------
#  Training
# ----------

# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, opt.batch_size / 64.0)

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        recon_imgs, img_embeddings = discriminator(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_recon, _ = discriminator(real_imgs)
        fake_recon, _ = discriminator(gen_imgs.detach())

        d_loss_real = pixelwise_loss(real_recon, real_imgs)
        d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

        d_loss = d_loss_real
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += margin - d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sampled_imgs = generator(fixed_z)  # 固定tensor采样出的随着训练进行的图片变化，方便观察
            save_image(sampled_imgs.data[:25],
                       os.path.join(global_config.generated_image_root, "%d.png" % batches_done), nrow=5,
                       normalize=True)
            writer.add_scalar("loss/G_loss", g_loss.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
            writer.add_scalar("loss/D_loss", d_loss.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
            writer.add_scalars("loss/loss", {"g_loss": g_loss.item(), "d_loss": d_loss.item()},
                               global_step=batches_done)  # 两个loss画在一张图里

    if epoch % opt.generator_interval == 0:
        # 保存生成器
        torch.save(generator.state_dict(),
                   os.path.join(global_config.pretrained_generator_root, "%05d_ckpt_g.pth" % epoch))  # 只保存一个生成器

    # 最后再保存一遍所有信息
    # 定义所有需要保存并加载的参数，以字典的形式
    state = {
        'epoch': epoch,
        'G_state_dict': generator.module.state_dict(),
        'D_state_dict': discriminator.module.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
    }
    torch.save(state, os.path.join(global_config.checkpoint_root, "%05d_ckpt.pth" % epoch))  # 保存checkpoint的时候用字典，方便恢复
    torch.save(generator.state_dict(), os.path.join(global_config.pretrained_generator_root,
                                                    "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片
