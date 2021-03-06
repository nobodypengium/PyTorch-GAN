import argparse
import os
import sys

import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

sys.path.append("../..")
from implementations import global_config

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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
# 添加预读取模型相关参数：保存模型的间隔与是否预训练模型
parser.add_argument("--generator_interval",type=int,default=20,help="interval between saving generators, epoch")
# 若启用多卡训练
parser.add_argument("--gpus",type=str,default=None,help="gpus you want to use, e.g. \"0,1\"")
opt = parser.parse_args()
print(opt)

if opt.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 设置可见GPU
    print("Now using gpu " + opt.gpus +" for training...")

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 为了逐渐学习细节，GAN一般从大的框架学起，逐渐上采样增加细节。本代码生成32x32的图片
        self.init_size = opt.img_size // 4 # 本文从生成图像四倍缩小的图像开始学起，8x8，然后逐渐Upsample添加细节
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)) # 隐变量大小本文设置100，图片越大蕴含信息越多就需要更大的隐变量，需要将隐变量首先规整到期望的8x8小图大小，所以先经过线性层获得8x8=86哥数，再规整成图片大小

        self.conv_blocks = nn.Sequential( #Upsample -> Conv2d -> BN
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

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv2d+LeakyReLU+BN的基本判别器块，分辨率逐渐缩小，信息逐渐浓缩的过程
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential( # 这里进行了4次降采样，缩放2倍
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image 根据缩放倍数和图像大小，可以计算出降采样后的维度
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1) # 整理成1维变量，以输入最后的线性层
        validity = self.adv_layer(out)

        return validity


# !!! Minimizes MSE instead of BCE 使用均方差代替交叉熵（拔掉log）
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        global_config.data_root,
        train=True,
        download=False,
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
writer = SummaryWriter(log_dir=global_config.log_root)
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths # 使用MSE损失函数，真实图片标答统一为1，虚假图片标答统一为0，不存在论文中的中间c
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

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

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step() # 固定D训练G，只更新G的单数

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step() # 固定G训练D，只更新D的参数

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sampled_imgs = generator(fixed_z)  # 固定tensor采样出的随着训练进行的图片变化，方便观察
            save_image(sampled_imgs.data[:25], os.path.join(global_config.generated_image_root,"%d.png" % batches_done), nrow=5, normalize=True)
            writer.add_scalar("loss/G_loss", g_loss.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
            writer.add_scalar("loss/D_loss", d_loss.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
            writer.add_scalars("loss/loss", {"g_loss":g_loss.item(),"d_loss":d_loss.item()}, global_step=batches_done)  # 横轴iter纵轴D_loss

    if epoch % opt.generator_interval == 0:
        # 保存生成器
        torch.save(generator.state_dict(),os.path.join(global_config.pretrained_generator_root,"%05d_ckpt_g.pth"%epoch)) # 只保存一个生成器

# 最后再保存一遍所有信息
# 定义所有需要保存并加载的参数，以字典的形式
state = {
    'epoch': epoch,
    'G_state_dict': generator.module.state_dict(),
    'D_state_dict': discriminator.module.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_D': optimizer_D.state_dict(),
}
torch.save(state,os.path.join(global_config.checkpoint_root,"%05d_ckpt.pth"%epoch)) # 保存checkpoint的时候用字典，方便恢复
torch.save(generator.state_dict(), os.path.join(global_config.pretrained_generator_root, "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片

