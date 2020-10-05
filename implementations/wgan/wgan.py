import argparse
import os
import numpy as np
import math
import sys

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
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，而不需要处理损失，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        global_config.data_root,
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化log记录器
writer = SummaryWriter(log_dir=os.path.join(global_config.log_root,"log"))
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z =Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sampled_imgs = generator(fixed_z)  # 固定tensor采样出的随着训练进行的图片变化，方便观察
            save_image(sampled_imgs.data[:25],
                       os.path.join(global_config.generated_image_root, "%d.png" % batches_done), nrow=5,
                       normalize=True)
            writer.add_scalar("loss/G_loss", loss_G.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
            writer.add_scalar("loss/D_loss", loss_D.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
            writer.add_scalars("loss/loss", {"g_loss": loss_G.item(), "d_loss": loss_D.item()},
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
torch.save(state,
           os.path.join(global_config.checkpoint_root, "%05d_ckpt.pth" % epoch))  # 保存checkpoint的时候用字典，方便恢复
torch.save(generator.state_dict(), os.path.join(global_config.pretrained_generator_root,
                                                "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片
