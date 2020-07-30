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
from tensorboardX import SummaryWriter

import sys
sys.path.append("..")
import global_config as global_config

# 接收输入参数部分，包括学习率、优化器参数、通道数、图像大小等
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples, iter")
# 添加预读取模型相关参数：保存模型的间隔与是否预训练模型
parser.add_argument("--ckpt_interval",type=int,default=10,help="interval between saving models, epoch")
parser.add_argument("--pretrained",type=int,default=None,help="load the checkpoint")
parser.add_argument("--epoch",type=int,default=None,help="the epoch you want to continue")
# 若启用多卡训练
parser.add_argument("--gpus",type=str,default=None,help="gpus you want to use, e.g. \"0,1\"")
opt = parser.parse_args()
print(opt)
print("Now using gpu " + opt.gpus +" for training...")

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True): # 构建基本构造块 Linear+BatchNorm+LeakyReLU，注意使用nn中的，因为nn中的类自带可训练参数
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False), # *意味对list解包，把每个list里的nn.Module拆出来，以便放进Sequential
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh() # 在PyTorch中，任何图片都要被表示到[-1,1]中
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
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
            nn.Sigmoid(), # 判别是0-1的，这里同样根据表达整理输出
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator 定义生成器判别器，实例化
generator = Generator()
discriminator = Discriminator()

# Optimizers 定义优化器，采用ADAM优化方法
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 若使用GPU，将数据都挪到GPU上，.cuda()等同于.to(device)
# ---------方式1-------------
# cuda = True if torch.cuda.is_available() else False # 决定是否使用GPU
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
# ---------方式2-------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# 如果在输出参数中指定需要继续某个checkpoint训练，则读取checkpoint中记录的字典，包括模型参数、优化器状态和epoch信息
start_epoch=0
if opt.pretrained is not None:
    # 根据输入parameter找到需要恢复的epoch的模型参数文件
    path = "results/%05d-minst"%opt.pretrained
    checkpoint_path = os.path.join(path,"%05d_ckpt.pth"%opt.epoch)
    # 把参数加载进实例化好的模型中
    checkpoint = torch.load(checkpoint_path,map_location=device)
    start_epoch = checkpoint["epoch"]+1
    generator.load_state_dict(checkpoint['G_state_dict'])
    discriminator.load_state_dict(checkpoint['D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    print("Load %05d epoch success!"%(start_epoch-1))
else:
    path_index=0 # 建议将所有的结果放到result文件夹，且以训练次数/日期等进行编号子文件夹，在子文件夹中条理清晰地记录图像、scalar等
    while(os.path.exists(r"results/%05d-minst"%path_index)):
        path_index = path_index+1
    path = "results/%05d-minst"%path_index
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path,"image"),exist_ok=False)
if opt.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 设置可见GPU

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)


# 生成MNIST的dataset并由其生成MNIST的dataloader
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

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化log记录器
writer = SummaryWriter(log_dir=os.path.join(path,"log"))
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z = torch.tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)),dtype=torch.float).to(device)


# ----------
#  训练过程
# ----------


for epoch in range(start_epoch,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths 创建标答：维度为(batch,1），valid为全1矩阵，fake为全0矩阵
        # valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        # fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)
        valid = torch.ones(size=(imgs.size(0),1),dtype=torch.float).to(device)
        fake = torch.zeros(size=(imgs.size(0),1),dtype=torch.float).to(device)

        # Configure input D的真实输入
        # real_imgs = Variable(imgs.type(Tensor))
        real_imgs = imgs.float().to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad() # 每个网络训练一步之前都要清空梯度

        # Sample noise as generator input 生成G的输入，隐向量
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)),dtype=torch.float).to(device)

        # Generate a batch of images 将z送入生成器，得到生成图片
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator 将判别器的输出与全0比较，期望判别器判别生成图片是假的，BCE loss
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward() # 先反向传播 （此处只训练G而不管D，没有对D进行任何操作）
        optimizer_G.step() # 再自动更新参数

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples D的损失由真实图片和生成图片组成，都是BCE
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ----------
        # 正式训练过程到此结束，下面开始打log
        # ----------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # 此处根据interval保存batch的图片
        if batches_done % opt.sample_interval == 0:
            sampled_imgs = generator(fixed_z)# 固定tensor采样出的随着训练进行的图片变化，方便观察
            save_image(sampled_imgs.data[:25], os.path.join(path,"image","%d.png" % batches_done), nrow=5, normalize=True)
            writer.add_scalar("G_loss",g_loss.item(),global_step=batches_done) # 横轴iter纵轴G_loss
            writer.add_scalar("D_loss",d_loss.item(),global_step=batches_done) # 横轴iter纵轴D_loss

    if epoch % opt.ckpt_interval ==0:
        # 定义所有需要保存并加载的参数，以字典的形式
        state = {
            'epoch': epoch,
            'G_state_dict': generator.module.state_dict(),
            'D_state_dict': discriminator.module.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }
        torch.save(state,os.path.join(path,"%05d_ckpt.pth"%epoch)) # 保存checkpoint的时候用字典，方便恢复
        torch.save(generator.state_dict(),os.path.join(path,"%05d_ckpt_g.pth"%epoch)) # 只保存一个生成器

# 最后再保存一遍所有信息
torch.save(state, os.path.join(path, "%05d_ckpt.pth" % epoch))  # 保存checkpoint的时候用字典，方便恢复
torch.save(generator, os.path.join(path, "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片