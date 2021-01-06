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

if opt.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 设置可见GPU
    print("Now using gpu " + opt.gpus +" for training...")

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# 此处为了跟EBGAN/Upsample版DCGAN比较，遵循本实验中EBGAN的初始化设置
# 另外Pytorch也可以不使用自定义初始化方法，请参考Conv2d类继承的_ConvNd类中的reset_parameters方法
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

        self.init_size = opt.img_size // 4 # 32//4=8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)) # 62->128*8*8

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 128*8*8 -> 128*16*16
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # 128*16*16 -> 128*16*16
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # 128*16*16 -> 128*32*32
            nn.Conv2d(128, 64, 3, stride=1, padding=1), # 128*16*16 -> 64*32*32
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), #64*32*32 -> 3*32*32
            nn.Tanh(), # 转换为[-1,1]，即转换到方便保存的RGB色彩
        )

    def forward(self, noise):
        out = self.l1(noise) # (N,62)->(N,128*8*8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) # (N,128*8*8) -> (N,128,8,8)
        img = self.conv_blocks(out)# 2组Upsample+Conv2d+LeakyRelu 1组Conv2d+Tanh转换色彩 (N,128*8*8) -> (N,3,32,32)
        return img

# 判别器架构为Encoder-Decoder(AE,AutoEncoder) 架构，遵循 [输入图片] - 下采样网络 - [编码] - 上采样网络 - [输出图片] 的过程
# 其中，下采样网络经过[Conv]+[FC(Linear)]（仅考虑参数层）得到特征码
# 上采样网络经由[FC(Linear)] + [Upsample+Conv]得到重建图片
# 之后损失函数将使用本class定义的Encoder-Decoder的输入图片和输出图片构造重建损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling--------------------------------------------------------------------------------------
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU()) #                    ↑
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2 #                                                    Encoder
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8), #                                                                  ↓
            nn.ReLU(inplace=True),#-----------------------------------------------------------------------Relu层之后为中间编码
            nn.Linear(32, down_dim),#                                                                   ↑
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        ) #                                                                                           Decoder
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1)) #    ↓
        #-------------------------------------------------------------------------------------------------
    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out


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

# Initialize weights 初始化权重，将在 implementations/dcgan/dcgan_upsampleconv_g.py 中测试这种权重初始化方法与默认初始化方法(kaiming_uniform)比较
# 此处为了跟EBGAN/Upsample版DCGAN比较，遵循本实验中EBGAN的初始化设置
# 另外Pytorch也可以不使用自定义初始化方法，请参考Conv2d类继承的_ConvNd类中的reset_parameters方法
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader 从下载好的数据集中构建DataLoader
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

# Optimizers 固定学习率Adam优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化log记录器
writer = SummaryWriter(log_dir=os.path.join(global_config.log_root,"log"))
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z =Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

# ----------
#  Training
# ----------

# BEGAN hyper parameters 控制对正负样本惩罚的力度差距不要太大，动态调整正负样本贡献损失的量，使得生成器不要过于关注生成劣质的负样本，也要生成好的正样本
gamma = 0.75 # E(L(G(z))]/E[L(x)] 控制对正负样本惩罚的力度比例
lambda_k = 0.001 # 类似学习率λ的概念，控制每次根据γ调整多少
k = 0.0 # 初始k，直接控制L(G(z)贡献的量

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input 真实图片
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images 生成图片
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator WGAN的Gloss，G的L1距离
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples 获得真假图片的重建图片
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        # 计算真假图的重建损失，最终loss为两分布间的WGAN损失
        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake # 与EBGAN区别：在loss上不做显式的1-Lip限制

        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights 更新BEGAN中生成图贡献的损失的权值
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).data[0]

        # --------------
        # Log Progress 记录指标
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
            writer.add_scalar("loss/G_loss", d_loss.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
            writer.add_scalar("loss/D_loss", g_loss.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
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
    torch.save(state,
               os.path.join(global_config.checkpoint_root, "%05d_ckpt.pth" % epoch))  # 保存checkpoint的时候用字典，方便恢复
    torch.save(generator.state_dict(), os.path.join(global_config.pretrained_generator_root,
                                                    "%05d_ckpt_g.pth" % epoch))  # 只保存一个带有模型信息和参数的生成器，用于后续生成图片
