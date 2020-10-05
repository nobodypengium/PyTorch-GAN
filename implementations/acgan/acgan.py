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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--sample_rows", type=int, default=10, help="rows for sampled images")
# 添加预读取模型相关参数
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
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 8  # Initial size before upsampling 4*4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 256 * self.init_size ** 2))

        # 进行忠实于原文的更改，使用反卷积层（注意本repo原本使用的是Upsample+Conv2d），参见原文4.Result前5行
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 转置卷积，带有学习特性和上采样功能，增加的像素通过学习获得
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise) # 原文没说如何混合label和noise信息，github实现中两种方法都有
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss() # GAN损失，用于判别真伪
auxiliary_loss = torch.nn.CrossEntropyLoss() # 交叉熵损失，用于判别分类

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# 若多卡运算，将数据放到两个GPU上，注意是对nn.Model处理，而不需要处理损失，实际上在forward的时候只接收一半的数据
if opt.gpus is not None:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
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

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done,z=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    if z is None:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join(global_config.generated_image_root,"%d.png" % batches_done), nrow=n_row, normalize=True)

# 初始化log记录器
writer = SummaryWriter(log_dir=global_config.log_root)
# 使用一个固定的latent code来生成图片，更易观察模型的演化
fixed_z = Variable(FloatTensor(np.random.normal(0, 1, (opt.sample_rows ** 2, opt.latent_dim))))

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss_adv=adversarial_loss(validity, valid)
        g_loss_aux=auxiliary_loss(pred_label, gen_labels)
        g_loss = 0.5 * (g_loss_adv + g_loss_aux)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss_adv = adversarial_loss(real_pred, valid)
        d_real_loss_aux = auxiliary_loss(real_aux, labels)
        d_real_loss = (d_real_loss_adv + d_real_loss_aux) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss_adv=adversarial_loss(fake_pred, fake)
        d_fake_loss_aux=auxiliary_loss(fake_aux, gen_labels)
        d_fake_loss = (d_fake_loss_adv + d_fake_loss_aux) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt) #argmax在分类维度上找出概率最大的分类，看有多少个相等的，并除总数，得到准确率（Top-1 Acc）
        d_real_acc = np.mean(np.argmax(real_aux.data.cpu().numpy(),axis=1) == labels.data.cpu().numpy()) #在真实数据集上的Top-1 Acc
        d_fake_acc = np.mean(np.argmax(fake_aux.data.cpu().numpy(),axis=1) == gen_labels.data.cpu().numpy()) #在生成数据集上的Top-1 Acc

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        writer.add_scalar("loss/G_loss", g_loss.item(), global_step=batches_done)  # 横轴iter纵轴G_loss
        writer.add_scalar("loss/D_loss", d_loss.item(), global_step=batches_done)  # 横轴iter纵轴D_loss
        writer.add_scalars("loss/loss", {"g_loss": g_loss.item(), "d_loss": d_loss.item()},
                           global_step=batches_done)  # 两个loss画在一张图里
        writer.add_scalar("acc/real_acc",d_real_acc,global_step=batches_done) # 辅助分类器的准确率
        writer.add_scalar("acc/fake_acc",d_fake_acc,global_step=batches_done)
        writer.add_scalar("acc/all_acc",d_acc,global_step=batches_done)
        writer.add_scalars("acc/acc",{"real":d_real_acc,"fake":d_fake_acc,"all":d_acc},global_step=batches_done)
        writer.add_scalars("loss_contribute/d_real_loss",{"d_real_loss":d_real_loss.item()/2,"d_real_loss_adv":d_real_loss_adv.item()/2,"d_real_loss_aux":d_real_loss_aux.item()/2},global_step=batches_done) # /2由于总D损失是由real和fake相加/2的，所以这里记录也/2
        writer.add_scalars("loss_contribute/d_fake_loss",{"d_fake_loss":d_fake_loss.item()/2,"d_fake_loss_adv":d_fake_loss_adv.item()/2,"d_fake_loss_aux":d_fake_loss_aux.item()/2},global_step=batches_done)
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=opt.sample_rows, batches_done=batches_done, z=fixed_z)

    if epoch % opt.generator_interval == 0:
        # 保存生成器
        torch.save(generator.state_dict(),
                   os.path.join(global_config.pretrained_generator_root, "%05d_ckpt_g.pth" % epoch))  # 只保存一个生成器

    # 最后再保存一遍所有信息
    # 多卡训练情况下，需要从嵌套类DataParallel中取出
    if opt.gpus is not None:
        generator_module=generator.module
        discriminator_module=discriminator.module
    else:
        generator_module=generator
        discriminator_module = discriminator

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