import torch
import os
import re
import numpy as np
from torchvision.utils import save_image

def generate_sequence(index:int,latent_dim:int,dataset,row=5,col=5):
    """
    从训练中某次编号的文件夹下的所有保存的生成器处使用相同的latent code生成图片
    :param index: 训练的次数
    :param latent_dim: 隐变量的维度
    :param dataset: 数据集名字，用于构建文件名
    :return: 无
    """
    # 生成固定的latent code，一边观看模型对同一个latent code生成图像的变化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = torch.tensor(np.random.normal(0, 1, (row*col, latent_dim)),dtype=torch.float).to(device)

    # 构造来源文件序列
    result_root = r"results/"
    ckpt_root = os.path.join(result_root,"{:05d}-{}".format(index,dataset))
    files = os.listdir(ckpt_root)
    for file in files:
        if(re.match(".*_ckpt_g.pth",file)):
            # 调用每个存储的带模型和参数的生成器预测图片
            G = torch.load(os.path.join(ckpt_root,file)).to(device)
            imgs:torch.Tensor = G(z)
            save_image(imgs.data,os.path.join(result_root,"same_latent",file[:-4]+".png"))
            print("Finished " + os.path.join(result_root,"same_latent",file[:-4]+".png"))
    print("Finished!")


if __name__=="__main__":
    generate_sequence(1,100,"minst")
