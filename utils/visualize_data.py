from torchvision import datasets, transforms, utils
from torch.utils import data
import torch
import matplotlib.pyplot as plt

import implementations.global_config as global_config

def sample_images(dataset: data.Dataset, row=4, col=8, label=True):
    """
    输入dataset格式数据集，绘制是否带label的网格图片图
    :param dataset: 继承自data.Dataset格式的数据集
    :param row:网格图行数
    :param col:网格图列数
    :param label:图中是否带有标记
    :return:
    """
    data_loader = data.DataLoader(dataset, batch_size=col, shuffle=True) # 构造可以生成小批数据的读取器
    cnt = 1  # 指示输出第几个位置的图片
    for i_batch, (imgs, labels) in enumerate(data_loader):
        for img, label in zip(imgs,labels):
            plt.subplot(row, col, cnt)
            cnt = cnt + 1
            plt.imshow(img.numpy().transpose((1, 2, 0)).squeeze(2),cmap="gray")  # 注意转换
            plt.title(label.numpy(),color='blue')
        if i_batch >= row-1:
            break
    plt.show()

if __name__=='__main__':
    my_trans = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #构建一种变换“套路”，对每张图片应用这种套路
    train_data = datasets.MNIST(global_config.data_root, train=True, download=False, transform=my_trans) # 通过torchvision中自带的类或其他方法，将数据集转化为torch.utils.data.Dataset类
    sample_images(train_data)