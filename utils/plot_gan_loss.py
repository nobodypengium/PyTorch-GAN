import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xlrd
from scipy import interpolate

def smooth(scalar,weight=0.85):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_losses(path,sheet,col):
    """
    从xlsx中读取loss，返回迭代次数（横轴），以及对应的生成器损失和判别器损失
    :param path: xlsx文件路径
    :param sheet: 工作簿名称，如"Sheet1"
    :param col: 起止行数，如 [1,470]
    :returns: 迭代数，生成器损失，判别器损失
    """
    # 从tensorboard中加载日志数据
    workbook = xlrd.open_workbook(path)
    worksheet=workbook.sheet_by_name(sheet)
    iteration = worksheet.col_values(0,start_rowx=col[0],end_rowx=col[-1])
    g_loss = worksheet.col_values(1, start_rowx=col[0], end_rowx=col[-1])
    d_loss = worksheet.col_values(2, start_rowx=col[0], end_rowx=col[-1])
    return iteration,g_loss,d_loss

def smooth_loss(iteration:list,g_loss,d_loss,weight=0.8,smooth_point_num=None):
    """
    接收迭代次数（x轴），生成器判别器损失（两个y轴数据），获得扩展点个数的x轴和拟合的损失函数
    :param iteration: 插值前的x轴，原始数据
    :param g_loss: 插值前的生成器损失，原始数据
    :param d_loss: 插值前的判别器损失，原始数据
    :param smooth_point_num: 总共插值点数，默认原始数据的100倍
    :return: iteration_interpolate:插值后的x轴, g_interpolate_func:用于插值g的拟合函数, d_interpolate_func:用于插值d的拟合函数
    """
    if smooth_point_num is None:
        smooth_point_num = len(iteration)*100  # 插值法需要有一些中间点，总共的点数，扩充x轴
    iteration_interpolate = np.linspace(iteration[0], iteration[-1], num=smooth_point_num, endpoint=True)
    # 实现函数
    g_interpolate_func = interpolate.interp1d(iteration, smooth(g_loss, 0.8), kind=1)
    d_interpolate_func = interpolate.interp1d(iteration, smooth(d_loss, 0.8), kind=1)
    return iteration_interpolate, g_interpolate_func, d_interpolate_func

def draw_separate_loss():
    # 读取数据
    iteration, g_loss, d_loss = load_losses(path=r"PytorchGAN/dcgan/1e-5/loss.xlsx", sheet="Sheet1", col=[1, 470])
    # 平滑数据
    iteration_interpolate, g_interpolate_func, d_interpolate_func = smooth_loss(iteration, g_loss, d_loss)
    # 画上下分开的两张图
    matplotlib.rcParams.update({'font.size': 16})  # 更改字体大小
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    # axes[0].set_ylim(1.1,1.4)
    # axes[1].set_ylim(-2.9,-2.6)
    axes[0].plot(iteration, g_loss, '+', label="g_loss")
    axes[1].plot(iteration, d_loss, '+', label="d_loss")
    axes[0].plot(iteration_interpolate, g_interpolate_func(iteration_interpolate), linewidth=2, label="g_loss")
    axes[1].plot(iteration_interpolate, d_interpolate_func(iteration_interpolate), linewidth=2, label="d_loss")
    axes[0].set(xlabel="iter", ylabel="g_loss")
    axes[1].set(xlabel="iter", ylabel="d_loss")
    # plt.title("Loss of Architecture 4",fontsize=12)
    # plt.legend()
    plt.show()

def draw_together_loss():
    # 读取数据
    iteration, g_loss, d_loss = load_losses(path=r"PytorchGAN/dcgan/1e-5/loss.xlsx", sheet="Sheet1", col=[1, 468])
    # 平滑数据
    iteration_interpolate, g_interpolate_func, d_interpolate_func = smooth_loss(iteration, g_loss, d_loss)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(iteration, g_loss, '+', label="g_loss", color="#9C9B7A")
    ax.plot(iteration, d_loss, '*', label="d_loss", color="#FFD393")
    ax.plot(iteration_interpolate, g_interpolate_func(iteration_interpolate), linewidth=2, label="g_loss",
            color="#405952")
    ax.plot(iteration_interpolate, d_interpolate_func(iteration_interpolate), linewidth=2, label="d_loss",
            color="#FF974F")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_separate_loss()
