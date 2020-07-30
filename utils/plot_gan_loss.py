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

# 从tensorboard中加载日志数据
workbook = xlrd.open_workbook(r"PytorchGAN/gan/loss.xlsx")
worksheet=workbook.sheet_by_name("Sheet1")
iteration = worksheet.col_values(0,start_rowx=1,end_rowx=470)
g_loss = worksheet.col_values(1, start_rowx=1, end_rowx=470)
d_loss = worksheet.col_values(2, start_rowx=1, end_rowx=470)

smooth_point_num = 4700 # 插值法需要有一些中间点，总共的点数，扩充x轴

iteration_interpolate = x = np.linspace(0, 187200, num=smooth_point_num, endpoint=True)
#实现函数
func1 = interpolate.interp1d(iteration, smooth(g_loss, 0.8), kind=5)
func2 = interpolate.interp1d(iteration, smooth(d_loss, 0.8), kind=5)

# 画上下分开的两张图
matplotlib.rcParams.update({'font.size': 16}) #更改字体大小
fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(16,10))
# axes[0].set_ylim(1.1,1.4)
# axes[1].set_ylim(-2.9,-2.6)
axes[0].plot(iteration, g_loss, '+', label="g_loss")
axes[1].plot(iteration, d_loss, '+', label="d_loss")
axes[0].plot(iteration_interpolate,func1(iteration_interpolate),linewidth=2,label="g_loss")
axes[1].plot(iteration_interpolate,func2(iteration_interpolate),linewidth=2,label="d_loss")
axes[0].set(xlabel="iter",ylabel="g_loss")
axes[1].set(xlabel="iter",ylabel="d_loss")
# plt.title("Loss of Architecture 4",fontsize=12)
# plt.legend()
plt.show()
