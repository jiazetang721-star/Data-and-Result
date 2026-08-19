from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

fig, axes = plt.subplots(2,3)
x=np.array([0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15])
df = pd.read_excel('data.xlsx',sheet_name='norm')  # 也可以读取 .xls 文件
y = df.to_numpy()
y=0.01*y
index=np.array([12*i for i in range(11)])

# 平均的用deterministic的，ro的用各自范数下表现最好的，也就是和最好的gap
ycopy=y.copy()
y+=1e-10
for j in range(3):
    ycopy[index + 4 * j, 0]=-(y[index + 4 * j, 0]-y[index+4*j,0])/y[index+4*j,0]
    ycopy[index + 4 * j +1, 0] = -(y[index + 4 * j +1, 0] - y[index + 4 * j, 0]) / y[index + 4 * j, 0]
    ycopy[index + 4 * j +2, 0] = -(y[index + 4 * j +2, 0] - y[index + 4 * j, 0]) / y[index + 4 * j, 0]
    ycopy[index + 4 * j +3, 0] = -(y[index + 4 * j +3, 0] - y[index + 4 * j, 0]) / y[index + 4 * j, 0]
    ycopy[index + 4 * j, 1]=-(y[index + 4 * j, 1]-y[index+4*j+j+1,1])/y[index+4*j+j+1,1]
    ycopy[index + 4 * j +1, 1] = -(y[index + 4 * j +1, 1] - y[index + 4 * j + j + 1, 1]) / y[index + 4 * j + j + 1, 1]
    ycopy[index + 4 * j +2, 1] = -(y[index + 4 * j +2, 1] - y[index + 4 * j + j + 1, 1]) / y[index + 4 * j + j + 1, 1]
    ycopy[index + 4 * j +3, 1] = -(y[index + 4 * j +3, 1] - y[index + 4 * j + j + 1, 1]) / y[index + 4 * j + j + 1, 1]

y=ycopy.copy()
# y=np.abs(y)

# skip=np.array([0,3,5,7,10])
# x=x[skip]
# index=np.array([12*i for i in skip])

for i in range(2):# average worst
    for j in range(3):# 1 2 inf norm
        if j==2 and i==1:# 0.14和0.15的时候inf为0，没办法除
            axes[i, j].plot(x[:9], y[index + 4 * j, i][:9], marker='.',label='deterministic')
            axes[i, j].plot(x[:9], y[index + 4 * j + 1, i][:9],marker='.', label='1-norm')
            axes[i, j].plot(x[:9], y[index + 4 * j + 2, i][:9], marker='.',label='2-norm')
            axes[i, j].plot(x[:9], y[index + 4 * j + 3, i][:9], marker='.',label='inf-norm')
            axes[i, j].legend()  # 在ax上添加图例
            axes[i, j].set_xticks(x[[0, 2, 4, 6, 8, 10]])
        else:
            axes[i,j].plot(x,y[index+4*j,i],marker='.',label='deterministic')
            axes[i,j].plot(x, y[index+4*j+1,i], marker='.',label='1-norm')
            axes[i,j].plot(x, y[index+4*j+2,i], marker='.',label='2-norm')
            axes[i,j].plot(x, y[index+4*j+3,i], marker='.',label='inf-norm')
            axes[i,j].legend()  # 在ax上添加图例
            axes[i,j].set_xticks(x[[0,2,4,6,8,10]])
            # axes[i,j].text(0,1.02, r'$\times10^6$', transform=axes[i,j].transAxes)



row_titles = ['Average', 'Worst']
for ax, title in zip(axes[:, 0], row_titles):
    ax.set_ylabel(title, rotation=0, size=10, labelpad=10, ha='right', va='center')

# 列标题（顶部）
col_titles = ['1-norm', '2-norm', 'inf-norm']
for ax, title in zip(axes[0, :], col_titles):
    ax.set_title(title, pad=10, size=10)

plt.tight_layout()
plt.show()