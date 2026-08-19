from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

fig, axes = plt.subplots(1,3)
x=[[40,45,50,55,60,65,70,75,80],
   [6,7,8,9,10,11,12,13,14],
   [80,90,100,110,120,130,140]]
y=[[-0.7591,-0.8330,-0.9017,-0.9316,-0.9797,-1.0438,-1.1204,-1.1816,-1.2401],
   [-1.8099,-1.6233,-1.4059,-1.1890,-0.9797,-0.8880,-0.8008,-0.7105,-0.6068],
   [-1.2894,-1.1314,-0.9797,-0.8886,-0.8165,-0.7415,-0.6725]]
x1=[[40,80],[6,14],[80,140]]
y1=[[-0.7591,-1.2401],[-1.8099,-0.6068],[-1.2894,-0.6725]]

x[1]=x[1][::-1]
x[2]=x[2][::-1]
y[1]=y[1][::-1]
y[2]=y[2][::-1]
x1[1]=x1[1][::-1]
x1[2]=x1[2][::-1]
y1[1]=y1[1][::-1]
y1[2]=y1[2][::-1]


print(x[1][1])
title=['a','b','c']
xlabel=[r'(a) $d_{max}$',r'(b) $u_0$',r'(c) $c$']

for i in range(3):
    x[i]=[str(x[i][j]) for j in range(len(x[i]))]
    x1[i] = [str(x1[i][j]) for j in range(len(x1[i]))]

for i in range(3):
    axes[i].plot(x[i],y[i])
    axes[i].plot(x1[i], y1[i],linestyle=':')
    # axes[i].set_title(title[i])
    axes[i].set_xlabel(xlabel[i])
axes[0].set_ylabel(r'$worst-case~cost~(10^7)$')

plt.tight_layout()
plt.show()

