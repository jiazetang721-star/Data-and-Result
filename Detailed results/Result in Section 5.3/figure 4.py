# ownsetting 下的最优目标，由于顾客行为造成的系统性损失，说明企业可以考虑上门回收等集中调度，也可以反映出mechanism偏好，也就是在各自的设定下，最优决策呈现出什么特点
# DDP-98 DDN-98 DDC-10 DDC-15 DIN-98 DIC-10 DIC-15
# 命名规则："I-unique_num-variable_name"

import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from scipy.spatial.distance import pdist, squareform

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的地面距离（单位：公里）
    """
    R = 6371.0  # 地球平均半径，单位公里

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



x=np.array(range(5,16)).astype(str)
model=['DDP','(a) DDN','(b) DDC-10','(c) DDC-15','(d) DIN','(e) DIC-10','(f) DIC-15']
marker=['p','s','o','+','D','*','^']
jatter=np.array([-0.06,+0.06,-0.04,+0.04,0,-0.02,+0.02])
with open('DDP-98.pkl', 'rb') as f:
    DDP98= pickle.load(f)
with open('DDN-98.pkl', 'rb') as f:
    DDN98= pickle.load(f)
with open('DDC-10.pkl', 'rb') as f:
    DDC10 = pickle.load(f)
with open('DDC-15.pkl', 'rb') as f:
    DDC15 = pickle.load(f)
with open('DIN-98.pkl', 'rb') as f:
    DIN98= pickle.load(f)
with open('DIC-10.pkl', 'rb') as f:
    DIC10 = pickle.load(f)
with open('DIC-15.pkl', 'rb') as f:
    DIC15 = pickle.load(f)
with open('compare_x.pkl', 'rb') as f:
    x_record = pickle.load(f)
with open('compare_y.pkl', 'rb') as f:
    y_record = pickle.load(f)
with open('unique_solution.pkl', 'rb') as f:
    solution = pickle.load(f)
with open('unique_solution_index.pkl', 'rb') as f:
    index = pickle.load(f)
index=index.astype(int)
result=[DDP98,DDN98,DDC10,DDC15,DIN98,DIC10,DIC15]

########################################################### 一些目标值的计算
objective=np.zeros((11,7,7)) #总目标值，11表示I，每行是一个解，每列是一个环境
building_cost=np.zeros((11,7,7)) #一阶段建造成本
transportation_cost=np.zeros((11,7,7)) # 运输成本
penalty_cost=np.zeros((11,7,7)) # 惩罚成本
profit=np.zeros((11,7,7)) # 回收的收益
collected=np.zeros((11,7,7)) # 总共的回收量
utilization=np.zeros((11,7,7)) # 总的利用率

CC_num=np.zeros((11,7))
RF_num=np.zeros((11,7))
ACDCC=np.zeros((11,7))
ACDRF=np.zeros((11,7))
ANDCCRF=np.zeros((11,7)) #每个CC到最近RF的平均距离

# local_wise的目标只用在DDP环境下计算，不考虑其他环境下的测试了，就说明其他解不好在哪里
# 衡量均衡性一个方法是计算出整体的方差和极差，另一个方法就是绘制箱线图，每个指标一个图，7个子图表示不同的解，还是11个I表示不同的CC规模，信息量应该也不需要这么大
# 都用cv计算吧，因为整体水平在system-wise的时候已经考虑了一下了，这里主要看各设施之间运转是不是均衡，可能虽然都很低但是分配的比较均衡，同时极差也可以看出来绝对值的差异
recycling_rate_cv=np.zeros((11,7))
recycling_rate_ptp=np.zeros((11,7))
collected_cv=np.zeros((11,7))
collected_ptp=np.zeros((11,7))
utilization_cv=np.zeros((11,7))
utilization_ptp=np.zeros((11,7))

fixcollected=np.zeros((11,7)) #计算一下ownsetting下估计的collected是多少
DDP_fixcollected=np.zeros((11,7)) #计算一下DDP下估计的collected是多少

for i in range(11):
    testname = r'I=' + str(i+5)
    os.chdir(testname)
    I = int(np.load("I.npy"))
    K = int(np.load("K.npy"))
    J = int(np.load("J.npy"))
    fc = np.load("fc.npy")
    fr = np.load("fr.npy")
    d = np.load("d.npy")
    Capacity = np.load("Capacity.npy")
    u = np.load("u.npy")
    c = np.load("c.npy")
    r = np.load("r.npy")
    fp = np.load("fp.npy")
    CC_point=np.load("CC_point.npy")
    RF_point = np.load("RF_point.npy")
    theta=np.load(("theta.npy"))

    for j in range(7): # 表示7个解
        fixx=solution[i][index[i,j]][:i+5]
        fixy = solution[i][index[i, j]][i + 5:]
        CC_num[i,j]=fixx.sum()
        RF_num[i, j] = fixy.sum()
        CCones=CC_point[np.where(fixx>=1e-3)[0]]
        RFones = RF_point[np.where(fixy >= 1e-3)[0]]

        fixv = 1 / (u @ np.append(fixx, 1))  # K维向量 回收点分母
        fixomega = fixv.reshape(-1, 1) * (np.append(fixx, 1))  # K*I+1
        fixp = u * fixomega
        DDP_fixcollected[i, j] = d @ (1 - fixp[:, -1])


        if j == 0:
            fixv = 1 / (u @ np.append(fixx, 1))  # K维向量 回收点分母
            fixomega = fixv.reshape(-1, 1) * (np.append(fixx, 1))  # K*I+1
            fixp = u * fixomega
            fixcollected[i, j] = d@(1-fixp[:,-1])
        elif j in {1,2,3}:  # 在DDP的基础上求和，p只是回收的比率
            fixv = 1 / (u @ np.append(fixx, 1))  # K维向量 回收点分母
            fixomega = fixv.reshape(-1, 1) * (np.append(fixx, 1))  # K*I+1
            fixp = u * fixomega
            fixp = fixp[:, :-1].sum(axis=1)
            fixcollected[i, j] = d @ fixp
        elif j in {4,5,6}:  # 回收量是一个给定值，比如回收20%
            fixp = theta * np.ones(K)
            fixcollected[i, j] = d @ fixp


        for k in range(7): # 表示7个环境
            z = result[k][str(i + 5) + '-' + str(index[i, j]) + '-' + 'z'].reshape((i + 5,J))
            h = result[k][str(i + 5) + '-' + str(index[i, j]) + '-' + 'h']
            objective[i,j,k]=fixx@fc+fixy@fr+(c*z).sum()-(r*z.sum(axis=0)).sum()+fp@h
            building_cost[i,j,k] = fixx@fc+fixy@fr  # 一阶段建造成本
            transportation_cost[i,j,k] = (c*z).sum()  # 运输成本
            penalty_cost[i,j,k] = fp@h  # 惩罚成本
            profit[i,j,k] = (r*z.sum(axis=0)).sum() # 回收的收益
            collected[i,j,k]=z.sum()+h.sum() #总共回收回来的量
            utilization[i,j,k]=z.sum()/(Capacity*fixy).sum() #总产能利用率

            if k==0: # 表示在DDP环境下，其他环境下的均衡与否先不考虑了
                p=result[k][str(i+5)+'-'+str(index[i,j])+'-'+'p['].reshape((K,i+5+1))
                recycling_rate = 1-p[:,-1]
                recycling_rate_cv[i,j]=np.std(recycling_rate)/np.mean(recycling_rate)
                recycling_rate_ptp[i,j]=np.ptp(recycling_rate)
                collected_amount=z.sum(axis=1)+h
                collected_amount = collected_amount[fixx>=1e-3] #只考虑建立了的
                collected_cv[i,j]=np.std(collected_amount)/np.mean(collected_amount)
                collected_ptp[i,j]=np.ptp(collected_amount)
                processed=z.sum(axis=0)
                processed = processed[fixy >= 1e-3]  # 只考虑建立了的
                utilization_cv[i,j]=np.std(processed)/np.mean(processed)
                utilization_ptp[i,j]=np.ptp(processed)

###################################################################################################################### 画图可视化，这里定制每个地方画什么
lines=[] #记录所有的曲线，注意append的时候只取plot返回的第一个变量[0]
ax_record=[] # 记录所有ax
fig,axes=plt.subplots(2,3)
fig.suptitle('collected amount')

for i in range(2):
    for j in range(3):
        ax=axes[i,j]
        ax.set_title(model[3*i+j+1])
        ax.plot(x,collected[:,3*i+j+1,3*i+j+1],label="own setting",marker='o')
        ax.plot(x, collected[:, 3*i+j+1, 0], label="DDP setting",marker='^')
        # ax.plot(x, collected[:, 0, 0], label="DDP optimal")
        # ax.plot(x,fixcollected[:, 3*i+j+1],label="own estimate",marker='*',markersize='10')
        # ax.plot(x, DDP_fixcollected[:, 3 * i + j + 1], label="DDP estimate",marker='d')
        ax.legend(loc='upper left')

plt.show()