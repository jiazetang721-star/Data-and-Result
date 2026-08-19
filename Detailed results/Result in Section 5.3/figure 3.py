# DDP 下的最优目标，misjudge的代价，不同解的表现差在哪里
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


def network_clustering(CCpoint,RFpoint):
    CC_Center=np.mean(CCpoint,axis=0)
    CCpoint=np.vstack([CCpoint,CC_Center])
    RFpoint=np.vstack([RFpoint,CC_Center])
    # 构建距离矩阵
    CC_dist_matrix = squareform(pdist(CCpoint, lambda u, v: haversine_distance(u[0], u[1], v[0], v[1])))
    RF_dist_matrix = squareform(pdist(RFpoint, lambda u, v: haversine_distance(u[0], u[1], v[0], v[1])))

    ACD1=np.mean(CC_dist_matrix[:-1,-1])
    ACD2=np.mean(RF_dist_matrix[:-1,-1])

    return ACD1,ACD2 # CC的距离CC中心平均距离，RF到CC中心的平均距离




x=np.array(range(5,16)).astype(str)
model=['DDP','DDN','DDC-10','DDC-15','DIN','DIC-10','DIC-15']
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
processed=np.zeros((11,7,7))
utilization=np.zeros((11,7,7)) # 总的利用率

CC_num=np.zeros((11,7))
RF_num=np.zeros((11,7))
ACDCC=np.zeros((11,7))
ACDRF=np.zeros((11,7))

# local_wise的目标只用在DDP环境下计算，不考虑其他环境下的测试了，就说明其他解不好在哪里
# 衡量均衡性一个方法是计算出整体的方差和极差，另一个方法就是绘制箱线图，每个指标一个图，7个子图表示不同的解，还是11个I表示不同的CC规模，信息量应该也不需要这么大
# 都用cv计算吧，因为整体水平在system-wise的时候已经考虑了一下了，这里主要看各设施之间运转是不是均衡，可能虽然都很低但是分配的比较均衡，同时极差也可以看出来绝对值的差异
recycling_rate_cv=np.zeros((11,7))
recycling_rate_ptp=np.zeros((11,7))
collected_cv=np.zeros((11,7))
collected_ptp=np.zeros((11,7))
utilization_cv=np.zeros((11,7))
utilization_ptp=np.zeros((11,7))

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

    for j in range(7): # 表示7个解
        fixx=solution[i][index[i,j]][:i+5]
        fixy = solution[i][index[i, j]][i + 5:]
        CC_num[i,j]=fixx.sum()
        RF_num[i, j] = fixy.sum()
        CCones=CC_point[np.where(fixx>=1e-3)[0]]
        RFones = RF_point[np.where(fixy >= 1e-3)[0]]
        ACDCC[i,j],ACDRF[i,j]=network_clustering(CCones, RFones)

        for k in range(7): # 表示7个环境
            z = result[k][str(i + 5) + '-' + str(index[i, j]) + '-' + 'z'].reshape((i + 5,J))
            h = result[k][str(i + 5) + '-' + str(index[i, j]) + '-' + 'h']
            objective[i,j,k]=fixx@fc+fixy@fr+(c*z).sum()-(r*z.sum(axis=0)).sum()+fp@h
            building_cost[i,j,k] = fixx@fc+fixy@fr  # 一阶段建造成本
            transportation_cost[i,j,k] = (c*z).sum()  # 运输成本
            penalty_cost[i,j,k] = fp@h  # 惩罚成本
            profit[i,j,k] = (r*z.sum(axis=0)).sum() # 回收的收益
            collected[i,j,k]=z.sum()+h.sum() #总共回收回来的量
            processed[i, j, k] = z.sum()  # 处理的量
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
                processed_tem=z.sum(axis=0)
                utilization_rate = processed_tem[fixy >= 1e-3] /Capacity[fixy >= 1e-3] # 只考虑建立了的
                utilization_cv[i,j]=np.std(utilization_rate)/np.mean(utilization_rate)
                utilization_ptp[i,j]=np.ptp(utilization_rate)

second_stage_profit=profit-transportation_cost-penalty_cost

########################################################### 画图可视化
# net profit+input-output ratio, buildingcost+buildingnumber, transportationcost+clusteringindex, e-wasteamount+utilization,
lines=[] #记录所有的曲线，注意append的时候只取plot返回的第一个变量[0]
ax_record=[] # 记录所有ax
fig,axes=plt.subplots(2,2)
# fig.suptitle('performance analysis')

# ax=axes[0,0]
# ax_record.append(ax)
# ax.set_title("objective")
# for j in range(7):
#     # lines.append(ax.plot(x, (objective[:, j, 0] -objective[:, 0, 0]) / (-objective[:, 0, 0]), label=model[j], marker=marker[j], markersize=5)[0])
#     lines.append(ax.plot(x, objective[:, j, 0] , label=model[j], marker=marker[j],markersize=5)[0])

ax=axes[0,0]
ax.set_title("(a) relative objective gap")
ax.set_visible(False)
gs_inner = ax.get_subplotspec().subgridspec(2, 1,height_ratios=[1,3], hspace=0.1)
ax_top = fig.add_subplot(gs_inner[0, 0])

# ax_top.set_xticks([5,6,7,8,9,10,11,12,13,14,15])
ax_top.set_xticks([])
ax_bottom = fig.add_subplot(gs_inner[1, 0],sharex=ax_top)

# 去掉 ax1 的底部边框和 ax2 的顶部边框
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.xaxis.tick_top() # ax1不显示底部刻度
ax_top.set_title("(a) relative objective gap")


ax_bottom.xaxis.set_tick_params(labelbottom=True, bottom=True)
ax_bottom.xaxis.tick_bottom()

ax_record.append(ax_top)
ax_record.append(ax_bottom)
for j in range(4):
    # lines.append(ax_bottom.plot(x,objective[:,j,0],label=model[j],marker=marker[j],markersize=5)[0])
    lines.append(ax_bottom.plot(x, (objective[:, j, 0]-objective[:,0,0])/(-objective[:,0,0]), label=model[j], marker=marker[j], markersize=5)[0])
for j in range(4,7):
    # lines.append(ax_top.plot(x,objective[:,j,0],label=model[j],marker=marker[j],markersize=5,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j])[0])
    lines.append(ax_top.plot(x, (objective[:, j, 0]-objective[:,0,0])/(-objective[:,0,0]), label=model[j], marker=marker[j], markersize=5,
                             color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j])[0])
d = .015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# 1. 强制开启底部子图的刻度标签显示
ax_bottom.xaxis.set_tick_params(labelbottom=True, bottom=True)

# 2. 设置刻度位置（如果你想让 ax_bottom 只在下面显示刻度）
ax_bottom.xaxis.tick_bottom()

# 3. 设置具体的刻度数值 (必须和 ax_top 保持同步，因为它们是 sharex)
ticks = [5,6,7,8,9,10,11,12,13,14,15]
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels([str(t) for t in ticks])

# 4. 可选：如果你觉得 ax_top 顶部的刻度太乱，想关掉它
ax_top.xaxis.set_tick_params(labeltop=False, top=False)


ax=axes[0,1]
ax_record.append(ax)
ax.set_title("(b) ROI")
for j in range(7):
    # lines.append(ax.plot(x, (-objective[:, j, 0]/building_cost[:, j, 0]+objective[:, 0, 0]/building_cost[:,0, 0])/(-objective[:, 0, 0]/building_cost[:,0, 0]), label=model[j],marker=marker[j],markersize=5)[0])
    lines.append(ax.plot(x,(-objective[:, j, 0] / building_cost[:, j, 0]), label=model[j], marker=marker[j],markersize=5)[0])

ax=axes[1,0]
ax_record.append(ax)
ax.set_title("(c) collected amount")
for j in range(7):
    lines.append(ax.plot(x,collected[:,j,0], label=model[j],marker=marker[j],markersize=5)[0])


ax=axes[1,1]
ax_record.append(ax)
ax.set_title("(d) utilization")
for j in range(7):
    lines.append(ax.plot(x,utilization[:,j,0],label=model[j],marker=marker[j],markersize=5)[0])

##############################################################################################################

# 创建全局图例
label_lines=lines[0:7]
leg = fig.legend(label_lines, [l.get_label() for l in label_lines], loc='center right',fontsize=12,
    markerscale=2.0,    # 将图例中的标记点（如散点）放大到原来的2倍
    handlelength=3.0,   # 增加图例线条的长度
    handletextpad=0.5 )  # 调整图标与文字之间的间距)
leg.set_draggable(True)
# 建立映射关系
line_map = {}
for j in range(7):
    line_map[leg.get_lines()[j]]= [lines[7*i+j] for i in range(int(len(lines)/7))]

# 全屏
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
for legline in line_map.keys():
    legline.set_picker(True)
    legline.set_pickradius(10)  # 增加到 10 像素，鼠标靠近就能触发


def on_pick(event):
    legline = event.artist
    if legline not in line_map: return

    orig_lines = line_map[legline]
    visible = not orig_lines[0].get_visible()

    for line in orig_lines:
        line.set_visible(visible)

    legline.set_alpha(1.0 if visible else 0.2)

    # --- 核心步骤：重新计算坐标轴范围 ---
    for ax in ax_record:
        # 排除掉不可见的线，重新计算数据极限
        ax.relim(visible_only=True)
        # 根据新的数据极限自动缩放视图
        ax.autoscale_view()

    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', on_pick)
plt.subplots_adjust(right=0.85)

plt.show()
