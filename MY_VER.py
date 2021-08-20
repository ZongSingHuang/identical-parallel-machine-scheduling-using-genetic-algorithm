# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

@author: zongsing.huang
"""

# =============================================================================
# x in {0, 1}
# 最大化問題的最佳適應值為-D；最小化問題的最佳適應值為0
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

def fitness(X, pt, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    F = np.zeros([P, M])
    
    for i in range(P):
        machine = X[i]
        
        for j in range(M):
            mask = machine==j
            F[i, j] = pt[mask].sum()
    
    F = F.max(axis=1)
    
    return F

def selection(X, F):
    P = X.shape[0]
    
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    if F_sum==0:
        normalized_F = np.zeros(P)
    else:
        normalized_F = F/F_sum

    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    
    new_idx = -1*np.ones(P).astype(int)
    r = np.random.uniform(size=P)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    p1 = X[new_idx][:int(P/2)]
    p2 = X[new_idx][int(P/2):]
    
    return p1, p2

def crossover(p1, p2, pc, M, N):
    P = p1.shape[0]
    D = p1.shape[1]
    temp1 = np.zeros_like(p1) - 1
    temp2 = np.zeros_like(p2) - 1
    c1 = np.zeros_like(p1) - 1
    c2 = np.zeros_like(p2) - 1
    
    for i in range(P):
        mask = np.random.choice(2, size=D, replace=True, p=[1-pc, pc]).astype(bool) # 強制決定概率
        # mask = np.random.choice(2, size=D, replace=True).astype(bool) # 隨機概率
        temp1[i, mask] = p1[i, mask]
        temp1[i, ~mask] = p2[i, ~mask]
        mask = np.random.choice(2, size=D, replace=True, p=[1-pc, pc]).astype(bool) # 強制決定概率
        # mask = np.random.choice(2, size=D, replace=True).astype(bool) # 隨機概率
        temp2[i, mask] = p2[i, mask]
        temp2[i, ~mask] = p1[i, ~mask]
    pc = 1.0 # 強制決定概率時要啟用
    
    for i in range(P):
        r1 = np.random.uniform()
        if r1<=pc:
            c1[i] = temp1[i]
        else:
            c1[i] = p1[i]
            
        r2 = np.random.uniform()
        if r2<=pc:
            c2[i] = temp2[i]
        else:
            c2[i] = p2[i]
    
    return c1, c2

def mutation(c1, pm, M, N):
    P = c1.shape[0]
    D = c1.shape[1]
    
    for i in range(P):
        for j in range(D):
            r = np.random.uniform()
            if r<=pm:
                while True:
                    temp = np.random.randint(low=0, high=M)
                    if c1[i, j]!=temp:
                        c1[i, j] = temp
                        break
    return c1

def elitism(X, F, new_X, new_F, er, M, N):
    P = X.shape[0]
    elitism_size = int(P*er)
    
    if elitism_size>0:
        idx = np.argsort(F)
        elitism_idx = idx[:elitism_size]
        elite_X = X[elitism_idx]
        elite_F = F[elitism_idx]
        
        for i in range(elitism_size):
            
            if elite_F[i]<new_F.mean():
                idx = np.argsort(new_F)
                worst_idx = idx[-1]
                new_X[worst_idx] = elite_X[i]
                new_F[worst_idx] = elite_F[i]
    
    return new_X, new_F

def immigrant(new_X, new_F, ir, pt, M, N):
    P = new_X.shape[0]
    D = new_X.shape[1]
    N = P
    immigrant_size = int(P*er)
    
    if immigrant_size>0:
        
        for i in range(immigrant_size):
            immigrant_X = np.random.choice(M, size=[1, D])
            immigrant_F = fitness(immigrant_X, pt, N, M)
            
            if immigrant_F<new_F.mean():
                idx = np.argsort(new_F)
                worst_idx = idx[-1]
                new_X[worst_idx] = immigrant_X
                new_F[worst_idx] = immigrant_F
    
    return new_X, new_F

#%% 參數設定
species = 't30'
if species=='t7':
    # gbest_F is 10
    pt = np.array([6, 6, 4, 4, 4, 3, 3])
    M = 3 # 機台數
    N = 7 # 工單數
if species=='t9':
    # gbest_F is 10
    pt = np.array([9, 7, 12, 15, 8, 10, 11, 13, 7])
    M = 4 # 機台數
    N = 9 # 工單數
elif species=='t10':
    # gbest_F is 25
    pt = np.array([3, 2, 6, 4, 5, 7, 8, 6, 2, 6])
    M = 2 # 機台數
    N = 10 # 工單數
elif species=='t30':
    # gbest_F is 41
    pt = np.array([3, 2, 6, 4, 5, 7, 9, 13, 4, 12,
                   10, 8, 22, 11, 8, 26, 14, 6, 17, 27,
                   11, 17, 26, 16, 7, 23, 15, 18, 15, 13])
    M = 10 # 機台數
    N = 30 # 工單數

P = 20 # 一定要偶數
D = N
G = 200
pc = 0.9
pm = 0.05
er = 0.1
ir = 0.1
lb = 0*np.ones([P, D])
ub = M*np.ones([P, D])
ppp = np.zeros(50)

#%% 初始化
# 若P不是偶數，則進行修正
if P%2!=0:
    P = 2 * (P//2)

for t in range(50):
    X = np.random.randint(low=lb, high=ub)
    gbest_X = np.zeros(D)
    gbest_F = np.inf
    loss_curve = np.zeros(G)
    
    #%% 迭代
    # 適應值計算
    F = fitness(X, pt, M, N)
        
    for g in range(G):
        # 更新F
        if F.min()<gbest_F:
            best_idx = np.argmin(F)
            gbest_X = X[best_idx]
            gbest_F = F[best_idx]
        loss_curve[g] = gbest_F
        
        # 選擇
        p1, p2 = selection(X, F)
        
        # 交配
        c1, c2 = crossover(p1, p2, pc, M, N)
        
        # 突變
        c1 = mutation(c1, pm, M, N)
        c2 = mutation(c2, pm, M, N)
        
        # 更新X
        new_X = np.vstack([c1, c2])
        np.random.shuffle(new_X)
        
        # 適應值計算
        new_F = fitness(new_X, pt, N, M)
        
        # 菁英
        new_X, new_F = elitism(X, F, new_X, new_F, er, M, N)
        
        # 移民
        new_X, new_F = immigrant(new_X, new_F, ir, pt, M, N)
        
        X = new_X.copy()
        F = new_F.copy()
    
    # print('gbest_F is {gbest_F:.0f}'.format(gbest_F=gbest_F))
    ppp[t] = gbest_F

print('mean:{mean:.2f}, std:{std:.2f}'.format(mean=ppp.mean(), std=ppp.std()))

#%% 適應值
plt.figure()
plt.title('gbest_F is {gbest_F:.0f}'.format(gbest_F=gbest_F))
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')

#%% 甘特圖
# https://www.gushiciku.cn/pl/gXr9/zh-hk
Machine = [[] for i in range(M)]
Job = [[] for i in range(M)]
cumulative_running_time = np.zeros(N) # 機台累積運轉時間
cumulative_processing_time = np.zeros(N) # 工件累積加工時間

for J_idx, M_idx in enumerate(gbest_X):
    cost = pt[J_idx] # 取得工件J_idx的加工時間
    
    cumulative_processing_time[J_idx] = cumulative_processing_time[J_idx] + cost
    cumulative_running_time[M_idx] = cumulative_running_time[M_idx] + cost
    
    if cumulative_running_time[M_idx]<cumulative_processing_time[J_idx]:
        cumulative_running_time[M_idx]=cumulative_processing_time[J_idx]
    elif cumulative_running_time[M_idx]>cumulative_processing_time[J_idx]:
        cumulative_processing_time[J_idx]=cumulative_running_time[M_idx]

    present = cumulative_processing_time[J_idx] - cost
    increment = cost

    Machine[M_idx].append((present, increment))
    Job[M_idx].append('Job ' +str(J_idx+1))
    # print(Machine[0], Machine[1], Machine[2])

plt.figure(dpi=100, facecolor='white')
plt.title("Gantt Chart", pad=10, fontsize=16, fontweight='bold') # 標題
for i in range(M):
    color = ['#BC3C28', '#0972B5', '#E28726', '#21854D']
    plt.broken_barh(Machine[i], (10*(i+1), 5), facecolors=color[i%4], edgecolor='black', label='Machine '+str(i+1), zorder=2)
max_makespan = -np.inf
loc_makespan = None
for i in range(M):
    last = Machine[i][-1][0] + Machine[i][-1][1]
    if max_makespan<last:
        max_makespan = last
        loc_makespan = [last, 10*(i+1)]
plt.scatter(loc_makespan[0], loc_makespan[1]+2.5, marker='3', s=300)
for txt_set, loc_set, m in zip(Job, Machine, range(M)):
    for txt, loc in zip(txt_set, loc_set):
        plt.text(loc[0]+0.25, 10*(m+1)+2, txt, fontsize='small', alpha=1)
plt.legend(frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -.3), fontsize='small') # 每一row顯示4個圖例
plt.xlabel('Time', fontsize=15) # X軸標題
plt.ylabel('Machine', fontsize=15) # Y軸標題
plt.yticks([10*(i+1)+2.5 for i in range(M)], ['M'+str(i+1) for i in range(M)])
plt.grid(linestyle="-", linewidth=.5, color="gray", alpha=.6) # 網格
plt.tight_layout() # 自動校正
