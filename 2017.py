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

def fitness(X, pt, N, M):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P, M])
    
    for i in range(P):
        for j in range(M):
            selected_machine = X[i, N:]==j
            selected_job = X[i, :N][selected_machine]
            F[i, j] = pt[selected_job].sum()
        
    F = F.max(axis=1)
    
    return F

def selection(X, F):
    P = X.shape[0]
    
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    if F_sum==0:
        # 因為題目太簡單，所以還沒迭代完成就會使所有染色體都達到最佳解
        # 因為F_sum=0，所以F/F_sum = 0/0 會跳警告
        # 因此這邊下一個機制
        normalized_F = np.zeros(P)
    else:
        normalized_F = F/F_sum
    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    
    new_idx = -1*np.zeros(P).astype(int)
    r = np.random.uniform(size=P)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    p1 = X[new_idx][:int(P/2)]
    p2 = X[new_idx][int(P/2):]
    
    return p1, p2

def crossover(p1, p2, pc):
    P = p1.shape[0]
    D = p1.shape[1]
    N = int(D/2)
    c1 = np.zeros_like(p1) - 1
    
    for i in range(P):
        job_p1 = p1[i, :N]
        job_p2 = p2[i, :N]
        new_job = np.zeros(N) - 1
        crossover_pt = np.random.uniform(size=[N])<=pc
        new_job[crossover_pt] = job_p1[crossover_pt]
        new_job[~crossover_pt] = job_p2[~np.in1d(job_p2, new_job[crossover_pt])]
        
        machine_p1 = p1[i, N:]
        machine_p2 = p2[i, N:]
        new_machine = np.zeros(N) - 1
        crossover_pt = np.random.uniform(size=[N])<=pc
        new_machine[crossover_pt] = machine_p1[crossover_pt]
        new_machine[~crossover_pt] = machine_p2[~crossover_pt]
        
        c1[i] = np.hstack([new_job, new_machine])
    
    return c1

def mutation(c1, pm, M):
    
    P = c1.shape[0]
    D = c1.shape[1]
    N = int(D/2)
    
    for i in range(P):
        # Bit Flip
        r = np.random.uniform()
        if r<=pm:
            job = c1[i, :N]
            mutation_pt1, mutation_pt2 = np.random.choice(N, size=2, replace=False)
            job[mutation_pt1], job[mutation_pt2] = job[mutation_pt2], job[mutation_pt1]
            c1[i, :N] = job
        
        r = np.random.uniform()
        if r<=pm:
            machine = c1[i, N:]
            mutation_pt = np.random.choice(N)
            while True:
                new_machine = np.random.randint(M)
                if machine[mutation_pt]!=new_machine:
                    machine[mutation_pt] = new_machine
                    c1[i, N:] = machine
                    break
    return c1

def elitism(X, F, new_X, new_F, er):
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

def immigrant(new_X, new_F, ir, M, pt):
    P = new_X.shape[0]
    D = new_X.shape[1]
    N = int(D/2)
    immigrant_size = int(P*er)
    
    if immigrant_size>0:
        
        for i in range(immigrant_size):
            X_job = np.random.choice(N, size=[N], replace=False)
            X_machine = np.random.randint(M, size=[N])
            immigrant_X = np.hstack([X_job, X_machine])
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
elif species=='t10':
    # gbest_F is 25
    pt = np.array([3, 2, 6, 4, 5, 7, 8, 6, 2, 6])
    M = 2 # 機台數
elif species=='t30':
    # gbest_F is 41
    pt = np.array([3, 2, 6, 4, 5, 7, 9, 13, 4, 12,
                   10, 8, 22, 11, 8, 26, 14, 6, 17, 27,
                   11, 17, 26, 16, 7, 23, 15, 18, 15, 13])
    M = 10 # 機台數

P = 20 # 一定要偶數
D = 2*len(pt)
G = 200
N = len(pt) # 工單數
pc = 0.9
pm = 0.05
er = 0.1
ir = 0.1
ppp = np.zeros(50)

#%% 初始化
# 若P不是偶數，則進行修正
if P%2!=0:
    P = 2 * (P//2)

for t in range(50):
    X_job = np.array([np.random.choice(N, size=[N], replace=False) for i in range(P)])
    X_machine = np.random.randint(M, size=[P, N])
    X = np.hstack([X_job, X_machine])
    gbest_X = np.zeros(D)
    gbest_F = np.inf
    loss_curve = np.zeros(G)
    
    #%% 迭代
    # 適應值計算
    F = fitness(X, pt, N, M)
        
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
        c1 = crossover(p1, p2, pc)
        c2 = crossover(p2, p1, pc)
        
        # 突變
        c1 = mutation(c1, pm, M)
        c2 = mutation(c2, pm, M)
        
        # 更新X
        new_X = np.vstack([c1, c2])
        np.random.shuffle(new_X)
        
        # 適應值計算
        new_F = fitness(new_X, pt, N, M)
        
        # 菁英
        new_X, new_F = elitism(X, F, new_X, new_F, er)
        
        # 移民
        new_X, new_F = immigrant(new_X, new_F, ir, M, pt)
        
        X = new_X.copy()
        F = new_F.copy()
    
    print('gbest_F is {gbest_F:.0f}'.format(gbest_F=gbest_F))
    ppp[t] = gbest_F

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

for J_idx, M_idx in zip(gbest_X[:N], gbest_X[N:]):
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
    if Machine[i]!=[]:
        last = Machine[i][-1][0] + Machine[i][-1][1]
        if max_makespan<last:
            max_makespan = last
            loc_makespan = [last, 10*(i+1)]
    else:
        print('有機台是空的')
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
