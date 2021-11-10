# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:43:25 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
class GA():
    def __init__(self, OBJ, P, D, G, M, PC, PM, PE, PI, TABU):
        self.OBJ = OBJ
        self.P = P # 染色體總數，須為>0的偶數
        self.D = D # 維度/訂單 總數
        self.G = G # 最大迭代次數，需為>0的整數
        self.M = M # PI 機台數
        self.PC = PC # 交配率，介於 [0, 1] 的浮點數
        self.PM = PM # 突變率，介於 [0, 1] 的浮點數
        self.PE = PE # 菁英率，介於 [0, 1] 的浮點數
        self.PI = PI # 移民率，介於 [0, 1] 的浮點數
        self.TABU = TABU # RELEASE TABLE
        self.X_GBEST = None
        self.F_GBEST = np.inf
        self.loss_curve = np.zeros(self.G)

#%%
    def opt(self):
        self.X = self.inital(self.P) # 初始化
        self.F = self.OBJ(self.X) # 適應值計算
        
        for g in range(self.G):
            # 最佳解更新
            if self.F.min()<self.F_GBEST:
                idx = self.X.argmin()
                self.F_GBEST = self.F.min()
                self.X_GBEST = self.X[idx]
            self.loss_curve[g] = self.F_GBEST.copy()
            
            # 選擇
            self.X_new1 = self.selection(self.X)
            
            # 交配
            self.X_new2 = self.crossover(self.X_new1)
            
            # 突變
            self.X_new3 = self.mutation(self.X_new2)
            
            # 適應值計算
            self.F_new3 = self.OBJ(self.X_new3)
            
            # 菁英
            self.X_new3, self.F_new3 = self.elitism(
                self.X, self.F, self.X_new3, self.F_new3)
            
            # 移民
            self.X_new3, self.F_new3 = self.immigrant(self.X_new3, self.F_new3)
            
            # 取代
            self.X, self.F = self.X_new3.copy(), self.F_new3.copy()
            
        return 0

#%%
    def inital(self, P):
        X = np.random.randint(low=0, high=self.M, size=[P, self.D])
        X = self.repair_chromosome(X)
        
        return X
    
    def selection(self, X):
        # 輪盤法
        F_sum = self.F.sum()
        F_normalized = self.F/F_sum
        
        idx = np.random.choice(a=self.P, p=F_normalized, size=[self.P])
        
        X_new1 = self.X[idx].copy()
        
        return X_new1
    
    def crossover(self, X_new1):
        # 雙點交配
        P1 = self.X_new1[:int(self.P/2)].copy()
        P2 = self.X_new1[int(self.P/2):].copy()
        C1 = np.zeros_like(P1)
        C2 = np.zeros_like(P2)
        
        R = np.random.uniform(size=[int(self.P/2)])
        
        for i, r in enumerate(R):
            if r<=self.PC:
                idx1, idx2 = np.sort( np.random.choice(self.D, size=2, replace=False) )
                C1[i] = np.hstack([ P1[i, :idx1], P2[i, idx1:idx2], P1[i, idx2:] ])
                C2[i] = np.hstack([ P2[i, :idx1], P1[i, idx1:idx2], P2[i, idx2:] ])
            else:
                C1[i] = P1[i].copy()
                C2[i] = P2[i].copy()
        
        X_new2 = np.vstack([C1, C2])
        
        X_new2 = self.repair_chromosome(X_new2)
        
        return X_new2
    
    def mutation(self, X_new2):
        # 單點突變
        X_new3 = X_new2.copy()
        X_mutation = self.inital(self.P)
        R = np.random.unifoirm(size=[self.P, self.D])
        
        mask = R<=self.PM
        X_new3[mask] = X_mutation[mask]
        
        return X_new3
    
    def elitism(self, X, F, X_new, F_new):
        elitism_size = int(self.P*self.PE)

        if elitism_size>0:
            idx = F.argsort()[:elitism_size]
            X_elitism = X[idx].copy()
            F_elitism = F[idx].copy()
            
            for i in range(elitism_size):
                if F_elitism[i]<F_new.mean():
                    idx = F_new.argsort()[-1]
                    X_new[idx] = X_elitism[i].copy()
                    F_new[idx] = F_elitism[i].copy()
        else:
            pass
        
        return X_new, F_new
    
    def immigrant(self, X, F):
        immigrant_size = int(self.P*self.PI)
        X_new = X.copy()
        F_new = F.copy()
        
        if immigrant_size>0:
            X_immigrant = self.inital(immigrant_size)
            F_immigrant = self.OBJ(X_immigrant)
            
            for i in range(immigrant_size):
                if F_immigrant[i]<F_new.mean():
                    idx = F_new.argsort()[-1]
                    X_new[idx] = X_immigrant[i].copy()
                    F_new[idx] = F_immigrant[i].copy()
        else:
            pass
        
        return X_new, F_new

#%%
    def repair_chromosome(self, X):
        X_repaired = X.copy()
        
        for i, chromosome in enumerate(X):
            for j, gene in enumerate(chromosome):
                feasible_machine = np.where(self.TABU.loc[j]==1)[0]
                infeasible_machine = np.where(self.TABU.loc[j]==0)[0]
                
                if gene in feasible_machine:
                    pass
                
                if gene in infeasible_machine:
                    X_repaired[i, j] = np.random.choice(feasible_machine)
                    
        return X_repaired
    
    def plot_curve(self):
        plt.figure()
        plt.title('Loss Curve')
        plt.plot(self.loss_curve, marker='o', linewidth=2, markersize=12, label='Loss')
        plt.grid()
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()