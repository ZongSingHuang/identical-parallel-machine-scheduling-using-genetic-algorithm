# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:44:58 2021

@author: zongsing.huang
"""

import numpy as np

def t7():
    COST = np.array([6, 6, 4, 4, 4, 3, 3])
    M = 3 # 機台數
    N = 7 # 工單數
    F_ideal = 10
    
    return COST, M, N, F_ideal

def t9():
    COST = np.array([9, 7, 12, 15, 8, 10, 11, 13, 7])
    M = 4 # 機台數
    N = 9 # 工單數
    F_ideal = 10
    
    return COST, M, N, F_ideal

def t10():
    COST = np.array([3, 2, 6, 4, 5, 7, 8, 6, 2, 6])
    M = 2 # 機台數
    N = 10 # 工單數
    F_ideal = 25
    
    return COST, M, N, F_ideal

def t30():
    COST = np.array([ 3,  2,  6,  4, 5,  7,  9, 13,  4, 12,
                     10,  8, 22, 11, 8, 26, 14,  6, 17, 27,
                     11, 17, 26, 16, 7, 23, 15, 18, 15, 13])
    M = 10 # 機台數
    N = 30 # 工單數
    F_ideal = 41
    
    return COST, M, N, F_ideal

def OBJ(X, COST, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    F = np.zeros([P, M])
    
    for i in range(M):
        subset = X==i
        
        for j in range(P):
            F[j, i] = F[j, i] + COST[subset[j]].sum()
    
    F = F.max(axis=1)
    
    return F