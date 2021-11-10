# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:12:26 2021

@author: zongsing.huang
"""

#%% 模組
import functools

import numpy as np
import matplotlib.pyplot as plt

import benchmark
from algorithm import GA


#%% 開始排配GA
COST, M, N, F_ideal = benchmark.t7()
P = 100
D = N
G = 500
PC = 0.80
PM = 0.01
PE = 0.1
PI = 0.1
OBJ = functools.partial(benchmark.OBJ, COST=COST, M=M, N=N)
# optimizer = GA(OBJ=OBJ, P=P, D=D, G=G, M=M, TABU=TABU)
# optimizer.opt()


