# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:41:23 2024

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""

import numpy as np, matplotlib.pyplot as plt,  pandas as pd
import os

cd = os.path.dirname(__file__)
data_folder = os.path.join(cd, "..", "data")

Ns = [50, 100, 150, 200]
dfs = [pd.read_csv(os.path.join(data_folder, f'LV_NA_{N}_NB_{N}.csv'), index_col=0) for N in Ns]
m2s = [df.m2 for df in dfs]

# plt.figure(1, figsize=(17, 8))
fig, ax = plt.subplots(figsize=(21, 8.5))
for i in range(len(dfs)):
    alpha=1
    lw = 1.7 + i*0.86
    plt.plot(m2s[i], '-', 
             linewidth = lw,
             label=f'$\mathbf{{N = {Ns[i]*2}}}$', alpha=alpha)
plt.legend(fontsize=24)
plt.grid(color='gray', linestyle='-', linewidth=0.31, alpha=0.41)
plt.xlabel(r'$\mathbf{t/\tau_0}$', fontsize=44)
plt.ylabel(r'$\mathbf{ \langle m_2(t) \rangle} $', fontsize=44)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tight_layout()