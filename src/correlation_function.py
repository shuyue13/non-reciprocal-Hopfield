# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:41:23 2024

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""

import numpy as np, matplotlib.pyplot as plt,  pandas as pd
import os
cd = os.path.dirname(__file__)
fdir = os.path.join(cd, "..", "data")
files = [f for f in os.listdir(fdir) if 'C_t' in f and '.csv' in f ]
files = sorted(files, key=lambda x: int(x.split('_N')[1].split('.csv')[0]))
N = [ int((f.split('.')[0]).split('N')[1]) for f in files ]
dfs = [pd.read_csv(os.path.join(fdir, f), index_col=0) for f in files]

fig, ax = plt.subplots(figsize=(13, 6))
for i in range(len(dfs)):
    alpha=1
    lw = 1.7 + i*0.86
    plt.plot(dfs[i], '-', 
             linewidth = lw,
             label=f'$N = {N[i]}$', alpha=alpha)
plt.legend()
plt.xlabel(r'\textbf{$\tau/\tau_0$}')
plt.ylabel(r'\textbf{$C_{2,2}(15, \tau)$}')
plt.tight_layout()