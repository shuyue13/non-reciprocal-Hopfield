# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:40:51 2024

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""

import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.stats import linregress


cd = os.path.dirname(__file__)
data_folder = os.path.join(cd, "..", "data")
f1 = os.path.join(data_folder, "Hopf_T.xlsx")
f2 = os.path.join(data_folder, 'Fold_T.xlsx')
f3 = os.path.join(data_folder, 'Fold_T_p12.xlsx')
data1 = pd.read_excel(f1)
data2 = pd.read_excel(f2)
data3 = pd.read_excel(f3)

Ns = data1['N']
T1, T1_err = data1['T'], data1['T_err']
T2, T2_err = data2['T'], data2['T_err']
T3, T3_err = data3['T'], data3['T_err']

# Hopf
m1, b1, R1, p_val, std_err1 = linregress(Ns, T1)

# Fold
m2, b2, R2, p_val, std_err2 = \
    linregress(Ns, T2)
m3, b3, R3, p_val, std_err3 = \
    linregress(Ns, np.log(T3))

x = np.linspace(min(Ns)-31, max(Ns)+41, 100)
y1 = m1*x + b1
y2 = m2*x + b2
y3 = np.exp(b3)* np.exp(m3*x)

fig, ax = plt.subplots(figsize=(19, 8.5))
line1, = plt.plot(Ns, T1, 'X', ms=15, label=r'$\mathbf{Near\ Hopf}$')
line2, = plt.plot(Ns, T2, '^', ms=15, label=r'$\mathbf{Above\ Fold}$')
line3, = plt.plot(Ns, T3, 'd', ms=15, label=r'$\mathbf{Below\ Fold}$')

plt.plot(x, y1, line1.get_color(), linewidth=4.1,
         label=rf'\bf{{Fit}}: $ \mathbf{{T_{{near\_Hopf}} = ' \
         rf'{m1:.2f}N {b1:+.2f}}} $')
plt.plot(x, y2, line2.get_color(), linewidth=4.1, 
         label=rf'\bf{{Fit}}: $ \mathbf{{T_{{above\_Fold}} = '\
             rf'{m2:.2f}N {b2:+.2f}}} $')
prefactor = np.exp(b3)
plt.plot(x, y3, line3.get_color(), linewidth=4.1, 
         label=rf'\bf{{Fit}}: $ \mathbf{{T_{{below\_Fold}} = '\
             rf'{prefactor:.4f} \cdot e^{{{m3:.6f}N}}}} $')

plt.xlabel(r'\textbf{$N$}', fontsize=31)
plt.ylabel(r'\textbf{$T(N)$}', fontsize=31)
plt.legend(fontsize=22)