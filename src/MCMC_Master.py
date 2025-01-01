# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:18:00 2024


@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply
import sys, os
from numba import jit
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'


@jit(nopython=True)
def simulate(M1, M2, NA, NB, N, lp, lm, nrep, tsteps, taumax, seed=0):
    for j in range(nrep):
        MA, MB = NA, NB
        M1[j, 0], M2[j, 0] = MA + MB, MA - MB
        
        np.random.seed(2*j+1)
        # np.random.seed(j+seed)
        for i in range(0, tsteps):
            if np.random.randint(1, N + 1) <= NA:
                if np.random.randint(1, NA + 1) <= (MA + NA) / 2:
                    prob = (1 - np.tanh(2 * (lp * (MA - 1) - lm * MB) / N)) / 2
                    MA -= 2 * (prob > np.random.random())
                    if abs(MA) > NA: MA = np.sign(MA) * NA
                else:
                    prob = (1 + np.tanh(2 * (lp * (MA + 1) - lm * MB) / N)) / 2
                    MA += 2 * (prob > np.random.random())
                    if abs(MA) > NA: MA = np.sign(MA) * NA
            else:
                if np.random.randint(1, NB + 1) <= (MB + NB) / 2:
                    prob = (1 - np.tanh(2 * (lp * (MB - 1) + lm * MA) / N)) / 2
                    MB -= 2 * (prob > np.random.random())
                    if abs(MB) > NB: MB = np.sign(MB) * NB
                else:
                    prob = (1 + np.tanh(2 * (lp * (MB + 1) + lm * MA) / N)) / 2
                    MB += 2 * (prob > np.random.random())
                    if abs(MB) > NB: MB = np.sign(MB) * NB
                                
            if i % N == 0:
                index = i // N + 1
                # index = i // N
                if index <= taumax:
                    M1[j, index] = MA + MB
                    M2[j, index] = MA - MB
            
    return M1, M2

def run(N, lp, lm, nrep=10, taumax=100, seed=4):
    NA = NB = N // 2
    tsteps = N * taumax
    M1 = np.zeros((nrep, taumax), dtype='float32')
    M2 = np.zeros((nrep, taumax), dtype='float32')
    simulate(M1, M2, NA, NB, N, lp, lm, nrep, tsteps, taumax, seed)
    return M1, M2

# MC simulations
nreps = [1, 10, 100, 1000]  # Different number of repetitions
N = 200
tsteps = int(200)
taumax = tsteps
lp = 1.3
lm = 0.17

m2s = np.zeros( (taumax, len(nreps)), dtype='float32')
for i, nrep in enumerate(nreps):
    _, M2 = run(N, lp, lm, nrep, taumax)
    m2s[:, i] = M2.mean(axis=0) / N


# LV data
cd = os.path.dirname(__file__)
fdir = os.path.join(cd, "..", "data")
fname = f'LV_NA_{N//2}_NB_{N//2}.csv'
data_file = os.path.join(fdir, fname)
df_lv = pd.read_csv(data_file)

# Plot M2
plot_start = 0
plot_end = -1
fig, ax = plt.subplots(figsize=(19, 8.5))
plt.plot(df_lv['m2'], 
         label=r'$\mathbf{Liouvillian}$',
         color='black', linewidth=6.8, alpha=0.6)

for i in range(m2s.shape[1]):
    if i == 2: alpha = 1
    alpha=1
    a = str(nreps[i])
    plt.plot(m2s[:, i], '--', alpha=alpha,
             linewidth=4.1,
             label= rf'$\mathbf{{MC \ Runs: {a}}}$')

plt.legend()
plt.xlabel(r'$\mathbf{t/\tau_0}$', fontsize=41)
plt.ylabel(r'$\mathbf{m_2(t)}$', fontsize=41)
plt.tight_layout()