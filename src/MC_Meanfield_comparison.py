# -*- coding: utf-8 -*-
"""
Created on Mon July  8 14:51:31 2023

Compare the Glauber Dynamic vs. Mean Field Approximation

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""
import numpy as np, matplotlib.pyplot as plt
from numba import jit
from scipy.integrate import odeint
import sys
plt.rcParams['text.usetex'] = True

@jit(nopython=True)
def simulate(M1, M2, NA, NB, N, lp, lm, nrep, tsteps, taumax):
    for j in range(nrep):
        MA, MB = NA, NB
        M1[j, 0], M2[j, 0] = MA + MB, MA - MB
        
        np.random.seed(j)

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


def run(N, lp, lm, nrep, taumax):
    NA = NB = N // 2
    tsteps = N * taumax
    M1 = np.zeros((nrep, taumax), dtype='float32')
    M2 = np.zeros((nrep, taumax), dtype='float32')

    simulate(M1, M2, NA, NB, N, lp, lm, nrep, tsteps, taumax)

    return M1, M2



# ------------- MC parameters -------------
Tf= 1
tsteps = int(351)    # code test
t_min = 0

lp = 1.3
lm = 0.17
N = 100   
nrep = 1
taumax = tsteps
# ------------- End parameters -------------

Ns = [100, 1000, 10000, 100000]
ms = np.zeros( (taumax, len(Ns)), dtype='float32')

for i, N in enumerate(Ns):
    _, M2 = run(N, lp, lm, nrep, taumax)
    M2 = M2.reshape(-1)
    ms[:, i] = M2/N


def equations(m1m2, t, lp, lm, beta, tau0=1):
    m1, m2 = m1m2
    lamd_s = lp + lm
    lamd_a = lp - lm
    term2 = beta * (lamd_a*m1 + lamd_s*m2)
    term3 = beta * (lamd_s*m1 - lamd_a*m2)
    m1_dot = -m1/tau0 + 1/(2*tau0)*(np.tanh(term2) + np.tanh(term3))
    m2_dot = -m2/tau0 + 1/(2*tau0)*(np.tanh(term2) - np.tanh(term3))    
    return m1_dot, m2_dot


NA = NB = N // 2
MA, MB = NA, NB
MA, MB = NA, NB
p1, p2 = (MA + MB)/N, (MA - MB)/N

beta = 1
tau0 = 1
t = np.linspace(0,  tsteps, num=tsteps)
m1_int, m2_int = np.copy(p1), np.copy(p2)

num_points = tsteps
solution = odeint(equations, (m1_int, m2_int), t, args=(lp, lm, beta, tau0))
m1 = solution[:, 0]
m2 = solution[:, 1]

# Plot M2
plot_start = 0
plot_end = -1
fig, ax = plt.subplots(figsize=(21, 8.5))
plt.plot(t[plot_start:plot_end], m2[plot_start:plot_end], 'k-', 
          linewidth=3.1, alpha=1,
          label=r'$\mathbf{Mean\ Field}$', )

markers = ['o', '^', 'x', 'd']
for i in range(ms.shape[1]):
    if i == 2: alpha = 1
    else: alpha = 0.68
    plt.plot(t[plot_start:plot_end], ms[plot_start:plot_end, i], '--', 
             linewidth=1.,
             marker=markers[i], ms=8, 
             label=rf'$\mathbf{{N: {Ns[i]}}}$', 
             alpha=alpha)

plt.legend()
plt.xlabel(r'$\mathbf{t/\tau_0}$', fontsize=31)
plt.ylabel(r'$\mathbf{m_2(t)}$', fontsize=31)
plt.xticks(fontsize=41)
plt.yticks(fontsize=41)
plt.tight_layout()