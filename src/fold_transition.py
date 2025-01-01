# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:41:56 2024

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import HF_plot_style as sty

plt.rcParams['text.usetex'] = True
t_total = 100.0  
dt = 0.05
n = int(t_total / dt)
theta = np.zeros(n)
theta[0] = 0.0
D = 1
N = 1300
seeds = [8, 13, 141, 315]
thetas = np.zeros((len(seeds), n))
for j, s in enumerate(seeds):
    np.random.seed(s)
    theta = np.zeros(n)
    theta[0] = 0.0
    xi = np.random.normal(0, np.sqrt(dt), n-1) * np.sqrt(D / N)
    for i in range(1, n):
        theta[i] = theta[i - 1] + dt * (1 - np.cos(4 * theta[i - 1])) \
            + xi[i - 1]
    thetas[j] = theta

N = 1300
eps = [-0.5, 0, 0.5]
Vs = np.zeros((len(eps), n))
theta = np.linspace(0, 2*np.pi, num=Vs.shape[1])
for j, epsilon in enumerate(eps):
    Vs[j] = -(1 + epsilon) * theta + 0.25 * np.sin(4 * theta)

plots = [thetas, Vs]
xlabels = [r'\textbf{$t$}', r'\textbf{$\theta$}']
ylabels = [r'\textbf{$\theta(t)$}', r'\textbf{$V(\theta)$}']
for plot_index, (d, ylabel, xlabel) in enumerate(zip(plots, ylabels, xlabels)):
    fig, ax = plt.subplots(figsize=(19, 8.5))
    for i in range(len(d)):
        plt.plot(
            np.linspace(0, t_total, n), 
            d[i],
            linewidth=3.1,
            label = rf'$\epsilon: {eps[i]}$' if plot_index == 1 else None
        )
    plt.legend(fontsize=22) if plot_index == 1 else None
    plt.grid(color='gray', linestyle='-', linewidth=0.31, alpha=0.41)
    plt.xlabel(xlabel, fontsize=31)
    plt.ylabel(ylabel, fontsize=31)
    plt.tight_layout()
    plt.show()
    






