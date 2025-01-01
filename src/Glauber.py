# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:14:22 2024

Simulates the Glauber dynamics for a non-reciprocal Hopfield network system 
with two coupled attractors. An optional external force can be applied to the 
system after 1/4 of the total simulation time.

Global Parameters
---------------
lp : float
    Positive coupling strength between same subnet spins
lm : float
    Negative coupling strength between different subnet spins
T : float
    Temperature of the system
NA, NB : int
    Number of spins in subnet A and B respectively
N : int
    Total number of spins (NA + NB)
MA, MB : int
    Initial magnetization of subnet A and B respectively
tsteps : int
    Total number of time steps for simulation
    
Note:
The external force F is applied after tsteps/4 time steps, allowing the system
first evolves naturally before being subjected to external influence.
----

@authors: Shuyue Xue (shuyue.xue413@gmail.com, xueshuy1@msu.edu)
          Carlo Piermarocchi(piermaro@msu.edu)
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.stats import norm
from numba import jit

# ------------- global simulation parameters -------------
lp = 1.3
lm = 0.126105418 + 0.06

T = 1.0
NA = 80000
NB = 80000
N = NA + NB
MA = NA
MB = -NB
TMA = [MA]
TMB = [MB]
tsteps = N * 200
# ------------- End parameters -------------

@jit(nopython=True)
def run(F=0, num_simulations=3):
    """
    Parameters
    ----------
    F : float, optional
        External Force strength (default is 0)
    num_simulations : int, optional
        Number of independent simulation runs (default is 1)
        
    Returns
    -------
    tuple of np.ndarray
        m1 : Array of shape (num_simulations, tsteps//N + 1) 
        containing the sum of magnetizations in the two subnets
        
        m2 : Array of shape (num_simulations, tsteps//N + 1) 
        containing the difference of magnetizations in the two subnets
    """
    
    num_seeds = num_simulations
    m1 = np.zeros((num_seeds, tsteps // N + 1))
    m2 = np.zeros((num_seeds, tsteps // N + 1))
    
    for seed in range(num_seeds):
        np.random.seed(seed+8)
        MA = NA
        MB = -NB
        TMA = [MA]
        TMB = [MB]

        for i in range(1, tsteps + 1):
            DMA = 0
            DMB = 0
            if np.random.randint(1, N + 1) <= NA:
                if np.random.randint(1, NA + 1) <= (MA + NA) / 2:
                    if i < tsteps / 4:
                        prob = (1 - np.tanh(2 * (lp * (MA - 1) - lm * MB) / N)) / 2
                    else:
                        prob = (1 - np.tanh(2 * (lp * (MA - 1) - lm * MB) / N + F)) / 2
                    DMA = -2 * (prob > np.random.rand())
                    MA = MA + DMA
                    if abs(MA) >= NA:
                        DMA = 0
                        MA = np.sign(MA) * NA
                else:
                    if i < tsteps / 4:
                        prob = (1 + np.tanh(2 * (lp * (MA + 1) - lm * MB) / N)) / 2
                    else:
                        prob = (1 + np.tanh(2 * (lp * (MA + 1) - lm * MB) / N + F)) / 2
                    DMA = 2 * (prob > np.random.rand())
                    MA = MA + DMA
                    if abs(MA) >= NA:
                        DMA = 0
                        MA = np.sign(MA) * NA
            else:
                if np.random.randint(1, NB + 1) <= (MB + NB) / 2:
                    if i < tsteps / 4:
                        prob = (1 - np.tanh(2 * (lp * (MB - 1) + lm * MA) / N)) / 2
                    else:
                        prob = (1 - np.tanh(2 * (lp * (MB - 1) + lm * MA) / N + F)) / 2
                    DMB = -2 * (prob > np.random.rand())
                    MB = MB + DMB
                    if abs(MB) >= NB:
                        DMB = 0
                        MB = np.sign(MB) * NB
                else:
                    if i < tsteps / 4:
                        prob = (1 + np.tanh(2 * (lp * (MB + 1) + lm * MA) / N)) / 2
                    else:
                        prob = (1 + np.tanh(2 * (lp * (MB + 1) + lm * MA) / N + F)) / 2
                    DMB = 2 * (prob > np.random.rand())
                    MB = MB + DMB
                    if abs(MB) >= NB:
                        DMB = 0
                        MB = np.sign(MB) * NB
            if i % N == 0:
                TMA.append(MA)
                TMB.append(MB)

        m1[seed, :] = (np.array(TMA) + np.array(TMB)) / N
        m2[seed, :] = (np.array(TMA) - np.array(TMB)) / N
        
    return m1, m2

if __name__ == "__main__":
    F = 0.0
    m1, m2 = run(F, num_simulations=1)
    
    plt.plot(m1.mean(axis=0), label='m1')
    plt.plot(m2.mean(axis=0), label='m2')
    plt.legend()