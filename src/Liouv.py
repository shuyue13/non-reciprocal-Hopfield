# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:10:35 2024

A module for solving master equations using Liouvillian approach.

Key Components:
    - LiouvillianMatrix: Main class handling the construction and evolution of
    the system
    - Master equation solution 
    - Time correlation calculations for various time delays
    - Support for various initial state configurations
    
Parameters for system initialization:
    NA, NB : Number of spins in subnets A and B
    JP, JM : Coupling strengths (plus/minus) for interactions
    init_state : Initial state configuration (optional)
    
Example Usage:
    master = LiouvillianMatrix(NA=40, NB=40, JP=1.3, JM=0.17)
    correlation = master.C_ttaus(t=1, taumax=100)
    
@authors: Shuyue Xue (shuyue.xue413@gmail.com, xueshuy1@msu.edu)
          Carlo Piermarocchi(piermaro@msu.edu)
"""

import numpy as np, matplotlib.pyplot as plt,  pandas as pd
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import sys

class LiouvillianMatrix: 
    def __init__(self, NA, NB, JP, JM, init_state=None):
        self.NA = NA
        self.NB = NB
        self.N = NA + NB
        self.MA = np.arange(-NA, NA + 1, 2, dtype=np.int16)
        self.MB = np.arange(-NB, NB + 1, 2, dtype=np.int16)
        self.sLA, self.sLB = len(self.MA), len(self.MB)
        self.sLT = self.sLA * self.sLB
    
        self.MMA, self.MMB = np.meshgrid(self.MA, self.MB, indexing='ij')
        self.MMA = self.MMA.ravel().astype(np.int16)
        self.MMB = self.MMB.ravel().astype(np.int16)
        self.M1 = self.MMA + self.MMB
        self.M2 = self.MMA - self.MMB
        self.Liouv = dok_matrix((self.sLT, self.sLT), dtype=np.float32)
    
        self.liouvill(JP, JM)
        self.P0 = np.zeros(self.sLT, dtype=np.float32)
        if init_state is None:
            self.P0[-1] = 1.0           
        else:
            self.P0[init_state-1] = 1.0  
            
    def liouvill(self, JP, JM):
        # NPA, NPB
        NPA = (self.NA + self.MMA) // 2
        NPB = (self.NB + self.MMB) // 2
        WPA = (1 - np.tanh(2 * (JP * (self.MMA - 1) - JM * self.MMB) / self.N)) / 2
        WPB = (1 - np.tanh(2 * (JP * (self.MMB - 1) + JM * self.MMA) / self.N)) / 2
        WMA = (1 + np.tanh(2 * (JP * (self.MMA + 1) - JM * self.MMB) / self.N)) / 2
        WMB = (1 + np.tanh(2 * (JP * (self.MMB + 1) + JM * self.MMA) / self.N)) / 2
        
        self.Liouv.setdiag(NPA * WPA + NPB * WPB + \
                           (self.NA - NPA) * WMA + (self.NB - NPB) * WMB)

        self.set_off_diagonals(NPA, NPB, JP, JM)
        self.Liouv = self.Liouv.tocsc()
        self.Liouv = self.Liouv.astype(np.float32)
        return self.Liouv


    def set_off_diagonals(self, NPA, NPB, JP, JM):
        sLA, sLB, sLT = self.sLA, self.sLB, self.sLT
        
        idxA = np.arange(sLT).reshape(sLA, sLB)
        for i in range(sLA - 1):
            k = idxA[i].ravel()
            kp = idxA[i + 1].ravel()
            WPAP2 = (1 - np.tanh(2 * (JP * (self.MMA[kp] - 1) - JM * self.MMB[kp]) / self.N)) / 2
            self.Liouv[k, kp] = -NPA[kp] * WPAP2
        
        for i in range(1, sLA):
            k = idxA[i].ravel()
            kp = idxA[i - 1].ravel()
            WMAM2 = (1 + np.tanh(2 * (JP * (self.MMA[kp] + 1) - JM * self.MMB[kp]) / self.N)) / 2
            self.Liouv[k, kp] = -(self.NA - NPA[kp]) * WMAM2
        
        for j in range(sLB - 1):
            k = idxA[:, j].ravel()
            kp = idxA[:, j + 1].ravel()
            WPBP2 = (1 - np.tanh(2 * (JP * (self.MMB[kp] - 1) + JM * self.MMA[kp]) / self.N)) / 2
            self.Liouv[k, kp] = -NPB[kp] * WPBP2

        for j in range(1, sLB):
            k = idxA[:, j].ravel()
            kp = idxA[:, j - 1].ravel()
            WMBM2 = (1 + np.tanh(2 * (JP * (self.MMB[kp] + 1) + JM * self.MMA[kp]) / self.N)) / 2
            self.Liouv[k, kp] = -(self.NB - NPB[kp]) * WMBM2

    
    def set_p0(self, k):    # set P(k', 0)
        """Set initial P(0) to be concentrated at state k."""
        self.P0 *= 0        # Reset P0
        self.P0[k-1] = 1.0  # k to 1


    def m_exp(self, JP, JM, t, mem=1):
        """Calculates expectation value of either m1, m2 or |z| at time t. 
        Returns normalized results."""
        
        self.liouvill(JP, JM)
        if mem == 1:   M = self.M1 
        elif mem == 2: M = self.M2
        elif mem == 0: M = abs( self.M1 + self.M2 *1j )
        results = np.zeros(len(t), dtype=np.float32)
        for i, time in enumerate(t):
            P = expm_multiply(-self.Liouv * time, self.P0)
            results[i] = M @ P 
        return results / self.N        
        
    
    def get_P_t(self, t):
        """Get P(t) by evolving initial P0 under Liouvillian for time t."""
        if t == 0: return self.P0
        else: return expm_multiply(-self.Liouv * t, self.P0)


    def C_ttaus(self, t, taumax, mem=2):
        """Calculate time correlation function C(t,tau) 
        for different time delays with a reference time t.

        Parameters
        ----------
        t : int
            The reference time point at which to calculate correlations
        taumax : int
            Maximum time delay to calculate correlations for
        mem : int, optional
            Memory matrices M1 or M2
            Default is 2

        Returns
        -------
        ndarray
            Array of correlation values C(t,tau) for tau ranging 
            from 0 to taumax-1, and normalized by N^2 
        """
        
        if mem == 1: M = self.M1
        if mem == 2: M = self.M2
        Pt = self.get_P_t(t)
        c = np.zeros(taumax)
        for tau in range(taumax):
            c[tau] = (expm_multiply(-self.Liouv.T*tau, M) * M * Pt).sum()
        return c/self.N**2


    def state_info(self, k):
        """Gets the state information (MA, MB, P values) for a given state 
        index k."""
        print(f"State index {k}: \n", \
              f"MA = {self.MMA[k]}, MB = {self.MMB[k]},  P = {self.P0[k]}")


if __name__ == "__main__":    
    
    NA = NB = 40
    lp, lm = 1.3, 0.17
    t, taumax = 1, 100
     
    master = LiouvillianMatrix(NA, NB, lp, lm, init_state=None)
    c_tau = master.C_ttaus(t, taumax, mem=2)
    plt.plot(c_tau)