# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
"""
Created on Tue Dec 31 20:31:14 2024


@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os
from numba import jit
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import StrMethodFormatter

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'

def Corr(m1, m2, N, t_ref=0, taus=np.array([0])):
    """m1, m2 = (nrep, mag), C = < z(t_start + tau) * z(t)_conj >"""
    z = (m1 - m2 * 1j).astype(np.complex64)
    c = z[:, t_ref + taus] * np.conj(z[:, t_ref])[:, np.newaxis]
    c /= N**2
    c_mean = c.mean(axis=0)

    c_sem_r = np.std(c.real, axis=0, ddof=1) / np.sqrt(N)
    c_sem_im = np.std(c.imag, axis=0, ddof=1) / np.sqrt(N)
    c_sem_ab = np.std(np.abs(c), axis=0, ddof=1) / np.sqrt(N)

    return c_mean.real, c_mean.imag, np.abs(c_mean), c_sem_r, c_sem_im, c_sem_ab


@jit(nopython=True)
def scaling(taus, c, c_err, N, zeta=1 / 3, alpha=0):
    return taus / N**zeta, N**(-alpha) * c, N**(-alpha) * c_err


def load_data(N, lp, lm, fdir='../data/MF_cv'):
    lp_str = f"{lp:.2f}".replace('.', 'p')
    lm_str = f"{lm:.4f}".replace('.', 'p')
    fname = f'MC_N{N}_lp{lp_str}_lm{lm_str}.xlsx'
    file = os.path.join(fdir, fname)
    xls = pd.ExcelFile(file)
    df_m1 = pd.read_excel(xls, sheet_name="M1", header=0, index_col=0)
    df_m2 = pd.read_excel(xls, sheet_name="M2", header=0, index_col=0)
    return df_m1.values, df_m2.values


def scaled(N, lp, lm, t_ref=0, taus=np.array([0]), zeta=1 / 3, alpha=0, fdir='../data/MF_cv'):
    m1, m2 = load_data(N, lp, lm, fdir)
    print('Loaded data for', (N, lp, lm))
    r, im, ab, r_err, im_err, ab_err = Corr(m1, m2, N, t_ref, taus)
    t_scl, r_scl, r_err_scl = scaling(taus, r, r_err, N, zeta, alpha)
    _, im_scl, im_err_scl = scaling(taus, im, im_err, N, zeta, alpha)
    _, ab_scl, ab_err_scl = scaling(taus, ab, ab_err, N, zeta, alpha)
    return t_scl, r_scl, r_err_scl, im_scl, im_err_scl, ab_scl, ab_err_scl


fdir = '../data/MF_cv'
Ns = [1000, 10000, 30000, 50000]
zeta = 1 / 3
t_ref = 100
dtau = 3
taus = np.arange(0, 405 + dtau, dtau, dtype=np.int32)


lp = 1.25
lm = 0.1025

ts = np.zeros((len(Ns), len(taus)), dtype=np.float32)
rs = np.zeros((len(Ns), len(taus)), dtype=np.float64)
r_errs, ims, im_errs, ab, ab_errs = (np.copy(rs), np.copy(rs), np.copy(rs),
                                     np.copy(rs),np.copy(rs),
                                     )


for i, N in enumerate(Ns):
    ts[i], rs[i], r_errs[i], ims[i], im_errs[i], ab[i], ab_errs[i] = scaled(
        N, lp, lm, t_ref=t_ref, taus=taus, zeta=zeta, alpha=0
    )

lp_str = f"{lp:.2f}".replace('.', 'p')
lm_str = f"{lm:.4f}".replace('.', 'p')

# Colormap 
cmap_real = plt.cm.Greens
cmap_imag = plt.cm.Purples
cmap_abs = plt.cm.Grays
cmap_ebar = plt.cm.Blues

def get_color(cmap, idx, total, vmin=0.55, vmax=1.0):
    return cmap(vmin + (vmax - vmin) * idx / (total - 1))


gls, pls = [], []
fig, axs = plt.subplots(1, 1, figsize=(21, 8.5), sharex=True)

# Real and Imag of z
axs = [axs] # swtich to single plot
ax = axs[0]
for i in range(len(Ns)):
    if i == 0 : a = 1
    else: a = 0.68
    gl = ax.errorbar(ts[i], rs[i], yerr=r_errs[i],
                   fmt='-', marker=None,
                   linewidth=4.1,
                   color = get_color(cmap_real, i, 4), 
                   label=rf'$\mathbf{{N = {Ns[i]}}}$',
                   ecolor = get_color(cmap_real, i, 4), 
                   elinewidth=4.1, capsize=6, capthick=1.7,
                   alpha=a)
    
    pl = ax.errorbar(ts[i], ims[i], yerr=im_errs[i],
                    fmt='-', marker=None,
                    linewidth=4.1,
                    color=get_color(cmap_imag, i, 4),
                    ecolor= get_color(cmap_imag, i, 4),
                    elinewidth=4.1, capsize=6, capthick=1.7,
                    alpha=a)
    
    gls.append(gl)
    pls.append(pl)

# legend with tuples
paired_handles = [(gl[0], pl[0]) for gl, pl in zip(gls, pls)]  
paired_labels = [rf'$\mathbf{{N = {N}}}$' for N in Ns]  
handles, labels =ax.get_legend_handles_labels() # get the handles and labels
my_handles = [gls, pls]
# pair handles 
paired_handles = [tuple([gl, pl]) for (gl, pl) in zip(gls, pls) ]
# map legend to the paired handles
lnd = ax.legend(handles=paired_handles, labels=labels, 
          handler_map={tuple: HandlerTuple(ndivide=None)}, 
          handlelength=8, fontsize=24,
          title=r'$\mathbf{Re[C_z] \quad Im[C_z]}$',  
          title_fontproperties=\
              {'size': 24, 'weight': 'bold'},
          prop={'size': 24},
          frameon=False
          )

lnd.get_title().set_position((-60, 0))  # 'left' aligns the title text

ax.set_ylabel(rf'$\mathbf{{C_z(t={t_ref}~\tau_0, \tau)}}$', 
              fontsize=44, labelpad=8)
ax.set_xlabel(r'$\mathbf{\tau/(\tau_0 N^{1/3})}$', fontsize=44)

ax.set_xlim(-1, 31)
ax.tick_params(axis='both', which='both', labelsize=27, length=3.1, width=3.1)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
plt.tight_layout()

plt.savefig(f'lp{lp_str}_lm{lm_str}_Vpaper.pdf')
plt.close(fig)