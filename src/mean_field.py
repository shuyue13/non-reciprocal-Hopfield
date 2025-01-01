# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:37:43 2023

@author: Shuyue Xue (shuyue.xue413@gmail.com)
"""

import numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, matplotlib.patches as patches
from scipy.integrate import odeint

def equations(m1m2, t, lp, lm, beta):
    m1, m2 = m1m2
    lamd_s = lp + lm
    lamd_a = lp - lm
    term2 = beta * (lamd_a*m1 + lamd_s*m2)
    term3 = beta * (lamd_s*m1 - lamd_a*m2)
    
    m1_dot = -m1 + 1/2*(np.tanh(term2) + np.tanh(term3))
    m2_dot = -m2 + 1/2*(np.tanh(term2) - np.tanh(term3))    
    return m1_dot, m2_dot

def phase_plot(lp, lm, beta=1, ax=None):
    if None == ax:
        ax = plt.gca()
        
    num_points = 17
    x_start, x_end = -1., 1.
    y_start, y_end = -1., 1.
    xs = np.linspace(x_start, x_end, num_points)
    ys = np.linspace(y_start, y_end, num_points)
    X, Y = np.meshgrid(xs, ys)
    
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            u[i, j], v[i, j] = equations([x, y], 0, lp, lm, beta)
    
    ax.quiver(X, Y, u, v,
                angles='xy', scale_units='xy',
                headwidth = 31,
                width=0.0015,
                scale= 10,
                color='green',
                )
    
    t = np.linspace(0, 42, 313)
    m1_inits = np.linspace(x_start, x_end, 5)
    m2_inits = np.linspace(y_start, y_end, 5)
    m1_inits = np.hstack((m1_inits, [-0.1,-0.05, 0.05, 0.1]))
    m2_inits = np.hstack((m2_inits, [-0.1,-0.05, 0.05, 0.1]))
    for m1_in in m1_inits:
        for m2_in in m2_inits:
            solution = odeint(equations, (m1_in, m2_in), t, args=(lp, lm, beta))
            m1 = solution[:, 0]
            m2 = solution[:, 1]
            ax.plot(m1, m2, linewidth=0.6, c='k')            
    ax.set_title(r'$\beta\lambda_+$ = %s,  \quad  $\beta\lambda_-$ = %s' %(lp, lm), 
                 fontsize=17)


def sech(x):
    return 1 / np.cosh(x)


def f(m1, m2, lp, lm):
    return np.array([
        -m1 + (np.tanh((lp - lm) * m1 + (lp + lm) * m2) + np.tanh((lp + lm) * m1 - (lp - lm) * m2)) / 2,
        
        -m2 + (np.tanh((lp - lm) * m1 + (lp + lm) * m2) - np.tanh((lp + lm) * m1 - (lp - lm) * m2)) / 2,
        
        1 - lp * (np.cosh((lp - lm) * m1 + (lp + lm) * m2)**-2 + np.cosh((lp + lm) * m1 - (lp - lm) * m2)**-2) +
        (lp**2 + lm**2) * np.cosh((lp - lm) * m1 + (lp + lm) * m2)**-2 * np.cosh((lp + lm) * m1 - (lp - lm) * m2)**-2
    ])


def jacobianf(m1, m2, lp, lm):
    j11 = -1 + ((lp - lm) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 + (lp + lm) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2
    j12 = ((lp + lm) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 - (lp - lm) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2
    j13 = ((m2 - m1) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 + (m1 + m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2

    j21 = ((lp - lm) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 - (lp + lm) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2
    j22 = -1 + ((lp + lm) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 + (lp - lm) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2
    j23 = ((m2 - m1) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 - (m1 + m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) / 2

    j31 = 2 * lp * ((lp - lm) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 + (lp + lm) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) - 2 * (lp**2 + lm**2) * ((lp - lm) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 * sech((lp + lm) * m1 - (lp - lm) * m2)**2 + (lp + lm) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2 * sech((lp - lm) * m1 + (lp + lm) * m2)**2)
    
    j32 = 2 * lp * ((lp + lm) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 - (lp - lm) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) - 2 * (lp**2 + lm**2) * ((lp + lm) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 * sech((lp + lm) * m1 - (lp - lm) * m2)**2 - (lp - lm) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2 * sech((lp - lm) * m1 + (lp + lm) * m2)**2)
    
    j33 = 2 * lp * ((m2 - m1) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 + (m1 + m2) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2) - 2 * (lp**2 + lm**2) * ((m2 - m1) * np.tanh((lp - lm) * m1 + (lp + lm) * m2) * sech((lp - lm) * m1 + (lp + lm) * m2)**2 * sech((lp + lm) * m1 - (lp - lm) * m2)**2 + (m1 + m2) * np.tanh((lp + lm) * m1 - (lp - lm) * m2) * sech((lp + lm) * m1 - (lp - lm) * m2)**2 * sech((lp - lm) * m1 + (lp + lm) * m2)**2) + 2 * lm * sech((lp + lm) * m1 - (lp - lm) * m2)**2 * sech((lp - lm) * m1 + (lp + lm) * m2)**2

    j = np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])

    return j


def newton_method_sx(f, jacobianf, x0, y0, lp, lm0, 
                     max_iter=10000, tol=1e-6):
    x = x0
    y = y0
    lm = lm0
    roots = []
    determinants = []
    
    for i in range(max_iter):
        F = f(x, y, lp, lm)
        J = jacobianf(x, y, lp, lm)
        delta = -np.linalg.inv(J).dot(F)
        x += delta[0]
        y += delta[1]
        lm += delta[2]
        deter = np.linalg.det(J)
        determinants.append(deter)
        if np.linalg.norm(delta) < tol:
            roots.append((x, y, lm))
            break
    return roots, determinants


num_points = 100
lps = np.linspace(1., 3.1, num_points)
x0 = 0.895
y0 = 0.885
lm0 = -1.05
lms_up = []
for lp in lps:    
    roots, determinants = newton_method_sx(f, jacobianf,                                 
                                            x0, y0, lp, lm0)
    if roots:
        lms_up.append(roots[0][2])

temp = x0
x0 = y0
y0 = temp
lm0 *= -1
lms_down = []
for lp in lps:    
    roots, determinants = newton_method_sx(f, jacobianf,                                 
                                            x0, y0, lp, lm0)
    if roots:
        lms_down.append(roots[0][2])

lps, lms_up, lms_down = np.array(lps), np.array(lms_up), np.array(lms_down)
hopf_line = 1.0


plt.rcdefaults()
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'


fig, ax = plt.subplots(figsize=(17, 8))
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])
ax.set_yticks([])
gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 2.5], )
gs.update(hspace=0.26,
          wspace=0.1
          )

ax_central = fig.add_subplot(gs[:, -1])
pos = ax_central.get_position()
ax_central.set_position(
    [pos.x0 + 0.06, pos.y0, pos.width, pos.height])  

ax_central.axvline(x=hopf_line, color='purple', linewidth=2.1, 
                   label='Hopf Bifurcation')
ax_central.plot(lps, lms_up, lps, lms_down, color='darkgreen',
                 linewidth=2.1, label='Fold Bifurcation')      
ax_central.set_xlim((-.41, lps.max()))

xmin, xmax = ax_central.set_xlim()
ymin, ymax = ax_central.set_ylim()
ax_central.set_ylim((ymin, ymax))
ax_central.fill_between(lps, lms_up, lms_down, where=(lms_up > lms_down), 
                 color='darkgreen', alpha=0.68, 
                 label='Memory Retrieval',
                 hatch = '....'
                 ).set_rasterized(True)
ax_central.fill_between(lps, lms_up, ymax, where=(lms_up <= ymax), 
                 color = 'purple', alpha=0.41, 
                 hatch = '///'
                 ).set_rasterized(True)
ax_central.fill_between(lps, lms_down, ymin, where=(lms_down >= ymin), 
                 color = 'purple', alpha=0.41,
                 label='Limit Cycles',
                 hatch = '///').set_rasterized(True)
y_range = np.linspace(ymin, ymax, 100)
ax_central.fill_betweenx(y_range, xmin, hopf_line, 
                  color='green', alpha=0.13, 
                  label='Paramagnetic',
                  hatch = '|||').set_rasterized(True)

handles, labels = plt.gca().get_legend_handles_labels()
handles, labels = np.array(handles), np.array(labels)
uni_lables, uni_index = np.unique(labels, return_index=True)
ax_central.legend(handles[uni_index], labels[uni_index], fontsize=17)
# ax_central.set_title('Phase Diagram', fontsize=21)
ax_central.set_xlabel(r'$\beta\lambda_+$', fontsize=22)
ax_central.set_ylabel(r'$\beta\lambda_{-}$', fontsize=22)
ax_central.tick_params(axis='both', which='major', labelsize=15)

def add_circles(ax, coordinates, filled=True, edge_width=0.6):
    for coord in coordinates:        
        circle = patches.Circle(coord, radius=0.068, edgecolor='darkgreen',
                                facecolor='darkgreen' if filled else 'none', 
                                linewidth=edge_width)
        ax.add_patch(circle)

lps = [0.8, 0.9, 1.0, 1.3, 1.3, 1.3,]
lms = [0., 0.85, 0.85, 0.17, 0.1, 0.]
pairs = list(zip(lps, lms))

positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
plot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
label_count = 0
for (lp, lm), (i, j) in zip(pairs, positions):
    ax = fig.add_subplot(gs[i, j])
    phase_plot(lp, lm, ax=ax)
    if i != 2: ax.tick_params('x', labelbottom=False)
    else: 
        ax.tick_params(axis='x', which='major', labelsize=15)
        ax.set_xlabel('$m_1$', fontsize=22)
    if j != 0: ax.tick_params('y', labelleft=False)
    else: 
        ax.tick_params(axis='y', which='both', labelsize=15)
        ax.set_ylabel('$m_2$', fontsize=22)
    
    ax.annotate(plot_labels[label_count], xy=(0, 1), 
                xycoords='axes fraction', xytext=(-13, 1),
                textcoords='offset points', ha='left', va='bottom', 
                fontsize=19, color='k', fontweight='bold')
    
    if label_count == 5:
        solid_coords = [(0, -0.76), (0, 0.76), (-0.76, 0), (0.76, 0)]
        empty_coords = [(-0.31, -0.31), (-0.31, 0.31), (0.31, -0.31), 
                        (0.31, 0.31)]
        for coord in solid_coords:
            add_circles(ax, solid_coords)
        for coord in empty_coords:
            add_circles(ax, empty_coords, filled=False)  
            
    if label_count == 4:
        solid_coords = [(-0.73, 0.06), (0.73, -0.06), 
                        (0.06, 0.73), (-0.06, -0.73)]
        empty_coords = [(-0.55, 0.24), (0.55, -0.24),
                        (0.24, 0.55), (-0.24, -0.55)
                        ]
        for coord in solid_coords:
            add_circles(ax, solid_coords)
        for coord in empty_coords:
            add_circles(ax, empty_coords, filled=False)
    
    label_count += 1


for label, (lp, lm) in zip(plot_labels, pairs):
    ax_central.annotate(label, xy=(lp, lm), ha='center', va='center', 
                        fontsize=22, color='k', fontweight='bold')

ax_central.set_xlim(0.5, 1.5)
ax_central.set_ylim(-0.5, 1)