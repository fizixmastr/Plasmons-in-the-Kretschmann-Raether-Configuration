# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:21:42 2020

@author: chrambo
"""

# =============================================================================
# Plasmons in the Kretschmann-Raether Configuration
# =============================================================================

import pandas as pd
import numpy as np
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt
from scipy import interpolate

#%% Givens
res = 101 #Number of points
λ_min = 0.400 #Approximate lower threshold for visible light
λ_max = 0.800 #Approximate lower threshold for visible light
λ = np.linspace(λ_min, λ_max, res) #Optical range in micrometers range from advisor
#λ = np.array([.380, .381]) #placeholder value for testing
θ_min = 40 #lower limit in degrees
θ_max = 60 #Upper limit in degrees 
θ = np.linspace(θ_min, θ_max, res) #Sugested range from advisor
#θ = np.array([49, 50]) #placeholder value for testing
d = 0.05 #Film is given as 50 nm thick -->.05 micrometer

#%% Get Refractive index functions

n_Au = pd.read_csv('https://refractiveindex.info/data_csv.php?datafile=data/main/Au/Johnson.yml')
n_BK7 = pd.read_csv('https://refractiveindex.info/data_csv.php?datafile=data/glass/schott/N-BK7.yml')
n_Air = pd.read_csv('https://refractiveindex.info/data_csv.php?datafile=data/other/mixed%20gases/air/Ciddor.yml')

n_Au = np.transpose(np.array([n_Au.apply(pd.to_numeric, errors='coerce').to_numpy()[:49,0], 
    ((n_Au.apply(pd.to_numeric, errors='coerce')).to_numpy()[:49,1]+
    (n_Au.apply(pd.to_numeric, errors='coerce')).to_numpy()[-49:,1]*1j)]))
n_BK7 = (n_BK7.apply(pd.to_numeric, errors='coerce')).to_numpy()[:101,:]
n_Air = (n_Air.apply(pd.to_numeric, errors='coerce')).to_numpy()

n_Au = interpolate.interp1d(n_Au[:,0], n_Au[:,1])
n_BK7 = interpolate.interp1d(n_BK7[:,0], n_BK7[:,1])
n_Air = interpolate.interp1d(n_Air[:,0], n_Air[:,1])

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5])
plt.suptitle('Complex Refractive Index $n,k$ for Componet Materials')
axs[0].set(title='Gold (Au)', xlabel='$\lambda [\mu m]$',
           ylabel = '$n,k$', ylim = (0, 5) , xlim = (λ_min, λ_max))
axs[0].plot(λ,np.real(n_Au(λ)), label = 'Real')
axs[0].plot(λ,np.imag(n_Au(λ)), label = 'Imaginary')
axs[0].grid()
axs[0].legend()

axs[1].set(title='BK7 Glass', xlabel='$\lambda [\mu m]$',
           ylabel = '$n$', ylim = (0, 5), xlim = (λ_min, λ_max))
axs[1].plot(λ,n_BK7(λ))
axs[1].grid()

axs[2].set(title='Air', xlabel='$\lambda [\mu m]$',
           ylabel = '$n$', ylim = (0, 5), xlim = (λ_min, λ_max))
axs[2].plot(λ,n_Air(λ))
axs[2].grid()

ε_0 = n_BK7(λ)**2
ε_1 = n_Au(λ)**2
ε_2 = n_Air(λ)**2

#%% Calculate Reflectivities

k_x = np.empty([len(λ),len(θ)])
for n in range(len(λ)):
    for m in range(len(θ)):
        k_x[n, m] = (2*np.pi*(n_BK7(λ[n])/λ[n])*np.sin(np.deg2rad(θ[m])))

# k_z0 = np.empty([res,res], dtype = complex)
# for i in range(res):
#     k_z0[i,:] = sqrt(ε_0[i]*(2*np.pi/λ[i])**2 - k_x[i,:]**2)
# k_z1 = np.empty([res,res], dtype = complex)
# for i in range(res):
#     k_z1[i,:] = sqrt(ε_1[i]*(2*np.pi/λ[i])**2 - k_x[i,:]**2)
# k_z2 = np.empty([res,res], dtype = complex)
# for i in range(res):
#     k_z2[i,:] = sqrt(ε_2[i]*(2*np.pi/λ[i])**2 - k_x[i,:]**2)
    
k_z0 = np.empty([len(λ),len(θ)], dtype = complex)
k_z1 = np.empty([len(λ),len(θ)], dtype = complex)
k_z2 = np.empty([len(λ),len(θ)], dtype = complex)
for n in range(len(λ)):
    for m in range(len(θ)):
        k_z0[n,m] = sqrt(ε_0[n]*(2*np.pi/λ[n])**2 - k_x[n,m]**2)
        k_z1[n,m] = sqrt(ε_1[n]*(2*np.pi/λ[n])**2 - k_x[n,m]**2)
        k_z2[n,m] = sqrt(ε_2[n]*(2*np.pi/λ[n])**2 - k_x[n,m]**2)

# rp_01 = (k_z0/ε_0 - k_z1/ε_1)/(k_z0/ε_0 + k_z1/ε_1)
# rp_12 = (k_z1/ε_1 - k_z2/ε_2)/(k_z1/ε_1 + k_z2/ε_2)

# rs_01 = (k_z0 - k_z1)/(k_z0 + k_z1)
# rs_12 = (k_z1 - k_z2)/(k_z1 + k_z2)

rp_01 = np.empty([len(λ),len(θ)], dtype = complex)
rp_12 = np.empty([len(λ),len(θ)], dtype = complex)
rs_01 = np.empty([len(λ),len(θ)], dtype = complex)
rs_12 = np.empty([len(λ),len(θ)], dtype = complex)
for n in range(len(λ)):
    for m in range(len(θ)):
        rp_01[n, m] = (k_z0[n, m]/ε_0[n] - k_z1[n, m]/ε_1[n])/(k_z0[n, m]/ε_0[n] + k_z1[n, m]/ε_1[n])
        rp_12[n, m] = (k_z1[n, m]/ε_1[n] - k_z2[n, m]/ε_2[n])/(k_z1[n, m]/ε_1[n] + k_z2[n, m]/ε_2[n])
        rs_01[n, m] = (k_z0[n, m] - k_z1[n, m])/(k_z0[n, m] + k_z1[n, m])
        rs_12[n, m] = (k_z1[n, m] - k_z2[n, m])/(k_z1[n, m] + k_z2[n, m])

# Rp = np.abs(rp_01+rp_12*np.exp(2*1j*k_z1*d)/(1+rp_01*rp_12*np.exp(2*1j*k_z1*d)))**2
# Rs = np.abs(rs_01+rs_12*np.exp(2*1j*k_z1*d)/(1+rs_01*rs_12*np.exp(2*1j*k_z1*d)))**2

Rp = np.empty([len(λ),len(θ)])
Rs = np.empty([len(λ),len(θ)])
Ratio = np.empty([len(λ),len(θ)])
for n in range(len(λ)):
    for m in range(len(θ)):
        Rp[n, m] = np.abs((rp_01[n, m]+rp_12[n, m]*np.exp(2*1j*k_z1[n, m]*d))/(1+rp_01[n, m]*rp_12[n, m]*np.exp(2*1j*k_z1[n, m]*d)))**2
        Rs[n, m] = np.abs((rs_01[n, m]+rs_12[n, m]*np.exp(2*1j*k_z1[n, m]*d))/(1+rs_01[n, m]*rs_12[n, m]*np.exp(2*1j*k_z1[n, m]*d)))**2
        Ratio [n, m] = Rp[n, m]/Rs[n, m]

#%% Make plots
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5])
plt.suptitle('Reflectivity of $p$ Polarized Light')
axs[0].set(title='$R_p(\\theta)$', xlabel= '$\\theta[^\circ]$',
           ylabel = '$R_p$', xlim = (θ[0], θ[-1]))
axs[0].plot(θ, Rp[0,:], color = 'b', label = '$\lambda =$ ' + str(λ[0]) + ' µm')
axs[0].plot(θ, Rp[-1,:], color = 'r', label = '$\lambda =$ ' + str(λ[-1]) + ' µm')
axs[0].grid()
axs[0].legend()

axs[1].set(title = '$R_p(\lambda)$', xlabel = '$\lambda$ [μm]',
           ylabel = '$R_p$', xlim = (λ_min, λ_max))
axs[1].plot(λ, Rp[:,0], label = '$\\theta =$ ' + str(θ[0]) + '$^\circ$')
axs[1].plot(λ, Rp[:,-1], label = '$\\theta =$ ' + str(θ[-1]) + '$^\circ$')
axs[1].grid()
axs[1].legend()

axs[2].set(xlabel= '$\\theta[^\circ]$', ylabel = '$\lambda$ [μm]')
axs[2].pcolor(θ, λ, Rp)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5])
plt.suptitle('Reflectivity of $s$ Polarized Light')
axs[0].set(title='$R_s(\\theta)$', xlabel= '$\\theta[^\circ]$',
           ylabel = '$R_s$', xlim = (θ[0], θ[-1]))
axs[0].plot(θ, Rs[0,:], color = 'b', label = '$\lambda =$ ' + str(λ[0]) + ' µm')
axs[0].plot(θ, Rs[-1,:], color = 'r', label = '$\lambda =$ ' + str(λ[-1]) + ' µm')
axs[0].grid()
axs[0].legend()

axs[1].set(title = '$R_s(\lambda)$', xlabel = '$\lambda$ [μm]',
           ylabel = '$R_s$', xlim = (λ_min, λ_max))
axs[1].plot(λ, Rs[:,0], label = '$\\theta =$ ' + str(θ[0]) + '$^\circ$')
axs[1].plot(λ, Rs[:,-1], label = '$\\theta =$ ' + str(θ[-1]) + '$^\circ$')
axs[1].grid()
axs[1].legend()

axs[2].set(xlabel= '$\\theta[^\circ]$', ylabel = '$\lambda$ [μm]')
axs[2].pcolor(θ, λ, Rs)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5])
plt.suptitle('Ratio of $R_p/R_s$')
axs[0].set(title='$R_p/R_s(\\theta)$', xlabel= '$\\theta[^\circ]$',
           ylabel = '$R_p/R_s$', xlim = (θ[0], θ[-1]))
axs[0].plot(θ, Ratio[0,:], color = 'b', label = '$\lambda =$ ' + str(λ[0]) + ' µm')
axs[0].plot(θ, Ratio[-1,:], color = 'r', label = '$\lambda =$ ' + str(λ[-1]) + ' µm')
axs[0].grid()
axs[0].legend()

axs[1].set(title = '$R_p/R_s(\lambda)$', xlabel = '$\lambda$ [μm]',
           ylabel = '$R_p/R_s$', xlim = (λ_min, λ_max))
axs[1].plot(λ, Ratio[:,0], label = '$\\theta =$ ' + str(θ[0]) + '$^\circ$')
axs[1].plot(λ, Ratio[:,-1], label = '$\\theta =$ ' + str(θ[-1]) + '$^\circ$')
axs[1].grid()
axs[1].legend()

axs[2].set(xlabel= '$\\theta[^\circ]$', ylabel = '$\lambda$ [μm]')
axs[2].pcolor(θ, λ, Ratio)