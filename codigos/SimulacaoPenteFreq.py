# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:11:53 2023

@author: leand
"""

import numpy as np
import matplotlib.pyplot as plt

m = 0                   # Numero de modos
N = 5                   # Numero de pulsos
TR = 10e-9              # Tempo entre pulsos
fr = 100e6              # Frequencia de Repeticao
fase = 0.1              # Diferenca de Fase
w_fs = 2 * np.pi * fr   # Frequencia do pulso
ti = -3*TR
tf = 5*TR
Dt = tf - ti
g12 = 
g22 = 
w12 =
w_cw = 2 * np.pi * 400e15
k_fs = 2 * np.pi / 795e-9
k_cw = 2 * np.pi / 780e-9


def E_fs(t,z):
    E_0 = 1 / np.cosh( 1.763 * t / TR )
    return E_0 * np.exp( - 1j * ( w_fs * t - m * w_fs * TR + m * fase - k_fs * z) )

def Omega_cw(t,z):
    E_cw = 
    return E_cw * np.exp( - 1j * ( w_cw * t - k_cw * z ) )

# ELECTRIC FIELD
for n in range(N):
    t = np.linspace(ti + n*Dt, ti + (n+1)*Dt, num = 200, endpoint = True)
    
    plt.plot(t/TR , E_fs(t - ti - (n+1/2)*Dt ), c='blue' )
    plt.plot(t/TR, abs(E_fs(t - ti - (n+1/2)*Dt )), c='red', lw=2)

s12 = s12*(1j*d - g12)*dt - 1j*O(t)*(1-2*p22)*dt
p22 = -g22*p22*dt + 1j*O(t)*( s12 - np.conj(s12) )*dt