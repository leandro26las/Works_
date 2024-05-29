# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:11:53 2023

@author: leand
"""

import numpy as np
import matplotlib.pyplot as plt

N = 5                   # Numero de pulsos
TR = 10e-9              # Tempo entre pulsos
fr = 100e6              # Frequencia de Repeticao
fase = 0.1              # Diferenca de Fase
wc = 2 * np.pi * fr     # Frequencia do pulso
ti = -3*TR
tf = 5*TR
Dt = tf - ti
g12 = 
g22 = 
w12 =
wc =
d = w12 - wc


def E(t):
    E_0 = 1 / np.cosh( 1.763 * t / TR )
    return E_0 * np.exp( - 1j*( wc*t - n*wc*TR + n*fase ) )

def O(t):
    return 

# ELECTRIC FIELD
for n in range(N):
    t = np.linspace(ti + n*Dt, ti + (n+1)*Dt, num = 200, endpoint = True)
    
    plt.plot(t/TR , E(t - ti - (n+1/2)*Dt ), c='blue' )
    plt.plot(t/TR, abs(E(t - ti - (n+1/2)*Dt )), c='red', lw=2)

s12 = s12*(1j*d - g12)*dt - 1j*O(t)*(1-2*p22)*dt
p22 = -g22*p22*dt + 1j*O(t)*( s12 - np.conj(s12) )*dt