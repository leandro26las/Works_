
import numpy as np
# from numba import jit

gamma22 = 2*np.pi*5e6
gamma12 = gamma22 / 2 
w21 = 2*np.pi*400e12

wc = 2*np.pi*400e12
fr = 1e9
Tr = 1. / fr
Tp = 100e-15
fase = 0.0
n = 10
Omega_0 = 1.
delta = w21 - wc

# def Omega_0(t):
#     return (1. / np.cosh(1.763*t/Tp) ) * np.exp(-1j*(2*np.pi*fc*t - n*2*np.pi*fc*Tr - n*fase) )


def ds12dt(t, s12, p22):
    return (1j*delta - gamma12)*s12 - 1j*Omega_0*(1 - 2*p22) 

def dp22dt(t, s12, p22):
    return - gamma22*p22 + 1j*Omega_0*( s12 - np.conj(s12) )


def rk4(t0, tf, s12_0, p22_0, N):
    h = (tf - t0)/N
    s12 = np.zeros(N)
    p22 = np.zeros(N)
    s12[0] = s12_0
    p22[0] = p22_0
    for n in range(N-1):
        k1 = ds12dt(t0, s12[n], p22[n])
        l1 = dp22dt(t0, s12[n], p22[n])
        
        k2 = ds12dt(t0 + 0.5*h, s12[n] + 0.5*k1, p22[n] + 0.5*l1)
        l2 = dp22dt(t0 + 0.5*h, s12[n] + 0.5*k1, p22[n] + 0.5*l1)
        
        k3 = ds12dt(t0 + 0.5*h, s12[n] + 0.5*k2, p22[n] + 0.5*l2)
        l3 = dp22dt(t0 + 0.5*h, s12[n] + 0.5*k2, p22[n] + 0.5*l2)
        
        k4 = ds12dt(t0 + h, s12[n] + k3, p22[n] + l3)
        l4 = dp22dt(t0 + h, s12[n] + k3, p22[n] + l3)
        
        s12[n+1] = s12[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        p22[n+1] = p22[n] + (h/6)*(l1 + 2*l2 + 2*l3 + l4)
        
        t0 = t0 + h

    return s12.imag, p22.real

a = 0.0
b = 10.0
tam = 10000
s12_0 = 0.0
p22_0 = 0.0
t = np.linspace(a, b, num=tam, endpoint=True)
f = rk4(a, b, s12_0, p22_0, tam)

import matplotlib.pyplot as plt

plt.plot(t, f[0], label = 'coerencia', c = 'red')
plt.plot(t, f[1], label='populacao', c = 'black')
plt.legend(loc='best')