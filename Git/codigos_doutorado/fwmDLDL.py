# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:02:09 2024

@author: LabDF
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
h = 0.01
r = .3
v = .01
Gamma21 = Gamma23 = 1
Oa = r * Gamma21
Ob = v * Gamma21
Pi = 1.
gamma12 = gamma23 = gamma13 = 0.5 * Gamma21
delta_a = delta_b = delta_ab = 0.0

def F(*args):
    '''
        args[0] = p11;  args[1] = p22;  args[2] = p33;
        args[3] = s12;  args[4] = s13;  args[5] = s23;
        delta()[0] = delta_a;  delta()[1] = delta_b;
        delta()[2] = delta_a - delta_b;
    '''
    f0 = Gamma21*args[1] + 1j*( Oa * np.conj(args[3]) - Oa * args[3] ) - Pi * (args[0] - 1)
    f1 = - ( Gamma21 + Gamma23 + Pi ) * args[1] + 1j*( Oa * args[3] - Oa * np.conj(args[3]) + Ob * np.conj(args[5]) - Ob * args[5] )
    f2 = Gamma23 * args[1] + 1j*( Ob * args[5] - Ob * np.conj( args[5] ) ) - Pi * args[2]
    f3 = (  1j * delta_a - gamma12 - Pi ) * args[3] + 1j * ( Oa * (args[1] - args[0]) - Ob * args[4] )
    f4 = (  1j * (delta_a - delta_b) - gamma13 - Pi ) * args[4] + 1j * ( Oa * args[5] - Ob * args[3] )
    f5 = ( -1j * delta_b - gamma23 - Pi ) * args[5] + 1j * ( Oa * args[4] + Ob * ( args[2] - args[1] ) )

    return [ f0, f1, f2, f3, f4, f5 ]

def interations( x1, x2, x3, x4, x5, x6, step, count ):
    
    if count == 4:
        args = (1 + x1 * step, 0 + x2 * step, 0 + x3 * step, 0 + x4 * step, 0 + x5 * step, 0 + x6 * step )
        p0, p1, p2, p3, p4, p5 = F(*args)
    else:
        args = (1 + x1 * step/2, 0 + x2 * step/2, 0 + x3 * step/2, 0 + x4 * step/2, 0 + x5 * step/2, 0 + x6 * step/2 )
        p0, p1, p2, p3, p4, p5 = F(*args)
    return [ p0, p1, p2, p3, p4, p5 ]

def loop( x1, x2, x3, x4, x5, x6 ):
    args = ( x1, x2, x3, x4, x5, x6 )
    k1 = F(*args)
    k2 = interations( k1[0], k1[1], k1[2], k1[3], k1[4], k1[5], 0.1, 2 )
    k3 = interations( k2[0], k2[1], k2[2], k2[3], k2[4], k2[5], 0.1, 3 )
    k4 = interations( k3[0], k3[1], k3[2], k3[3], k3[4], k3[5], 0.1, 4 )

    y = [0.1/6*(a + 2*b + 2*c + d) for a,b,c,d in zip( k1,k2,k3,k4 )]

    return [x1 + y[0], x2 + y[1], x3 + y[2], x4 + y[3], x5 + y[4], x6 + y[5] ]

y = [ [ 1, 0, 0, 0, 0, 0 ] ]

for i in range(N):
    resultado = loop( y[i][0], y[i][1], y[i][2], y[i][3], y[i][4], y[i][5] )
    vetor = [ resultado[0], resultado[1], resultado[2], resultado[3], resultado[4], resultado[5] ]
    y.append(vetor)

y = np.array(y)

freq = np.arange(0, N*h+h, step=h)

rho11 = np.real(y[:,0])
plt.plot(freq, rho11, label="p11")
rho22 = np.real(y[:,1])
plt.plot(freq, rho22, label="p22")
rho33 = np.real(y[:,2])
plt.plot(freq, rho33, label="p33")

print( "\n\n SOMA DE POPULAÇÃO: ", rho33[-1] + rho22[-1] + rho11[-1], "\n\n" )

# sigma_12 = y[ : , 3]
# Conj_sigma_12 = np.conjugate(sigma_12)
# fwm12 = ( sigma_12 * np.conjugate(sigma_12) ) ** 2
# plt.plot(freq, np.real(fwm12), label="sigma_12")

# sigma_23 = y[ : , 5]
# Conj_sigma_23 = np.conjugate(sigma_23)
# fwm23 = ( sigma_23 * np.conjugate(sigma_23) ) ** 2
# plt.plot(freq, np.real(fwm23), label="sigma_23")

plt.legend( loc = "best" )
plt.show()
