import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gas_constant
from numba import njit, jit
import time


temp = 327.15
G = 1e-0        # ORDEM DE TX DECAIMENTO GAMMA
a = 9e-1        # ORDEM DE GAMMA_A (CONTROLE)
b = 3e-1        # ORDEM DE GAMMA_B (PROVA)
c = 1e-3        # ORDEM DE GAMMA_C (FEMTO)

k = 0e-4        # ORDEM DE TX DECAIMENTO g13, g24
m = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
r = 1e-2        # ORDEM DO VALOR DE Pi

N = 10000       # TAMANHO DO VETOR TEMPO
j = 100         # LIMITES DO VETOR INICIAL E FINAL

inicial = -j
final = j
h = (final - inicial)/N

g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G
M = (0.278 * 86.91 + 0.722 * 84.91) / 1e3
kcw = 2*np.pi / 780e-9
kfs = 2*np.pi / 795e-9

def dVdt(t, V, delta, p11_0, p33_0, Dcw):

    p11 = V[0]
    p22 = V[1]
    p33 = V[2]
    p44 = V[3]
    s12 = V[4]
    s13 = V[5]
    s14 = V[6]
    s23 = V[7]
    s24 = V[8]
    s34 = V[9]
    del_cw, del_fs = delta[0], delta[1]

    f11 = G/2 * (p22 + p44) + 1j * ( np.conj( Omega_a ) * np.conj( s12 ) - Omega_a * s12 + np.conj( Omega_c ) * np.conj( s14 ) - Omega_c * s14 ) - Pi * ( p11 - p11_0 )
    f22 = - ( G + Pi ) * p22 + 1j * ( Omega_a * s12 - np.conj( Omega_a ) * np.conj( s12 ) + Omega_b * np.conj( s23 ) - np.conj( Omega_b ) * s23 ) 
    f33 = G/2 * ( p22 + p44 ) + 1j * ( np.conj( Omega_b ) * s23 - Omega_b * np.conj( s23 ) + np.conj( Omega_c ) * np.conj( s34 ) - Omega_c * s34 )  - Pi * ( p33 - p33_0 )
    f44 = - (  G + Pi ) * p44 + 1j * ( Omega_c * s14 - np.conj( Omega_c ) * np.conj( s14 ) + Omega_c * s34 - np.conj( Omega_c ) * np.conj( s34 ) )
    f12 = ( 1j * ( del_cw - Dcw ) - g12 - Pi ) * s12 + 1j * ( np.conj( Omega_a ) * ( p22 - p11 ) - np.conj( Omega_b ) * s13 + np.conj( Omega_c ) * np.conj( s24 ) )
    f13 = - ( g13 + Pi ) * s13 + 1j * ( np.conj( Omega_a ) * s23 + np.conj( Omega_c ) * np.conj( s34 ) - Omega_b * s12 - Omega_c * s14 )
    f14 = ( 1j * ( del_fs - (kfs/kcw) * Dcw ) - g14 - Pi ) * s14 + 1j * ( np.conj( Omega_a ) * s24 + np.conj( Omega_c ) * ( p44 - p11 ) - np.conj( Omega_c ) * s13 )
    f23 = (-1j * (del_cw - Dcw) - g23 - Pi ) * s23 + 1j * ( Omega_a * s13 - Omega_b * ( p22 - p33 ) - Omega_c * s24 )
    f24 = ( 1j * ( ( del_fs - del_cw ) + ( 1 - (780/795) ) * Dcw ) - g24 - Pi ) * s24 + 1j * ( Omega_a * s14 + Omega_b * s34 - np.conj( Omega_c ) * np.conj( s12 ) - np.conj( Omega_c ) * s23 )
    f34 = ( 1j * ( del_fs - (kfs/kcw) * Dcw ) - g34 - Pi ) * s34 + 1j * ( np.conj( Omega_b ) * s24 + np.conj( Omega_c ) * ( p44 - p33 ) - np.conj( Omega_c ) * np.conj( s13 ) )

    return np.array( [ t, f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ], complex )

# ==========================================================================
""" RUNGE-KUTTA 4th ORDER, by LEANDRO """
# ==========================================================================

def rk4( t, V, d, c1, c2, v ):
    k1 = h * dVdt( t, V, d, a, b, v )
    k2 = h * dVdt( t + h/2, V + k1[1:]/2, d, c1, c2, v )
    k3 = h * dVdt( t + h/2, V + k2[1:]/2, d, c1, c2, v )
    k4 = h * dVdt( t + h, V + k3[1:], d, c1, c2, v )
    V += ( k1[1:] + 2*k2[1:] + 2*k3[1:] + k4[1:] ) / 6
    return V


@njit(fastmath=True)
def loop( V, delta, a, b):
    t = 0.0
    v = 60.
    for i in range(N):
        t = t + i*h
        ans = rk4( t, V, delta, a, b, v)
        V = ans
    return V

# ==========================================================================
""" CONDICOES INICIAIS """
# ==========================================================================
p11_0, p33_0 = 0.5, 0.5
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = p11_0, 0., p33_0, 0., 0., 0., 0., 0., 0., 0.
V0 = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )
dd = np.array( [ -100, 0.0 ] )   # dd[0] = delta_cw; dd[1] = delta_fs

# Dv = np.linspace( -60.0, 60.0, N + 1, endpoint=True)
# Du = np.sqrt( temp * gas_constant / M ) * kcw
# f_doppler = 1. / np.sqrt( 2 * np.pi * ( Du / kcw)**2 ) * np.exp( - 0.5*(Dv/Du)**2 )

# ==========================================================================
""" CALCULO DE COERENCIAS """
# ==========================================================================

print( loop(V0, dd, p11_0, p33_0) )


# fwmFS = ( sigma * np.conj( sigma ) )
# plt.plot( delta_, np.real(fwmFS), label = 'numeric-RK4', linewidth=1.0, color='r' )
# plt.title( "$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{a} = \\delta_{b} = $" + str(dd[0]) )
# plt.xlabel('$\\delta_{fs} (1/\\Gamma)$')
# plt.ylabel('$\\left|\\sigma_{14}\\right|^2$')
# plt.legend(loc="best")
# plt.show()