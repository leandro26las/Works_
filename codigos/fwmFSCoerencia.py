import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gas_constant
from numba import njit

G = 1e-0        # ORDEM DE TX DECAIMENTO GAMMA
a = 9e-1        # ORDEM DE GAMMA_A (CONTROLE)
b = 1e-2        # ORDEM DE GAMMA_B (PROVA)
c = 7e-1        # ORDEM DE GAMMA_C (FEMTO)

k = 0e-4        # ORDEM DE TX DECAIMENTO g13, g24
m = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
r = 1e-2        # ORDEM DO VALOR DE Pi

N = 10000       # TAMANHO DO VETOR TEMPO
inicial = -100.
final = 100.
h = (final - inicial)/N
dfs = 0.0
dcw = np.linspace(inicial, final, N + 1, endpoint=True )

g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G

kcw = 1. # 2*np.pi / 780
kfs = 780/795 # 2*np.pi / 795
temp = 327.15
M = (0.278 * 86.91 + 0.722 * 84.91) / 1e3
Dv = np.linspace( -5.0, 5.0, 1001, endpoint=True)
Du = G # np.sqrt( temp * gas_constant / M ) * kcw
f_doppler = 1. / np.sqrt( 2 * np.pi * ( Du )**2 ) * np.exp( - 0.5 * (Dv/Du)**2 )
# plt.plot(Dv, f_doppler)

@njit(fastmath=True)
def F(t, V, del_cw, del_fs, p11_0, p33_0, D_vel):
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

    f11 = G/2 * (p22 + p44) + 1j * ( np.conj( Omega_a ) * np.conj( s12 ) - Omega_a * s12 + np.conj( Omega_c ) * np.conj( s14 ) - Omega_c * s14 ) - Pi * ( p11 - p11_0 )
    f22 = - ( G + Pi ) * p22 + 1j * ( Omega_a * s12 - np.conj( Omega_a ) * np.conj( s12 ) + Omega_b * np.conj( s23 ) - np.conj( Omega_b ) * s23 ) 
    f33 = G/2 * ( p22 + p44 ) + 1j * ( np.conj( Omega_b ) * s23 - Omega_b * np.conj( s23 ) + np.conj( Omega_c ) * np.conj( s34 ) - Omega_c * s34 )  - Pi * ( p33 - p33_0 )
    f44 = - (  G + Pi ) * p44 + 1j * ( Omega_c * s14 - np.conj( Omega_c ) * np.conj( s14 ) + Omega_c * s34 - np.conj( Omega_c ) * np.conj( s34 ) )
    f12 = ( 1j * ( del_cw - D_vel ) - g12 - Pi ) * s12 + 1j * ( np.conj( Omega_a ) * ( p22 - p11 ) - np.conj( Omega_b ) * s13 + np.conj( Omega_c ) * np.conj( s24 ) )
    f13 = - ( g13 + Pi ) * s13 + 1j * ( np.conj( Omega_a ) * s23 + np.conj( Omega_c ) * np.conj( s34 ) - Omega_b * s12 - Omega_c * s14 )
    f14 = ( 1j * ( del_fs - (kfs/kcw) * D_vel ) - g14 - Pi ) * s14 + 1j * ( np.conj( Omega_a ) * s24 + np.conj( Omega_c ) * ( p44 - p11 ) - np.conj( Omega_c ) * s13 )
    f23 = (-1j * (del_cw - D_vel) - g23 - Pi ) * s23 + 1j * ( Omega_a * s13 - Omega_b * ( p22 - p33 ) - Omega_c * s24 )
    f24 = ( 1j * ( ( del_fs - del_cw ) + ( 1 - (kfs/kcw) ) * D_vel ) - g24 - Pi ) * s24 + 1j * ( Omega_a * s14 + Omega_b * s34 - np.conj( Omega_c ) * np.conj( s12 ) - np.conj( Omega_c ) * s23 )
    f34 = ( 1j * ( del_fs - (kfs/kcw) * D_vel ) - g34 - Pi ) * s34 + 1j * ( np.conj( Omega_b ) * s24 + np.conj( Omega_c ) * ( p44 - p33 ) - np.conj( Omega_c ) * np.conj( s13 ) )

    return np.array( [f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ] )

# ==========================================================================
""" RUNGE-KUTTA 4th ORDER, by LEANDRO """
# ==========================================================================

# @njit(fastmath=True)
# def loop_rk4(R, a, b):
#     for it in range( N + 1 ):
#         k1 = h * F( it , R[it], dcw[0], dfs, a, b, Dv[0] )
#         k2 = h * F( it + h/2, R[it] + k1/2, dcw[0], dfs, a, b, Dv[0])
#         k3 = h * F( it + h/2, R[it] + k2/2, dcw[0], dfs, a, b, Dv[0])
#         k4 = h * F( it + h, R[it] + k3, dcw[0], dfs, a, b, Dv[0] )
#         R[it+1] = R[it] + ( k1 + 2*k2 + 2*k3 + k4 ) / 6
#         it += h
#     return R

def integrate(arr, integer):
    soma = arr * f_doppler[integer]
    return soma

@njit(fastmath=True)
def loop_rk4(R, V, velocity, a, b):
    for id in range(len(dcw)+1):
        for iv in range(len(Dv)+1):
            rk4 = V
            for it in range(N+1):
                k1 = h * F( it*h , rk4, dcw[id], dfs, a, b, Dv[iv] )
                k2 = h * F( it*h + h/2, rk4 + k1/2, dcw[id], dfs, a, b, Dv[iv])
                k3 = h * F( it*h + h/2, rk4 + k2/2, dcw[id], dfs, a, b, Dv[iv])
                k4 = h * F( it*h + h, rk4 + k3, dcw[id], dfs, a, b, Dv[iv] )
                rk4 = rk4 + ( k1 + 2*k2 + 2*k3 + k4 ) / 6
            velocity[iv] = rk4 * f_doppler[iv]
        R[id] = np.sum(velocity, axis=0)
    return R

# ==========================================================================
""" CONDICOES INICIAIS """
# ==========================================================================

p11_0, p33_0 = 0.5, 0.5
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = p11_0, 0., p33_0, 0., 0., 0., 0., 0., 0., 0.
Rf = np.zeros( [N+1,10], dtype=complex )
Rv = np.zeros( [N+1,10], dtype=complex )
Vf = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )
Rf[0] = Vf
# ==========================================================================
""" CALCULO DE COERENCIAS """
# ==========================================================================

# M = loop_rk4(Rf, p11_0, p33_0 )

M = loop_rk4( Rf, Vf, Rv, p11_0, p33_0)

# tempo = np.array([i*h for i in range(N+1)])
# rho11 = M[:,0].real
# rho22 = M[:,1].real
# rho33 = M[:,2].real
# rho44 = M[:,3].real

# plt.plot(range(len(tempo)), tempo, linewidth=2.0 );

# plt.plot( tempo, rho11, tempo, rho33, label = 'numeric-RK4', linewidth=2.0 );

# fwmFS = ( sigma * np.conj( sigma ) )
# plt.title( "$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{a} = \\delta_{b} = $" + str(dd[0]) )
# plt.xlabel('$\\delta_{fs} (1/\\Gamma)$')
# plt.ylabel('$\\left|\\sigma_{14}\\right|^2$')
# plt.legend(loc="best")
# plt.show()