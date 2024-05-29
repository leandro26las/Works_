import numpy as np
import matplotlib.pyplot as plt

G = 1.          # ORDEM DE TX DECAIMENTO GAMMA
a = 1e-2        # ORDEM DE GAMMA_A (CONTROLE)
b = 1e-2        # ORDEM DE GAMMA_B (PROVA)
c = 1e-3        # ORDEM DE GAMMA_C (FEMTO)
s = 0e-2        # ORDEM DE GAMMA_S (SINAL)
k = 1e-4        # ORDEM DE TX DECAIMENTO g13, g24
m = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
r = 0e-3        # ORDEM DO VALOR DE Pi


N = 1000        # TAMANHO DO VETOR TEMPO
j = 4           # LIMITES DO VETOR INICIAL E FINAL

inicial = -j
final = j
h = (final - inicial)/N

g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_s = s * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G
p1 = 1.
p3 = 0.

def F(t, V, delta):

    f11 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_a ) * np.conj( V[4]) - Omega_a * V[4] + np.conj( Omega_s ) * np.conj( V[6] ) - Omega_s * V[6] ) - Pi * ( V[0] - p1 )
    f22 = - ( G + Pi ) * V[1] + 1j * ( Omega_a * V[4] - np.conj( Omega_a ) * np.conj( V[4]) + Omega_b * np.conj( V[7] ) - np.conj( Omega_b ) * V[7] ) 
    f33 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_b ) * V[7] - Omega_b * np.conj( V[7] ) + np.conj( Omega_c ) * np.conj( V[9] ) - Omega_c * V[9] )  - Pi * ( V[2] - p3 )
    f44 = - (  G + Pi ) * V[3] + 1j * ( Omega_s * V[6] - np.conj( Omega_s ) * np.conj( V[6] ) + Omega_c * V[9] - np.conj( Omega_c ) * np.conj( V[9] ) )
    f12 = ( 1j * delta[0] - g12 - Pi ) * V[4] + 1j * ( np.conj( Omega_a ) * ( V[1] - V[0] ) - np.conj( Omega_b ) * V[5] + np.conj( Omega_s ) * np.conj( V[8] ) )
    f13 = ( - g13 - Pi ) * V[5] + 1j * ( np.conj( Omega_a ) * V[7] + np.conj( Omega_s ) * np.conj( V[9] ) - Omega_b * V[4] - Omega_c * V[6] )
    f14 = ( 1j * delta[1] - g14 - Pi ) * V[6] + 1j * ( np.conj( Omega_a ) * V[8] + np.conj( Omega_s ) * ( V[3] - V[0] ) - np.conj( Omega_c ) * V[5] )
    f23 = (-1j * delta[0] - g23 - Pi ) * V[7] + 1j * ( Omega_a * V[5] - Omega_b * ( V[1] - V[2] ) - Omega_c * V[8])
    f24 = ( 1j * ( delta[1] - delta[0] ) - g24 - Pi ) * V[8] + 1j * ( Omega_a * V[6] + Omega_b * V[9] - np.conj( Omega_s ) * np.conj( V[4] ) - np.conj( Omega_c ) * V[7] )
    f34 = ( 1j * delta[1] - g34 - Pi ) * V[9] + 1j * ( np.conj( Omega_b ) * V[8] + np.conj( Omega_c ) * ( V[3] - V[2] ) - np.conj( Omega_s ) * np.conj( V[5] ))

    return np.array( [t, f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ] )

# ==========================================================================
""" RUNGE-KUTTA 4th ORDER, by LEANDRO """
# ==========================================================================

def rk4( t, M, d ):
    k1 = h * F( t, M, d )
    k2 = h * F( t + h / 2, M + k1[1:]/2, d )
    k3 = h * F( t + h / 2, M + k2[1:]/2, d )
    k4 = h * F( t + h , M + k3[1:], d )
    Y0 = np.append(t,M)
    Y = Y0 + ( k1 + 2 * (k2 + k3) + k4 )/6
    return Y

# ==========================================================================
""" RUNGE-KUTTA 4th ORDER, by ScyPy"""
# ==========================================================================



# ==========================================================================
""" CONDICOES INICIAIS """
# ==========================================================================

t = np.array([0.0])
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.
eq = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )
V0 = np.append(t, eq)

dd = np.array( [ 0.0, 0.0 ] )   # dd[0] = delta_cw; dd[1] = delta_fs

# ==========================================================================
""" CALCULO DE COERENCIAS """
# ==========================================================================

sigma = np.zeros(N, dtype=complex)
rho11 = np.zeros(N, dtype=complex)
rho22 = np.zeros(N, dtype=complex)
rho33 = np.zeros(N, dtype=complex)
rho44 = np.zeros(N, dtype=complex)
delta_ = np.zeros(N)

for q in range(N):
    dd[1] = inicial + q * h
    for i in range(N):
        ans = rk4( V0[0] + i*h, V0[1:], dd )
        V0 = ans
    rho11[q] = np.real(ans[1])
    rho22[q] = np.real(ans[2])
    rho33[q] = np.real(ans[3])
    rho44[q] = np.real(ans[4])
    sigma[q] = ans[7]   # COERENCIA sigma_14
    delta_[q] = dd[1]
    V0 = np.append(t, eq)

fwmFS = ( sigma * np.conj( sigma ) ) ** 2
plt.plot( delta_, np.real(fwmFS), label = 'numeric-RK4', linewidth=1.0, color='r' )
plt.title( "$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{a} = \\delta_{b} = $" + str(dd[0]) )
plt.xlabel('$\\delta_{fs} (1/\\Gamma)$')
plt.ylabel('$\\left|\\sigma_{14}\\right|^2$')



# ==========================================================================
""" CALCULO ANALITICO DA COERENCIA DA MISTURA """
# ==========================================================================

for q in range( len(delta_) ):
    A = ( 1j * delta_[q] - g14 - Pi )*( 1j * ( delta_[q] - dd[0] ) - g24 - Pi )
    B = ( 1j * delta_[q] - g14 - Pi )*( - g13 - Pi )
    F = 1. + ( B * abs( Omega_a )**2 + A * abs( Omega_c )**2 ) / ( A * B ) )
    sigma[q] = -1j* ( (np.conj(Omega_a) * ( rho44[q] - rho11[q] ) ) / ( 1j* delta_[q] - g14 - Pi) ) + ( np.conj(Omega_a) * np.conj(Omega_c) * Omega_b ) * (
              -1j * ( ( ( A + B ) * ( rho33[q] - rho22[q] ) ) / ( F * A * B * ( - 1j * dd[0] - g23 - Pi ) ) )
              +1j * ( ( rho22[q] - rho11[q] ) / ( B * F * ( 1j * dd[0] - g12 - Pi ) ) )  
              +1j * ( ( rho44[q] - rho33[q] ) / ( A * F * ( 1j * delta_[q] - g34 - Pi ) ) )
    )

fwmFSAnalitico = ( sigma * np.conj( sigma ) ) ** 2
plt.plot( delta_, np.real(fwmFSAnalitico)/1e-4, linewidth=1.0, color='k', label='Analitical' )

plt.legend(loc="best")
plt.show()