import numpy as np
import matplotlib.pyplot as plt

G = 1           # ORDEM DE TX DECAIMENTO GAMMA
a = 1.5e0        # ORDEM DE GAMMA_A (CONTROLE)
b = 1e-1        # ORDEM DE GAMMA_B (PROVA)
c = 1.5e0        # ORDEM DE GAMMA_C (FEMTO)
k = 1e-4        # ORDEM DE TX DECAIMENTO g13, g24
n = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
m = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
r = 0e-3        # ORDEM DO VALOR DE Pi

N = 2000       # TAMANHO DO VETOR TEMPO
j = 3.0           # LIMITES DO VETOR INICIAL E FINAL

inicial = -j
final = j
h = (final - inicial)/N
g12 = g23 = g14 = g34 = n * G
g13 = g24 = k * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G
p1 = 1.
p3 = 0.

def F(t, V, delta):

    f11 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_a ) * np.conj( V[4]) - Omega_a * V[4]  ) - Pi * ( V[0] - p1 )
    f22 = - ( G + Pi ) * V[1] + 1j * ( Omega_a * V[4] - np.conj( Omega_a ) * np.conj( V[4]) + Omega_b * np.conj( V[7] ) - np.conj( Omega_b ) * V[7] ) 
    f33 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_b ) * V[7] - Omega_b * np.conj( V[7] ) + np.conj( Omega_c ) * np.conj( V[9] ) - Omega_c * V[9] )  - Pi * ( V[2] - p3 )
    f44 = - ( G + Pi ) * V[3] + 1j * ( Omega_c * V[9] - np.conj( Omega_c ) * np.conj( V[9] ) )
    f12 = ( 1j * delta[0] - g12 - Pi ) * V[4] + 1j * ( np.conj( Omega_a ) * ( V[1] - V[0] ) - np.conj( Omega_b ) * V[5] )
    f13 = ( - g13 - Pi ) * V[5] + 1j * ( np.conj( Omega_a ) * V[7] - Omega_b * V[4] - Omega_c * V[6] )
    f14 = ( 1j * delta[1] - g14 - Pi ) * V[6] + 1j * ( np.conj( Omega_a ) * V[8] - np.conj( Omega_c ) * V[5] )
    f23 = (-1j * delta[0] - g23 - Pi ) * V[7] + 1j * ( Omega_a * V[5] - Omega_b * ( V[1] - V[2] ) - Omega_c * V[8])
    f24 = ( 1j * ( delta[1] - delta[0] ) - g24 - Pi ) * V[8] + 1j * ( Omega_a * V[6] + Omega_b * V[9] - np.conj( Omega_c ) * V[7] )
    f34 = ( 1j * delta[1] - g34 - Pi ) * V[9] + 1j * ( np.conj( Omega_b ) * V[8] + np.conj( Omega_c ) * ( V[3] - V[2] ) )

    return np.array( [t, f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ] )

def rk4( t, M, d ):
    k1 = h * F( t, M, d )
    k2 = h * F( t + h / 2, M + k1[1:]/2, d )
    k3 = h * F( t + h / 2, M + k2[1:]/2, d )
    k4 = h * F( t + h , M + k3[1:], d )
    Y0 = np.append(t,M)
    Y = Y0 + ( k1 + 2 * (k2 + k3) + k4 )/6
    return Y

# ==========================================================================
""" CONDICOES INICIAIS """
# ==========================================================================

t = np.array([0.0])
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.
eq = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )
V0 = np.append(t, eq)
v = [V0]

dd = np.array( [ 0.0, 0.0 ] )

# ==========================================================================
""" CALCULO DE POPULACAO """
# ==========================================================================

for i in range(N):
    ans = rk4( V0[0] + i*h, V0[1:], dd )
    v.append(ans)
    V0 = ans
v = np.array(v)
time = np.real(v[:,0])

rho11 = np.real(v[:,1])
rho22 = np.real(v[:,2])
rho33 = np.real(v[:,3])
rho44 = np.real(v[:,4])

plt.plot( time, rho11, label="p11")
plt.plot( time, rho22, label="p22")
plt.plot( time, rho33, label="p33")
plt.plot( time, rho44, label="p44")
# plt.xlim((0,))
# plt.title( "$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{c} = \\delta_{p} = $" + str(dd[0]) + ", $\\delta_{fs} = $" + str(dd[2]) )
plt.legend(loc="best")
plt.show()
print("\n\n SOMA DAS POPULAÇÕES: ", rho11[-1] + rho22[-1] + rho33[-1] + rho44[-1], "\n\n")