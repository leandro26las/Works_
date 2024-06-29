import numpy as np
import matplotlib.pyplot as plt

k=4.
inicial = -k
final = k
N = 1000
h = (final - inicial)/N
G = 1.
G21 = G23 = G / 2
g12 = g23 = 0.5 * G21
g13 = 1e-2 * g12
Pi = 0 * G21
a = 0.5
b = 0.5
Omega_a = a * G21
Omega_b = b * G21
p1 = 1.
p3 = 0.

def F(t, V, delta):
    """
        p11 = V[0];
        p22 = V[1];
        p33 = V[2];
        s12 = V[3];
        s13 = V[4];
        s23 = V[5];
        p11_0 = V0[0];
        p33_0 = V0[1];
        delta_a = w21 - wa;
        delta_b = w23 - wb;
        delta_ab = delta_a - delta_b;
    """
    f11 = G21 * V[1] + 1j * ( Omega_a * np.conj( V[3] ) - Omega_a * V[3]  ) - Pi * ( V[0] - p1)
    f22 = - ( G21 + G23 + Pi ) * V[1] + 1j * ( Omega_a * V[3] - Omega_a * np.conj( V[3] ) - Omega_b * V[5] + Omega_b * np.conj( V[5] ) )
    f33 = G23 * V[1] + 1j * ( Omega_b * ( V[5] - np.conj( V[5] ) ) )  - Pi * ( V[2] - p3)
    f12 = ( 1j * delta[0] - g12 - Pi ) * V[3] + 1j * ( Omega_a * ( V[1] - V[0] ) - Omega_b * V[4] )
    f13 = ( 1j * ( delta[0] - delta[1] ) - g13 - Pi ) * V[4] + 1j * ( Omega_a * V[5] - Omega_b * V[3] )
    f23 = (-1j * delta[1] - g23 - Pi ) * V[5] + 1j * ( Omega_a * V[4] - Omega_b * ( V[1] - V[2] ) )

    return np.array( [t, f11, f22, f33, f12, f13, f23 ] )

def rk4( t, M, d ):
    k1 = h * F( t, M, d )
    k2 = h * F( t + h / 2, M + k1[1:]/2, d )
    k3 = h * F( t + h / 2, M + k2[1:]/2, d )
    k4 = h * F( t + h , M + k3[1:], d )
    Y0 = np.append(t,M)
    Y = Y0 + ( k1 + 2 * (k2 + k3) + k4 )/6
    return Y

t = np.array([0.0])

p11 = 1.
p22 = p33 = s12 = s13 = s23 = 0.
eq = np.array( [ p11, p22, p33, s12, s13, s23 ] , dtype = complex )

V0 = np.append( t, eq )
dd = np.array([0.0, 0.0])

v = [V0]
for i in range(N+1):
    ans = rk4( V0[0] + i*h, V0[1:], dd )
    v.append(ans)
    V0 = ans

v = np.array(v)

time  = np.real( v[ : , 0 ] )
rho11 = np.real( v[ : , 1 ] )
rho22 = np.real( v[ : , 2 ] )
rho33 = np.real( v[ : , 3 ] )

plt.plot( time, rho11, label="p11")
plt.plot( time, rho22, label="p22")
plt.plot( time, rho33, label="p33")

print("\n\n SOMA DAS POPULAÇÕES: ", rho11[-1] + rho22[-1] + rho33[-1], "\n\n")


# # S = np.array( [ [ p11, p22, p33, s12, s13, s23 ] ], dtype=complex )
# # sigma12 = np.zeros(passo)
# # sigma13 = np.zeros(passo)
# # sigma23 = np.zeros(passo)

# # for j in range(passo):
# #     delta1 = inicial + j * h
# #     delta2 = delta1
# #     for i in range(N):
# #         ans = loop( S[i], S[0], delta1, delta2 )
# #         S = np.append( S, [ans], axis=0 )
# #     sigma12 = S[ -1 , 3]
# #     sigma13 = S[ -1 , 4]
# #     sigma23 = S[ -1 , 5]

# # delta = np.arange(inicial, ( 1 - inicial )* h , h)
# # cc = ( sigma23 * np.conj( sigma23 ) ) ** 2
# # plt.scatter( delta, np.real(cc) )

plt.legend(loc="best")
plt.show()