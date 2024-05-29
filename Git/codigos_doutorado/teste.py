import numpy as np
inicial = -1
final = 1
N = 10
h = ( final - inicial ) / N

def func( t, X ):
    f1 = X[1]
    f2 = 6*X[0] - X[1]
    return np.array( [ t, f1, f2 ] )

def rk4( t, M ):
    k1 = h * func( t, M )
    k2 = h * func( t + h / 2, M + k1[1:]/2 )
    k3 = h * func( t + h / 2, M + k2[1:]/2 )
    k4 = h * func( t + h , M + k3[1:] )
    Y0 = np.append(t,M)
    Y = Y0 + ( k1 + 2 * (k2 + k3) + k4 )/6
    return Y

V0 = np.array( [0.0, 1.0, 0.0 ] )
vector = [V0]
for i in range(N):
    ans = rk4( V0[0] + i*h, V0[1:] )
    vector.append(ans)
    V0 = ans

vector = np.asmatrix(vector)

print(vector)