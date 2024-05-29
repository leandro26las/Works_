import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 1.          # ORDEM DE TX DECAIMENTO GAMMA
a = 1e-0         # ORDEM DE GAMMA_A (CONTROLE)
b = 5e-2        # ORDEM DE GAMMA_B (PROVA)
c = 1e-0         # ORDEM DE GAMMA_C (FEMTO)
s = 0e-2        # ORDEM DE GAMMA_S (SINAL)
k = 0e-4        # ORDEM DE TX DECAIMENTO g13, g24
m = 0.5         # ORDEM DE TX DECAIMENTO COERENCIAS
r = 1e-7        # ORDEM DO VALOR DE Pi
N = 25          # TAMANHO DO VETOR TEMPO

inicial = -30.
final = 30
passo = 12000
delta = np.linspace(inicial, final, passo, endpoint=True)


g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_s = s * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G
p11_0 = 1.
p33_0 = 0.


def F(t, V, d_cw, d_fs ):

    f11 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_a ) * np.conj( V[4]) - Omega_a * V[4] + np.conj( Omega_s ) * np.conj( V[6] ) - Omega_s * V[6] ) - Pi * ( V[0] - p11_0 )
    f22 = - ( G + Pi ) * V[1] + 1j * ( Omega_a * V[4] - np.conj( Omega_a ) * np.conj( V[4]) + Omega_b * np.conj( V[7] ) - np.conj( Omega_b ) * V[7] ) 
    f33 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_b ) * V[7] - Omega_b * np.conj( V[7] ) + np.conj( Omega_c ) * np.conj( V[9] ) - Omega_c * V[9] )  - Pi * ( V[2] - p33_0 )
    f44 = - (  G + Pi ) * V[3] + 1j * ( Omega_s * V[6] - np.conj( Omega_s ) * np.conj( V[6] ) + Omega_c * V[9] - np.conj( Omega_c ) * np.conj( V[9] ) )
    f12 = ( 1j * d_cw - g12 - Pi ) * V[4] + 1j * ( np.conj( Omega_a ) * ( V[1] - V[0] ) - np.conj( Omega_b ) * V[5] + np.conj( Omega_s ) * np.conj( V[8] ) )
    f13 = ( - g13 - Pi ) * V[5] + 1j * ( np.conj( Omega_a ) * V[7] + np.conj( Omega_s ) * np.conj( V[9] ) - Omega_b * V[4] - Omega_c * V[6] )
    f14 = ( 1j * d_fs - g14 - Pi ) * V[6] + 1j * ( np.conj( Omega_a ) * V[8] + np.conj( Omega_s ) * ( V[3] - V[0] ) - np.conj( Omega_c ) * V[5] )
    f23 = (-1j * d_cw - g23 - Pi ) * V[7] + 1j * ( Omega_a * V[5] - Omega_b * ( V[1] - V[2] ) - Omega_c * V[8])
    f24 = ( 1j * ( d_fs - d_cw ) - g24 - Pi ) * V[8] + 1j * ( Omega_a * V[6] + Omega_b * V[9] - np.conj( Omega_s ) * np.conj( V[4] ) - np.conj( Omega_c ) * V[7] )
    f34 = ( 1j * d_fs - g34 - Pi ) * V[9] + 1j * ( np.conj( Omega_b ) * V[8] + np.conj( Omega_c ) * ( V[3] - V[2] ) - np.conj( Omega_s ) * np.conj( V[5] ))
    
    return [ f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ]

# ==========================================================================
""" CONDICOES INICIAIS """
# ==========================================================================

t_int = np.array([0, N])
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.
cond_i = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )

# ==========================================================================
""" CALCULO DE POPULACOES NUMERICAMENTE """
# ==========================================================================

sol = solve_ivp(F, t_int, y0 = cond_i, t_eval=np.linspace(t_int[0], t_int[1], int(10 * N) ), args=(0.0,0.0))

plt.plot(sol.t, np.real(sol.y[0,:]), label='p11 - Numerico')
plt.plot(sol.t, np.real(sol.y[1,:]), label='p22 - Numerico')
plt.plot(sol.t, np.real(sol.y[2,:]), label='p33 - Numerico')
plt.plot(sol.t, np.real(sol.y[3,:]), label='p44 - Numerico')

print('SOMA POPULACOES CALCULO NUMERICO: ' + str( np.real(sol.y[0,-1]) + np.real(sol.y[1,-1]) + np.real(sol.y[2,-1]) + np.real(sol.y[3,-1]) ) + '\n' )

# ==========================================================================
""" CALCULO DE POPULACOES ANALITICAMENTE SEM TEMPO DE VOO """
# ==========================================================================

# dcw = 0.0
# dfs = 0.0

# K1 = ( 1/ ((1j * dcw - g12 ) * G ) - 1/( (1j * dcw + g12 ) * G ) )
# C1 = (1. - ( abs(Omega_a)**2 * K1 ) )
# K2 = ( 1/ ( ( 1j * dfs - g34 ) * G ) - 1/ ( (1j * dfs + g34 ) * G ) )
# C2 = (1. - ( abs(Omega_c)**2 * K2 ) )

# p11_e = ( ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p33_e = (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p22_e = ( - ( abs(Omega_c)**2 / C2 ) * K2 ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p44_e = ( - ( abs(Omega_c)**2 / C2 ) * K2 ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)

# plt.axhline(y= np.real(p11_e), xmin = min(sol.t), xmax = max(sol.t), label='p11 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p22_e), xmin = min(sol.t), xmax = max(sol.t), label='p22 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p33_e), xmin = min(sol.t), xmax = max(sol.t), label='p33 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p44_e), xmin = min(sol.t), xmax = max(sol.t), label='p44 - Analitico', c = 'k', ls = '--')

# plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, $\\Pi = $" + str(r) + "$\\Gamma$ \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{cw} = $" + str(0.0) + ", $\\delta_{fs} = $" + str(0.0))
# plt.xlabel('tempo (u.a)')
# plt.ylabel('Populacao')
# plt.legend(loc='best')
# plt.show()

# print('SOMA POPULACOES CALCULO NUMERICO: ' + str( p11_e + p22_e + p33_e + p44_e ))

# ==========================================================================
""" CALCULO DE POPULACOES ANALITICAMENTE COM TEMPO DE VOO """
# ==========================================================================
dcw = 0.0
dfs = 0.0

K1 = ( 1/(1j * dcw - g12 - Pi) - 1/(1j * dcw + g12 + Pi) )
C1 = (1. - ( abs(Omega_a)**2 * K1 )/(G + Pi) )
K2 = ( 1/(1j * dfs - g34 - Pi) - 1/(1j * dfs + g34 + Pi) )
C2 = (1. - ( abs(Omega_c)**2 * K2 )/(G + Pi) )
R = ( 0.5 * G * ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) - ( abs(Omega_a)**2 * K1 / C1 ) + Pi  )
S = ( 0.5 * G * ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) - ( abs(Omega_c)**2 * K2 / C2 ) + Pi  )
T = 1. - ( ( G * G * abs(Omega_a)**2 * abs(Omega_c)**2 * (K1/C1) * (K2/C2) ) / ( 4 * (G + Pi)**2 * R * S ) )

p11_e = (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0
p22_e = - ( ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) ) * ( (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0 )
p33_e = ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0
p44_e = - ( ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) ) * ( ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0 )

print('SOMA POPULACOES CALCULO NUMERICO: ' + str( p11_e + p22_e + p33_e + p44_e ))

plt.axhline(y= np.real(p11_e), xmin = min(sol.t), xmax = max(sol.t), label='p11 - Analitico', c = 'k', ls = '--')
plt.axhline(y= np.real(p22_e), xmin = min(sol.t), xmax = max(sol.t), label='p22 - Analitico', c = 'k', ls = '--')
plt.axhline(y= np.real(p33_e), xmin = min(sol.t), xmax = max(sol.t), label='p33 - Analitico', c = 'k', ls = '--')
plt.axhline(y= np.real(p44_e), xmin = min(sol.t), xmax = max(sol.t), label='p44 - Analitico', c = 'k', ls = '--')

plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, $\\Pi = $" + str(r) + "$\\Gamma$ \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{cw} = $" + str(0.0) + ", $\\delta_{fs} = $" + str(0.0))
plt.legend(loc='best')
plt.show()

# ==========================================================================
""" CALCULO DE TRANSMISSAO NUMERICAMENTE """
# ==========================================================================

# trans = np.imag(sol.y[7,:])

# plt.plot(sol.t, trans, label='Transmissao')
# plt.title("$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{cw} = $" + str(0.0) + "$\\delta_{cw} = $" + str(0.0))
# plt.xlabel(' tempo ')
# plt.ylabel('Im($ \\sigma_{23} $)')
# plt.legend(loc='best')
# plt.show()

# ==========================================================================
""" CALCULO NUMERICO PARA COERENCIA """
# ==========================================================================

sinal = np.zeros(passo, dtype=complex)
del_ = np.zeros(passo, dtype=complex)

for n in range(passo):
    sol = solve_ivp(F, t_int, y0 = cond_i, t_eval=np.linspace(t_int[0], t_int[1], int(10 * N) ), args=(0.0,delta[n]) )
    sinal[n] = sol.y[6,-1]
    # transmition[n] = np.imag(sol.y[7,-1])

signal = abs(sinal)**2

plt.plot(delta, signal, label='FWM')
plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, $\\Pi = $" + str(r) + "$\\Gamma$ \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{cw} = $" + str(0.0) )
plt.xlabel(' $ \\delta_{fs} $ ')
plt.ylabel('$ \\left| \\sigma_{14} \\right|^2 $')
plt.legend(loc='best')
plt.show()

# ==========================================================================
""" SOLUCAO ANALITICA COERENCIA COM TEMPO DE VOO """
# ==========================================================================
dcw = 0.0
dfs = 0.0

K1 = ( 1/(1j * dcw - g12 - Pi) - 1/(1j * dcw + g12 + Pi) )
C1 = (1. - ( abs(Omega_a)**2 * K1 )/(G + Pi) )
K2 = ( 1/(1j * dfs - g34 - Pi) - 1/(1j * dfs + g34 + Pi) )
C2 = (1. - ( abs(Omega_c)**2 * K2 )/(G + Pi) )
R = ( 0.5 * G * ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) - ( abs(Omega_a)**2 * K1 / C1 ) + Pi  )
S = ( 0.5 * G * ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) - ( abs(Omega_c)**2 * K2 / C2 ) + Pi  )
T = 1. - ( ( G * G * abs(Omega_a)**2 * abs(Omega_c)**2 * (K1/C1) * (K2/C2) ) / ( 4 * (G + Pi)**2 * R * S ) )

p11_e = (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0
p22_e = - ( ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) ) * ( (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0 )
p33_e = ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0
p44_e = - ( ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) ) * ( ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0 )
