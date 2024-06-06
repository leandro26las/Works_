import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import gas_constant

G = 1e-0        # ORDEM DE TX DECAIMENTO GAMMA
a = 9e-1        # ORDEM DE GAMMA_A (CONTROLE)
b = 5e-2        # ORDEM DE GAMMA_B (PROVA)
c = 7e-1        # ORDEM DE GAMMA_C (FEMTO)
s = 0e-2        # ORDEM DE GAMMA_S (SINAL)
m = 5e-1        # ORDEM DE TX DECAIMENTO COERENCIAS
k = 1e-8        # ORDEM DE TX DECAIMENTO g13, g24
r = 1e-8        # ORDEM DO VALOR DE Pi
N = 500         # TAMANHO DO VETOR TEMPO

temp = 327.15   
M = 0.278 * 86.91 + 0.722 * 84.91                            # MASSAS 85Rb: 84.91 u, 87Rb: 86.91 u
v_fx = 60
inicial = -100.                                              # DETUNING INICIAL
final = 100.                                                 # DETUNING FINAL
passo = 12001                                                # TAMANHO DO VETOR DETUNING
dcw = np.linspace(inicial, final, passo, endpoint=True)      # VETOR DETUNING CW
dfs = 0.0                                                    # VETOR DETUNING FS

v_atomos = np.linspace(0.0, v_fx, 3 * int(2 * v_fx) + 1, endpoint=True)                 # VETOR VELOCIDADE DOS ATOMOS
u = np.sqrt( temp * gas_constant / M )                                                   # VELOCIDADE MAIS PROVAVEL DO GAS DE 85,87 RB
F_doppler = 1. / np.sqrt( 2 * np.pi * u**2) * np.exp( - ( v_atomos**2 / 2 * u**2 ))

k12 = k23 = 1.                      # VETOR DE ONDA DO CAMPO CW
k34 = 0.981 * k12                   # VETOR DE ONDA DO CAMPO FS
g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_s = s * G
Omega_a = a * G
Omega_b = b * G
Omega_c = c * G
Pi = r * G
p11_0 = 0.5
p33_0 = 0.5

def F(t, V, d_cw, d_fs, p11_, p33_, vel ):

    f11 = ( G/2 * f22 + G/2 * f44 + 1j * ( np.conj( Omega_a ) * np.conj( V[4] ) - Omega_a * V[4] + np.conj( Omega_s ) * np.conj( V[6] ) - Omega_s * V[6] ) + Pi * p11_ ) / Pi
    f22 = ( 1j * ( Omega_a * V[4] - np.conj( Omega_a ) * np.conj( V[4]) + Omega_b * np.conj( V[7] ) - np.conj( Omega_b ) * V[7] ) ) / ( G + Pi )
    f33 = ( G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_b ) * V[7] - Omega_b * np.conj( V[7] ) + np.conj( Omega_c ) * np.conj( V[9] ) - Omega_c * V[9] ) + Pi * p33_ ) / Pi
    f44 = ( 1j * ( Omega_s * V[6] - np.conj( Omega_s ) * np.conj( V[6] ) + Omega_c * V[9] - np.conj( Omega_c ) * np.conj( V[9] ) ) ) / ( G + Pi )
    f12 = ( 1j * ( np.conj( Omega_a ) * ( V[1] - V[0] ) - np.conj( Omega_b ) * V[5] + np.conj( Omega_s ) * np.conj( V[8] ) ) ) / ( 1j * (d_cw - k12 * vel ) - g12 - Pi )
    f13 = ( 1j * ( np.conj( Omega_a ) * V[7] + np.conj( Omega_s ) * np.conj( V[9] ) - Omega_b * V[4] - Omega_c * V[6] ) ) / ( 1j * (  k12 - k23 ) * vel - g13 - Pi )
    f14 = ( 1j * ( np.conj( Omega_a ) * V[8] + np.conj( Omega_s ) * ( V[3] - V[0] ) - np.conj( Omega_c ) * V[5] ) ) / ( 1j * ( d_fs - ( k12 - k23 + k34 ) * vel ) - g14 - Pi )
    f23 = ( 1j * ( Omega_a * V[5] - Omega_b * ( V[1] - V[2] ) - Omega_c * V[8] ) ) / (-1j * ( d_cw - k23 * vel ) - g23 - Pi )
    f24 = ( 1j * ( Omega_a * V[6] + Omega_b * V[9] - np.conj( Omega_s ) * np.conj( V[4] ) - np.conj( Omega_c ) * V[7] ) ) / ( 1j * ( (d_fs - k34 * vel ) - (d_cw - k23 * vel ) ) - g24 - Pi )
    f34 = ( 1j * ( np.conj( Omega_b ) * V[8] + np.conj( Omega_c ) * ( V[3] - V[2] ) - np.conj( Omega_s ) * np.conj( V[5] ) ) ) / ( 1j * ( d_fs - k34 * vel ) - g34 - Pi )
    
    return [ f11, f22, f33, f44, f12, f13, f14, f23, f24, f34 ]

# ==========================================================================
"""                     CONDICOES INICIAIS                              """
# ==========================================================================

t_int = np.array([0, N])
p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 = p11_0, 0., p33_0, 0., 0., 0., 0., 0., 0., 0.
cond_i = np.array( [ p11, p22, p33, p44, s12, s13, s14, s23, s24, s34 ], dtype = complex )

# ==========================================================================
"""         CALCULO DE POPULACOES NO TEMPO (SOLUCAO NUMERICA)           """
# ==========================================================================

sol = solve_ivp(F, t_int, y0 = cond_i, t_eval=np.linspace(t_int[0], t_int[1], int(10 * N) ), args=( dcw[ int( len(dcw)/2 - 0.5 ) ], dfs, p11_0, p33_0, v_fx ) )

print('SOMA POPULACOES CALCULO NUMERICO: ' + str( np.real(sol.y[0,-1]) + np.real(sol.y[1,-1]) + np.real(sol.y[2,-1]) + np.real(sol.y[3,-1]) ) + '\n' )

# ==========================================================================
""" CALCULO TEMPORAL DE POPULACOES SEM TEMPO DE VOO (SOLUCAO ANALITICA)"""
# ==========================================================================

# K1 = ( 1. / ( ( 1j * ( dcw[int(len(dcw)/2 - 0.5)] - k12 * v_fx ) - g12 ) * G ) - 1. / ( (1j * ( dcw[int(len(dcw)/2 - 0.5)] - k12 * v_fx ) + g12 ) * G ) )
# C1 = ( 1. - ( abs(Omega_a)**2 * K1 ) )
# K2 = ( 1. / ( ( 1j * ( dfs - k34 * v_fx ) - g34 ) * G ) - 1. / ( (1j * ( dfs - k34  * v_fx ) + g34 ) * G ) )
# C2 = ( 1. - ( abs(Omega_c)**2 * K2 ) )

# p11_e = ( ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p33_e = ( ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p22_e = ( - ( abs(Omega_c)**2 / C2 ) * K2 ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
# p44_e = ( - ( abs(Omega_c)**2 / C2 ) * K2 ) * (  ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)

# print('SOMA POPULACOES CALCULO ANALITICO: ' + str( p11_e + p22_e + p33_e + p44_e ) )

# ==========================================================================
""" CALCULO TEMPORAL DE POPULACOES COM TEMPO DE VOO (SOLUCAO ANALITICA) """
# ==========================================================================

# K1 = ( 1. / (1j * ( dcw[int(len(dcw) - 0.5)] - k12 * v ) - g12 - Pi) - 1 / ( 1j * ( dcw[int(len(dcw) - 0.5)] - k12 * v ) + g12 + Pi ) )
# C1 = ( 1. - ( abs(Omega_a)**2 * K1 )/ ( G + Pi ) )
# K2 = ( 1. / (1j * ( dfs - k34 * v ) - g34 - Pi) - 1/(1j * ( dfs - k34 * v ) + g34 + Pi) )
# C2 = ( 1. - ( abs(Omega_c)**2 * K2 )/ ( G + Pi ) )
# R = ( 0.5 * G * ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) - ( abs(Omega_a)**2 * K1 / C1 ) + Pi  )
# S = ( 0.5 * G * ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) - ( abs(Omega_c)**2 * K2 / C2 ) + Pi  )
# T = 1. - ( ( G * G * abs(Omega_a)**2 * abs(Omega_c)**2 * (K1/C1) * (K2/C2) ) / ( 4 * (G + Pi)**2 * R * S ) )

# p11_e = (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0
# p22_e = - ( ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) ) * ( (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0 )
# p33_e = ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0
# p44_e = - ( ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) ) * ( ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0 )

# print('SOMA POPULACOES CALCULO ANALITICO: ' + str( p11_e + p22_e + p33_e + p44_e ))

# ==========================================================================
"""         CALCULO NUMERICO, EM FREQUENCIA, PARA COERENCIA 14           """
# ==========================================================================

# 1st FORMA:    PARA CADA DETUNING, VARRE NAS VELOCIDADES

# s14_vel = np.zeros( ( passo, len(v_atomos) ), dtype=complex)

# for d in range(passo):
#     for v in range( len(v_atomos) ):
#         sol = solve_ivp(F, t_int, y0 = cond_i, t_eval=np.linspace(t_int[0], t_int[1], int(10 * N) ), args=( dcw[d], dfs, p11_0, p33_0, v_atomos[v] ) )
#         s14_vel[d][v] = sol.y[6,-1] 

# signal = abs(s14_vel)**2

# ==========================================================================
""" SOLUCAO ANALITICA, EM FREQUENCIA, DA COERENCIA """
# ==========================================================================

# s14_vel_alt = np.zeros((passo,len(v_atomos)), dtype=complex)

# for n in range(passo):
#     for q in range(len(v_atomos)):
# #         # COM TEMPO DE VOO
# #         # K1 = ( 1. / (1j * ( dcw[n] - k12 * v_atomos[q] ) - g12 - Pi) - 1. / (1j * ( dcw[n] - k12 * v_atomos[q] ) + g12 + Pi) )
# #         # C1 = ( 1. - ( abs(Omega_a)**2 * K1 ) / (G + Pi) )
# #         # K2 = ( 1. / (1j * ( dfs - k34 * v_atomos[q] ) - g34 - Pi) - 1. / (1j * ( dfs - k34 * v_atomos[q] ) + g34 + Pi) )
# #         # C2 = ( 1. - ( abs(Omega_c)**2 * K2 ) / (G + Pi) )
# #         # R = ( 0.5 * G * ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) - ( abs(Omega_a)**2 * K1 / C1 ) + Pi  )
# #         # S = ( 0.5 * G * ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) - ( abs(Omega_c)**2 * K2 / C2 ) + Pi  )
# #         # T = 1. - ( ( G * G * abs(Omega_a)**2 * abs(Omega_c)**2 * (K1/C1) * (K2/C2) ) / ( 4 * (G + Pi)**2 * R * S ) )

# #         # p11_e = ( Pi / (R*T) ) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0
# #         # p22_e = -( ( abs(Omega_a)**2 / (G + Pi) ) * (K1/C1) ) * ( (Pi/(R*T)) * p11_0 - 0.5 * G * ( (abs(Omega_c)**2 * (K2/C2) * Pi ) / ( (G + Pi) * R * S * T) ) * p33_0 )
# #         # p33_e = ( Pi / (S*T) ) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0
# #         # p44_e = - ( ( abs(Omega_c)**2 / (G + Pi) ) * (K2/C2) ) * ( ( Pi/(S*T)) * p33_0 - 0.5 * G * ( ( abs(Omega_a)**2 * (K1/C1) * Pi) / ( (G + Pi) * R* S * T )) * p11_0 )
        
# #         # SEM TEMPO DE VOO
#         K1 = ( 1. / ( ( 1j * ( dcw[n] - k12 * v_atomos[q] ) - g12 ) * G ) - 1. / ( (1j * ( dcw[n] - k12 * v_atomos[q] ) + g12 ) * G ) )
#         C1 = ( 1. - ( abs(Omega_a)**2 * K1 ) )
#         K2 = ( 1. / ( ( 1j * ( dfs - k34 * v_atomos[q] ) - g34 ) * G ) - 1. / ( ( 1j * ( dfs - k34 * v_atomos[q] ) + g34 ) * G ) )
#         C2 = ( 1. - ( abs(Omega_c)**2 * K2 ) )
        
#         p33_e = ( ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) - 2 * K2 * ( abs(Omega_c)**2 / C2 ) + 1 ) ** (-1)
#         p11_e = ( ( abs(Omega_c)**2 / abs(Omega_a)**2 ) * (C1 / C2 ) * ( K2 / K1 ) ) * p33_e
#         p22_e = ( - ( abs(Omega_c)**2 / C2 ) * K2 ) * p33_e
#         p44_e = p22_e
        
#         A = ( 1j * ( dfs - ( k12 - k23 + k34 ) * v_atomos[q] ) - g14 ) * ( 1j * ( (dfs - dcw[n]) - ( k34 - k23 ) * v_atomos[q] ) - g24 )
#         B = ( 1j * ( dfs - ( k12 - k23 + k34 ) * v_atomos[q] ) - g14 ) * ( 1j * ( k12 - k23 ) * v_atomos[q] - g13 )
#         D = 1. + ( B * abs( Omega_a )**2 + A * abs( Omega_c )**2 ) / ( A * B )
#         s14_ = -1j * ( ( np.conj(Omega_s) * ( p44_e - p11_e ) ) / ( 1j * ( dfs - ( k12 - k23 + k34 ) * v_atomos[q] ) - g14 - Pi ) ) + 1j * ( np.conj( Omega_a ) * np.conj( Omega_c ) * Omega_b ) * (
#                      ( ( A + B ) * ( p33_e - p22_e ) ) / ( D * A * B * ( 1j * ( dcw[n] - k23 * v_atomos[q] ) + g23 ) )
#                    + ( p22_e - p11_e ) / ( B * D * ( 1j * ( dcw[n] - k12 * v_atomos[q] ) - g12 ) )
#                    + ( p44_e - p33_e ) / ( A * D * ( 1j * ( dfs - k34 * v_atomos[q] ) - g34 ) )
#         )
#         s14_vel_alt[n][q] = s14_ * ( 1. / np.sqrt( np.pi ) * np.exp( - ( v_atomos[q]**2 ) ) )
# fwm_vel_alt = abs(s14_vel_alt)**2

# ==========================================================================
"""         CALCULO NUMERICO, EM FREQUENCIA, PARA TRANSMISSAO           """
# ==========================================================================

# transmition = np.zeros((passo, len(v_atomos)), dtype=complex)

# for n in range(passo):
    # for q in range(len(v_atomos)):
#     sol = solve_ivp(F, t_int, y0 = cond_i, t_eval=np.linspace(t_int[0], t_int[1], int(10 * N) ), args=( dcw[n], dfs, p11_0, p33_0, v_atomos[4]) )
#     transmition[n][q] = np.imag( sol.y[7,-1] )

# ==========================================================================
"""                 PLOTS (POPULACAO X TEMPO)                           """
# ==========================================================================

plt.plot(sol.t, np.real(sol.y[0,:]), label='p11 - Numerico' )
plt.plot(sol.t, np.real(sol.y[1,:]), label='p22 - Numerico' )
plt.plot(sol.t, np.real(sol.y[2,:]), label='p33 - Numerico' )
plt.plot(sol.t, np.real(sol.y[3,:]), label='p44 - Numerico' )

# plt.axhline(y= np.real(p11_e), xmin = min(sol.t), xmax = max(sol.t), label='p11 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p22_e), xmin = min(sol.t), xmax = max(sol.t), label='p22 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p33_e), xmin = min(sol.t), xmax = max(sol.t), label='p33 - Analitico', c = 'k', ls = '--')
# plt.axhline(y= np.real(p44_e), xmin = min(sol.t), xmax = max(sol.t), label='p44 - Analitico', c = 'k', ls = '--')

# plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$ \n $\\Pi = $" + str(r) + "$\\Gamma$, $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$ \n $\\delta_{cw} = $" + str(dcw[int(len(dcw)/2 - 0.5)]) + ", $\\delta_{fs} = $" + str(0.0) + ", v(m/s) = " + str(v_fx) )
# plt.xlabel('tempo (u.a)')
# plt.ylabel('Populacao')

# ==========================================================================
'''                 PLOTS (COERENCIA X COERENCIA 14)                    '''
# ==========================================================================

# plt.plot(dcw, signal[:, 0], label='FWM - Numerico - v(m/s) = ' + str(v_atomos[0]), lw = 1.0)
# plt.plot(dcw, signal[:, int( v_atomos[0] + v_atomos[-1] ) ], label='FWM - Numerico - v(m/s) = ' + str( v_atomos[0] + v_atomos[-1] ), lw = 1.0)
# plt.plot(dcw, signal[:, -1], label='FWM - Numerico - v(m/s) = ' + str(v_atomos[-1]), lw = 1.0)

# plt.plot(dcw, fwm_vel_alt[:,0], c = 'r', label = 'FWM - Analitico - v(m/s) = ' + str(v_atomos[0]), ls = '--', lw = 1.0)
# # plt.plot(dcw, fwm_vel_alt[:, int(len(v_atomos)/2 - 0.5) ], c = 'g', label = 'FWM - Analitico - v(m/s) = ' + str(v_atomos[0] + v_atomos[-1]), ls = '--', lw = 1.0)
# plt.plot(dcw, fwm_vel_alt[:,-1], c = 'k', label = 'FWM - Analitico - v(m/s) = ' + str(v_atomos[-1]), ls = '--', lw = 1.0)
# plt.xlabel(' $ \\delta_{cw} $ ( $ \\Gamma $) ')
# plt.ylabel('$ \\left| \\sigma_{14} \\right|^2 $')
# plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{F}^{cw}$ = " + str(a) + "$\\Gamma$, $\\Omega_{f}^{cw}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$ \n $\\Pi = $" + str(r) + "$\\Gamma$, $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$ \n $\\delta_{fs} = $" + str(0.0) )

# plt.plot(v_atomos, fwm_vel_alt[int(len(v_atomos)/2 - 0.5)][:], c = 'k', label = 'FWM - Analitico', ls = '--', lw = 2.0)
# plt.plot(v_atomos, fwm_vel_alt[-1][:], c = 'r', label = 'FWM - Analitico', ls = '--', lw = 2.0)
# plt.plot(v_atomos, fwm_vel_alt[0][:], c = 'g', label = 'FWM - Analitico', ls = '--', lw = 2.0)
# plt.xlabel(' $ v $ ( m/s ) ')

# plt.ylabel('$ \\left| \\sigma_{14} \\right|^2 $')
# plt.title(" $\\Omega_{s}$ = " + str(s) + "$\\Gamma$, $\\Omega_{F}^{cw}$ = " + str(a) + "$\\Gamma$, $\\Omega_{f}^{cw}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$ \n $\\Pi = $" + str(r) + "$\\Gamma$, $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$ \n $\\delta_{fs} = $" + str(0.0) )

# ==========================================================================
"""                 PLOT TRANSMISSAO (SOLUCAO NUMERICA)                 """
# ==========================================================================

# plt.plot(delta, transmition, label='Transmissao - Numerico')
# plt.title("$\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+str(b)+ "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$, \n $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$, $\\delta_{fs} = $" + str(dfs) )
# plt.xlabel(' $ \\delta_{cw} $ ')
# plt.ylabel('Im($ \\sigma_{23} $)')

# ==========================================================================
'''                             FIM                                      '''
# ==========================================================================

plt.legend(loc='best' )
plt.show()