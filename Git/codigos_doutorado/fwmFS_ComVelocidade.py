import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.constants import gas_constant
import streamlit as st

fig, ax = plt.subplots()
st.set_page_config(layout='wide')
col1, col2 = st.columns(2)

G = 1e-0        # ORDEM DE TX DECAIMENTO GAMMA
a = 9e-1        # ORDEM DE GAMMA_A (CONTROLE)
b = 5e-2        # ORDEM DE GAMMA_B (PROVA)
c = 0e-1        # ORDEM DE GAMMA_C (FEMTO)

m = 5e-1        # ORDEM DE TX DECAIMENTO COERENCIAS
k = 0e-8        # ORDEM DE TX DECAIMENTO g13, g24
r = 1e-3        # ORDEM DO VALOR DE Pi

N = 500         # TAMANHO DO VETOR TEMPO
temp = 327.15   

M = (0.278 * 86.91 + 0.722 * 84.91) / 1e3                    # MASSAS 85Rb: 84.91 u, 87Rb: 86.91 u
v_fx = 600
inicial = -100.                                              # DETUNING INICIAL
final = 100.                                                 # DETUNING FINAL
passo = 12001                                                # TAMANHO DO VETOR DETUNING
dcw = np.linspace(inicial, final, passo, endpoint=True)      # VETOR DETUNING CW
dfs = 0.0                                                    # VETOR DETUNING FS

v_atomos = np.linspace(-v_fx, v_fx, passo , endpoint=True)   # VETOR VELOCIDADE DOS ATOMOS
v = v_atomos[ int( len( v_atomos )/2 - 1/2 ) ]
u = np.sqrt( temp * gas_constant / M )                                                   # VELOCIDADE MAIS PROVAVEL DO GAS DE 85,87 RB
F_doppler = 1. / np.sqrt( 2 * np.pi * u**2 ) * np.exp( - v_atomos**2 / ( 2 * u**2 ) )

k12 = k23 = 1.                      # VETOR DE ONDA DO CAMPO CW
k34 = 0.981 * k12                   # VETOR DE ONDA DO CAMPO FS
g12 = g23 = g14 = g34 = m * G
g13 = g24 = k * G
Omega_a = a * G
Omega_b = b * G
Omega_c = Omega_s = c * G
Pi = r * G
p11_0 = 0.5
p33_0 = 0.5

def F(t, V, d_cw, d_fs, p11_, p33_, vel ):

    f11 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_a ) * np.conj( V[4]) - Omega_a * V[4] + np.conj( Omega_s ) * np.conj( V[6] ) - Omega_s * V[6] ) - Pi * ( V[0] - p11_ )
    f22 = - ( G + Pi ) * V[1] + 1j * ( Omega_a * V[4] - np.conj( Omega_a ) * np.conj( V[4]) + Omega_b * np.conj( V[7] ) - np.conj( Omega_b ) * V[7] ) 
    f33 = G/2 * V[1] + G/2 * V[3] + 1j * ( np.conj( Omega_b ) * V[7] - Omega_b * np.conj( V[7] ) + np.conj( Omega_c ) * np.conj( V[9] ) - Omega_c * V[9] )  - Pi * ( V[2] - p33_ )
    f44 = - (  G + Pi ) * V[3] + 1j * ( Omega_s * V[6] - np.conj( Omega_s ) * np.conj( V[6] ) + Omega_c * V[9] - np.conj( Omega_c ) * np.conj( V[9] ) )
    f12 = ( 1j * (d_cw - k12 * vel ) - g12 - Pi ) * V[4] + 1j * ( np.conj( Omega_a ) * ( V[1] - V[0] ) - np.conj( Omega_b ) * V[5] + np.conj( Omega_s ) * np.conj( V[8] ) )
    f13 = ( 1j * (  k12 - k23 ) * vel - g13 - Pi ) * V[5] + 1j * ( np.conj( Omega_a ) * V[7] + np.conj( Omega_s ) * np.conj( V[9] ) - Omega_b * V[4] - Omega_c * V[6] )
    f14 = ( 1j * ( d_fs - ( k12 - k23 + k34 ) * vel ) - g14 - Pi ) * V[6] + 1j * ( np.conj( Omega_a ) * V[8] + np.conj( Omega_s ) * ( V[3] - V[0] ) - np.conj( Omega_c ) * V[5] )
    f23 = (-1j * ( d_cw - k23 * vel ) - g23 - Pi ) * V[7] + 1j * ( Omega_a * V[5] - Omega_b * ( V[1] - V[2] ) - Omega_c * V[8])
    f24 = ( 1j * ( (d_fs - k34 * vel ) - (d_cw - k23 * vel ) ) - g24 - Pi ) * V[8] + 1j * ( Omega_a * V[6] + Omega_b * V[9] - np.conj( Omega_s ) * np.conj( V[4] ) - np.conj( Omega_c ) * V[7] )
    f34 = ( 1j * ( d_fs - k34 * vel ) - g34 - Pi ) * V[9] + 1j * ( np.conj( Omega_b ) * V[8] + np.conj( Omega_c ) * ( V[3] - V[2] ) - np.conj( Omega_s ) * np.conj( V[5] ))
    
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

sol = solve_ivp( F, t_int, y0 = cond_i, t_eval = np.linspace( t_int[0], t_int[1], int(10 * N ) ), args=( dcw[ int(len(dcw)/2 - 0.5) ], dfs, p11_0, p33_0, v ) )

fig1 = go.Figure(sol, x = sol.t, y = np.real(sol.y[0,:]))

# ax.plot(sol.t, np.real(sol.y[0,:]), label='p11 - Numerico' )
# ax.plot(sol.t, np.real(sol.y[1,:]), label='p22 - Numerico' )
# ax.plot(sol.t, np.real(sol.y[2,:]), label='p33 - Numerico' )
# ax.plot(sol.t, np.real(sol.y[3,:]), label='p44 - Numerico' )

# ax.axhline(y= np.real(p11_e), xmin = min(sol.t), xmax = max(sol.t), label='p11 - Analitico', c = 'k', ls = '--')
# ax.axhline(y= np.real(p22_e), xmin = min(sol.t), xmax = max(sol.t), label='p22 - Analitico', c = 'k', ls = '--')
# ax.axhline(y= np.real(p33_e), xmin = min(sol.t), xmax = max(sol.t), label='p33 - Analitico', c = 'k', ls = '--')
# ax.axhline(y= np.real(p44_e), xmin = min(sol.t), xmax = max(sol.t), label='p44 - Analitico', c = 'k', ls = '--')

plt.title(" $\\Omega_{s}$ = " + str(c) + "$\\Gamma$, $\\Omega_{c}$ = " + str(a) + "$\\Gamma$, $\\Omega_{p}$ = "+ str(b) + "$\\Gamma$, $\\Omega_{fs} = $" + str(c) + "$\\Gamma$ \n $\\Pi = $" + str(r) + "$\\Gamma$, $\\gamma_{13} = \\gamma_{24} = $" + str(k) + "$\\Gamma$ \n $\\delta_{cw} = $" + str(dcw[int(len(dcw)/2 - 0.5)]) + ", $\\delta_{fs} = $" + str(0.0) + ", v(m/s) = " + str( round(v,2) ) )
plt.xlabel('tempo (u.a)')
plt.ylabel('Populacao')
plt.legend(loc='best')

# print('SOMA POPULACOES CALCULO NUMERICO: ' + str( np.real(sol.y[0,-1]) + np.real(sol.y[1,-1]) + np.real(sol.y[2,-1]) + np.real(sol.y[3,-1]) ) + '\n' )

# ==========================================================================
#       CALCULO TEMPORAL DE POPULACOES COM TEMPO DE VOO (SOLUCAO ANALITICA)
# ==========================================================================

# ==========================================================================
#                  CALCULO NUMERICO, EM FREQUENCIA, PARA COERENCIA 14           
# ==========================================================================

# ==========================================================================
#           CALCULO NUMERICO DAS EQUAÇÕES NOS ESTADOS ESTACIONÁRIOS         
# ==========================================================================

# ==========================================================================
#                SOLUCAO ANALITICA, EM FREQUENCIA, DA COERENCIA
# ==========================================================================

# ==========================================================================
#                CALCULO NUMERICO, EM FREQUENCIA, PARA TRANSMISSAO
# ==========================================================================

# ==========================================================================
#                                     DASHBOARD
# ==========================================================================

with st.sidebar:
    st.slider('Velocidade', v_atomos[0], v_atomos[-1] )

col1.pyplot(fig)