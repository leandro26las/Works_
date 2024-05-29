import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib.ticker import AutoMinorLocator

parent_dir = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Doutorado\Medidas\Transmissao\2024medidas_02_07'
files = os.listdir(parent_dir)

####################### TRANSIÇÕES DO RUBÍDIO 87 F = 2 ##############################

F_32 = 384230484.4685 - 2563.00597908911 + 193.7408

C_32 = F_32 - 266.650/2

C31 = F_32 - (266.650 + 156.947)/2

F22 = F_32 - 266.650

C21 = F22 - 156.947/2

F21 = F22 - 156.947

Diff_Rb87 = [F_32 - F_32, F_32 - C_32, F_32 - C31, F_32 - F22, F_32 - C21, F_32 - F21]

####################### TRANSIÇÕES DO RUBÍDIO 85 F = 3 ##############################

F43 = 384230406.373-1264.8885163 + 100.205

C43 = F43 - 120.640/2

C42 = F43 - (120.640 + 63.401)/2

F33 = F43 - 120.640

C32 = F33 - 63.401/2

F32 = F33 - 63.401

Diff_Rb85 = [F43 - F43, F43 - C43, F43 - C42, F43 - F33, F43 - C32]

#####################################################################################

# ENCONTRA A AMPLITUDE MÁXIMA DOS PICOS
def FindPeaksAmp(var):
    peaks, _ = find_peaks(var, height = (0.0, 0.60), prominence=(0.02, 0.5) )
    a_lista = [n for n in var[peaks]]
    return a_lista

# ENCONTRA A ESCALA TEMPORAL ONDE OCORREM OS PICOS
def FindPeaksTime(var1, var2):
    peaks, _ = find_peaks(var2, height = (0.0, 0.60), prominence=(0.02, 0.5) )
    t_lista = [m for m in var1[peaks]]
    return t_lista

# TRANSFORMA A ESCALA DO TEMPO EM FREQUÊNCIA
def FreqScale(t, var):
    t1 = FindPeaksTime(t, var)[0]
    t2 = FindPeaksTime(t, var)[-1]
    f1 = F_32
    f2 = C21
    A = ( f2 - f1 ) / ( t2 - t1 )
    f = f2 - A * ( t2 - t )
    return f - f.min()

# CALCULA O ERRO RELATIVO AO VALOR REAL DO VALOR MEDIDO
def Error(var1, var2):
    PeaksF = FindPeaksTime(var1, var2)
    Diff_Meas = PeaksF[0] - PeaksF[3]
    Diff_Real = Diff_Rb85[-2]
    error_f = abs(( Diff_Meas - Diff_Real )) / Diff_Real * 100.0
    return format(error_f, '0.2f')

################################ PRINCIPAL ##########################################

i = 0
arq1 = rf'{parent_dir}\{files[i]}'
print("\n \t Obtendo Arquivo 1\n  ", arq1)
dt1 = pd.read_csv(arq1, header = None, delimiter="\t", quoting=0, decimal=',', encoding='utf-8')
print("\n \t Lendo Arquivo 1 \n ")

# arq2 = rf'{parent_dir}\{files[i-1]}'
# print("\n \t Obtendo Arquivo 2 \n \n ", arq1)
# dt2 = pd.read_csv(arq2, header = 1, delimiter=",", quoting=1, encoding='utf-8')
# print("\n\t Lendo Arquivo 2 \n\n ")

# frep = str( float( dt1['frep'][0].replace(' Hz', '') ) + 840e6 )
# print("\n \t", frep, "\n \n ")

# time = dt1['second']
# abs_saturada = dt1['Volt'].rolling(50, center = True).mean()
# lockin = dt1['Volt.1'] #- dt2['Volt.1']

# print("\n \t Separando Colunas \n ")

# print("\n \t Preparando para Reescalar Sinais \n ")
# a = (abs_saturada - abs_saturada.min())/(abs_saturada.max() - abs_saturada.min())
# l = (lockin - lockin.min())/(lockin.max() - lockin.min())

# print("\n \t Obtendo Picos de Absorção Saturada \n ")
# peaks, _ = find_peaks(a, height=(0.0, 0.8), prominence=(0.02, 0.5) )
# plt.scatter(time[peaks], a[peaks], c = 'red', marker = 'x')

# print("\n \t Gerando Gráfico \n ")

# plt.plot(time, a, label = 'Abs Saturada', c = 'blue', lw = 2)
# plt.title('Transmissao Rb87 (F = 2) \n f$_{rep}$ = 844.04 MHz')
# plt.xlabel('Tempo (s)')
# plt.ylabel('Amplitude (u.a.)')
# plt.legend(loc='best')
# plt.show()