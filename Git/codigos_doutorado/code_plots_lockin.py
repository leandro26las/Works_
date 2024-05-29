# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:00:03 2023

@author: LabDF
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib.ticker import AutoMinorLocator

diretorio = 'plots'
parent_dir = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Doutorado\Medidas\FWM\FEMTO\2024medidas_02_20'
path = os.path.join(parent_dir, diretorio)

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
    f1 = F43
    f2 = C31
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

################################## PRINCIPAL #######################################################

# try :
#     os.mkdir(path)
# except OSError as error:
#     print(error)

for i in range(1, len(files)):
    arq1 = rf'{parent_dir}\{files[i]}'
    dados = pd.read_csv( arq1, header = 1, delimiter=",", quoting=1, encoding='utf-8' )
    dados['Volt'] = dados['Volt'].rolling(50, center=True).mean()
    time = dados['second']
    AbsSat = ( dados['Volt'] - dados['Volt'].min() ) / ( dados['Volt'].max() - dados['Volt'].min() )
    fwm = ( dados['Volt.1'] - dados['Volt.1'].min() ) / ( dados['Volt.1'].max() - dados['Volt.1'].min() )
    frequency = FreqScale(time, AbsSat)
    # frep = str( float(dados['frep'][0].replace(' Hz', '') ) + 840e6 )
    
    ################################# PLOTS ##########################################################
    print('plot ' + str(i))
    
    fig, ax = plt.subplots()
    ax.plot(frequency, AbsSat, c = 'red', lw=2)
    ax.plot(frequency, fwm, c = 'blue', lw=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    fig.supxlabel('Frequencia (MHz)')
    fig.supylabel('Amplitude (u.a.)')
    fig.suptitle('Transmissao Rb87 (F = 2) \n f$_{rep}$ = 844.04 MHz' )
    plt.savefig(path + rf'/figura{i}' + '.png')