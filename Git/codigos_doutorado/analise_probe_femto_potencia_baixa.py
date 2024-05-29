# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:24:57 2023

@author: leand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.subplots(2,1, sharex=True)
plt.subplots_adjust(hspace=0)


################### SINAL ABSORSAO SATURADA ################################
AbsSat_arquivo = pd.read_csv("C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_1agosto2023/probe1mW_sat_lckin_c_fs_0.csv", sep=';', decimal=',', header=18)
AbsSat = pd.DataFrame( { "time" : AbsSat_arquivo['Time (s)'], "Amp3" : AbsSat_arquivo['3 (VOLT)'] } )
AbsSaturada = AbsSat["Amp3"] - AbsSat["Amp3"][:100].mean()

################### CONVERSAO TEMPO PARA FREQUENCIA ##################
dt = AbsSat['time'] - AbsSat['time'].min()      #Tempo

t1 = 0.1373                         # tempo F34_85Rb
t2 = 0.3523                         # tempo C13_87Rb
f1 = 384229241690063.7              # frequencia (Hz) F34_85Rb
f2 = 384227903404128.94             # frequencia (Hz) C13_87Rb
f = f2 - (t2 - dt )*( ( f2 - f1 ) / ( t2 - t1 ) )
df = ( f - f.min() )/1e9


plt.subplot(211)
plt.ylabel( 'Amplitude (a.u.)' )
plt.plot(df, AbsSaturada )

################### SINAL DO PROBE ####################################

Probe_arquivo = pd.read_csv("C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_1agosto2023/probe1mW_sat_lckin_c_fs_0.csv", sep=';', decimal=',', header=18)
Probe = pd.DataFrame( { "time" : Probe_arquivo['Time (s)'], "Amp2" : Probe_arquivo['2 (VOLT)'] } )
NewProbe = Probe["Amp2"] - Probe["Amp2"][:100].mean()

# FREQUENCIA DE REPETICAO
arq2 = pd.read_csv("C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_1agosto2023/femto_diodo_probe_1mW.csv", header=17)
dados2 = pd.DataFrame( { "freq" : arq2['Trace1 X'] , "Amp" : arq2['Trace1 Y'] } )
freq, amp = dados2['freq'], dados2["Amp"]
frep, amp = dados2.iloc[ dados2["Amp"].idxmax(axis=0, skipna=True) ]

plt.subplot(212)
plt.plot(df, NewProbe, label = 'f$_{rep}$: ' + str(frep) + ' (Hz)')
plt.suptitle('Interacao Femto em Vapor Rb, Probe = 1 mW' )
plt.xlabel( 'Frequency (GHz)' )
plt.ylabel( 'Amplitude (a.u.)' )
plt.legend(loc='best', bbox_to_anchor=(0.5,1.0))
plt.show()

# plt.savefig('medida_EIT_'+str(i)+'.png')