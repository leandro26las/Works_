# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 15:26:20 2023

@author: leand
"""

import pandas as pd
import matplotlib.pyplot as plt

save_figure = 'femto_pump_200.png'

arq1 = pd.read_csv("C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_28julho2023/femto200.csv", sep=';', decimal=',', header=18)
#arq2 = pd.read_csv("C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_28julho2023/frep_"+str(i)+".csv", header=17)

dados1 = pd.DataFrame( { "time" : arq1['Time (s)'] , "Amp1" : arq1['1 (VOLT)'], "Amp2" : arq1['2 (VOLT)'] } )
t, x, y = dados1['time'], dados1["Amp1"], dados1["Amp2"]
dt = t - t.min()

# t1 = 0.00680                        # tempo F34_85Rb
# f1 = 384229241690063.7              # frequencia (Hz) F34_85Rb

# t2 = 0.01957                        # tempo C13_87Rb
# f2 = 384227903404128.94             # frequencia (Hz) C13_87Rb

# f = f2 - (t2 - dt )*( ( f2 - f1 ) / ( t2 - t1 ) )
# df = f - f.min()

# Caracterization Femtosecond
# dados2 = pd.DataFrame( { "freq" : arq2['Trace1 X'] , "Amp" : arq2['Trace1 Y'] } )
# freq, amp = dados2['freq'], dados2["Amp"]
# frep, amp = dados2.iloc[ dados2["Amp"].idxmax(axis=0, skipna=True) ]


plt.plot(dt,x)
plt.plot(dt,y)
plt.xlabel( 'Time (s)' )
plt.ylabel( 'Amplitude (a.u.)' )
plt.title('EIT, Pump Femto P = 200 mW, Diodo = 4.5 mW')
plt.show()
plt.savefig(save_figure)

# plt.plot(dt, y)
# plt.show()