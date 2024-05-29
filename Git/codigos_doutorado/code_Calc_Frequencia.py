# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:07:59 2023

@author: LabDF
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

path = r'C:/Users/leand/OneDrive/Área de Trabalho/Leandro/Doutorado/Medidas/medidas_28julho2023/medida9.csv'

dado = pd.read_csv(path, header=20)

tempo1_85Rb = -0.169
F34_85Rb = 384229241690063.7        #Hz

tempo2_87Rb = 0.374
C13_87Rb = 384227903404128.94       #Hz

tempo, intensidade = dado["TIME"] - dado["TIME"].min() , dado["CH2"]

# freq = ( tempo1_85Rb - tempo) * ( ( C13_87Rb - F34_85Rb ) / ( tempo2_87Rb - tempo1_85Rb) ) + F34_85Rb

plt.plot(tempo, intensidade); plt.xlabel("Tempo"); plt.grid(axis='both')
#plt.plot(freq, intensidade); plt.xlabel("Frequencia (Hz)");plt.grid(axis='both')

# abs_saturada = pd.DataFrame( {'frequencia': freq, 
#                               'abs_linear': dado["CH1"],
#                               'abs_saturada': dado["CH2"]} )

# abs_saturada.to_csv('medidafreq9.csv', index = False)

plt.show()
