# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:26:14 2023
@author: leand
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

diretorio = 'plots'
parent_dir = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Doutorado\Medidas\FWM\medidas_18novembro2023\arquivos'
path = os.path.join(parent_dir, diretorio)

try :
    os.mkdir(path)
except OSError as error:
    print(error)

for i in range(12):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    arq = rf'{parent_dir}\scope_{i}.csv'
    dt1 = pd.read_csv(arq, header = 1, delimiter=",", quoting=1, encoding='utf-8')
    
    time = dt1['second']
    abs_linear_femto = ( dt1['Volt'] - dt1['Volt'].min() ) / ( dt1['Volt'].max() - dt1['Volt'].min() )
    fwm_femto = ( dt1['Volt.1'] - dt1['Volt.1'].min() ) / ( dt1['Volt.1'].max() - dt1['Volt.1'].min() )
    
    ax.plot( time, abs_linear_femto, lw = 3, c= 'green', label='Transmissao' )
    ax.plot( time, fwm_femto, c='red', label='FWM' )
    plt.legend(loc = 'best', fontsize="7")
    plt.xlabel( 'Tempo (s)' )
    plt.title( 'FWM COM FEMTO, f$_{rep}$ = 842.79 MHz' )
    plt.savefig(path + rf'\figure_{i}.png')

# fig, ax = plt.subplots(2,1, layout="constrained")
# ax[0].plot(time, abs_linear_femto, label = 'Transmi c/ fs', c='green')
# ax[0].legend(loc = 'best', fontsize="7")
# ax[1].plot(time, fwm_femto, label = 'FWM do femto ')
# ax[1].legend(loc = 'best', fontsize="12")
# ax[1].set_xlabel( ' Tempo (s) ' )
# fig.suptitle('FWM COM FEMTO, f$_{rep}$ = 842.79 MHz')
# plt.savefig(path + rf'\figure_{i}.png')