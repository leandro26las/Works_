import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib.ticker import AutoMinorLocator

diretorio = 'plots'
parent_dir = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Doutorado\Medidas\espectros\2024medidas_02_07'
path = os.path.join(parent_dir, diretorio)
files = os.listdir(parent_dir)
# try :
#     os.mkdir(path)
# except OSError as error:
#     print(error)

################################ PRINCIPAL ##########################################

i = 9
arq1 = rf'{parent_dir}\{files[i]}'
print("\n \t Obtendo Arquivo 1\n  ", arq1)
dt1 = pd.read_csv(arq1, header = None, delimiter="\t", quoting=0, decimal=',', encoding='utf-8')
print("\n \t Lendo Arquivo 1 \n ")

comp_onda = dt1[0]
amp = dt1[1]

print("\n \t Gerando Gráfico \n ")

plt.plot(comp_onda, amp, c = 'red', lw = 2)
plt.xlim(740, 860)
plt.title('Espectro Femto Forte Depois da Célula')
plt.xlabel('$\lambda$ (nm)')
plt.ylabel('Amplitude (u.a.)')
plt.show()