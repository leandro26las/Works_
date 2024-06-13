# import zipfile
# path = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Cursos\MachineLearning\credit_risk_dataset.csv.zip'
# with zipfile.ZipFile( path, 'r' ) as zip:
#     zip.extractall()


# import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

path = r'F:\Git\Git\Cursos\MachineLearning\credit_risk_dataset.csv'

# st.set_page_config(layout='wide')
# st.title('Estudo sobre Machine Learning')
# st.header('Aula Pré-Processamento')

base_credit = pd.read_csv(path, sep=',')
base_credit.columns
nova_base = base_credit[base_credit['person_age'] < 100]

idade_pessoas = nova_base['person_age']
status_emprestimo = np.unique(nova_base['loan_status'], return_counts=True);
tipo_propriedade = np.unique(nova_base['person_home_ownership'], return_counts=True);
pretencao_emprestimo = np.unique(nova_base['loan_intent'], return_counts=True);
grade_pessoal_emprestimo = np.unique(nova_base['loan_grade'], return_counts=True);
percentual_emprestimo_salario = nova_base['loan_percent_income']
valor_emprestimo = nova_base['loan_amnt']
outros_emprestimos = nova_base['cb_person_cred_hist_length']
possui_historico_divida = nova_base['cb_person_default_on_file']
X_credit = nova_base.iloc[:, [1, 3, 6, 7, 9, 11]]
Y_credit = nova_base.iloc[:, [0, 2, 4, 5, 8, 10]]

fig0 = plt.hist(nova_base['cb_person_cred_hist_length'], bins=15);
fig1 = plt.hist(nova_base['person_age'], bins=15);
fig2 = plt.hist(nova_base['person_income'], bins=100);
fig3 = plt.scatter(nova_base['person_age'], nova_base['person_income'])
fig4 = plt.bar(idade_pessoas, outros_emprestimos, width = 5.0, align='center')
plt.show()