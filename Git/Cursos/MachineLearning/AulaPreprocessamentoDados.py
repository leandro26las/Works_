# import zipfile
# path = r'C:\Users\leand\OneDrive\Área de Trabalho\Leandro\Cursos\MachineLearning\credit_risk_dataset.csv.zip'
# with zipfile.ZipFile( path, 'r' ) as zip:
#     zip.extractall()

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

path = r'F:\Git\Git\Cursos\MachineLearning\credit_risk_dataset.csv'

st.set_page_config(layout='wide')
st.title('Estudo sobre Machine Learning e Visualização de Dados')
st.header('Aula Pré-Processamento e Plots')

base_credit = pd.read_csv(path, sep=',')
nova_base = base_credit[ base_credit['person_age'] < 100 ]

nova_base['loan_status'] = nova_base['loan_status'].apply( {1: "Ativo", 0: "Inativo"}.get )

status_emprestimo = np.unique( nova_base['loan_status'] )
tipo_propriedade = np.unique( nova_base['person_home_ownership'] )
pretencao_emprestimo = np.unique( nova_base['loan_intent'] )
grade_pessoal_emprestimo = np.unique( nova_base['loan_grade'] )

media_A = nova_base['loan_int_rate'][(nova_base['loan_grade']=="A") & (nova_base['loan_int_rate'].notna())].mean()
media_B = nova_base['loan_int_rate'][(nova_base['loan_grade']=="B") & (nova_base['loan_int_rate'].notna())].mean()
media_C = nova_base['loan_int_rate'][(nova_base['loan_grade']=="C") & (nova_base['loan_int_rate'].notna())].mean()
media_D = nova_base['loan_int_rate'][(nova_base['loan_grade']=="D") & (nova_base['loan_int_rate'].notna())].mean()
media_E = nova_base['loan_int_rate'][(nova_base['loan_grade']=="E") & (nova_base['loan_int_rate'].notna())].mean()
media_F = nova_base['loan_int_rate'][(nova_base['loan_grade']=="F") & (nova_base['loan_int_rate'].notna())].mean()
media_G = nova_base['loan_int_rate'][(nova_base['loan_grade']=="G") & (nova_base['loan_int_rate'].notna())].mean()

nova_base['loan_int_rate'][(nova_base['loan_grade']=="G") & (nova_base['loan_int_rate'].isna())] = media_G
nova_base['loan_int_rate'][(nova_base['loan_grade']=="F") & (nova_base['loan_int_rate'].isna())] = media_F
nova_base['loan_int_rate'][(nova_base['loan_grade']=="E") & (nova_base['loan_int_rate'].isna())] = media_E
nova_base['loan_int_rate'][(nova_base['loan_grade']=="D") & (nova_base['loan_int_rate'].isna())] = media_D
nova_base['loan_int_rate'][(nova_base['loan_grade']=="C") & (nova_base['loan_int_rate'].isna())] = media_C
nova_base['loan_int_rate'][(nova_base['loan_grade']=="B") & (nova_base['loan_int_rate'].isna())] = media_B
nova_base['loan_int_rate'][(nova_base['loan_grade']=="A") & (nova_base['loan_int_rate'].isna())] = media_A

percentual_emprestimo_salario = nova_base['loan_percent_income']
valor_emprestimo = nova_base['loan_amnt']
outros_emprestimos = nova_base['cb_person_cred_hist_length']
possui_historico_divida = nova_base['cb_person_default_on_file']
idade_pessoas = nova_base['person_age'].sort_values()
taxa_juros = nova_base['loan_int_rate']

# X_credit = nova_base.iloc[:, [1, 3, 6, 7, 9, 11]]
# Y_credit = nova_base.iloc[:, [0, 2, 4, 5, 8, 10]]

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

fig_taxa_emprestimo = px.scatter(nova_base, x = 'person_age', y = 'loan_int_rate',
                             color = 'loan_grade', size= 'loan_int_rate',
                             title="Taxa de Juros X Idade")
col1.plotly_chart(fig_taxa_emprestimo, use_container_width=True)

fig_intecao = px.pie(nova_base, values='loan_amnt', names='loan_intent', title= r'% Intenção do Emprestimo')
col2.plotly_chart(fig_intecao, use_container_width=True)

fig_loan_grade = px.bar(nova_base, x='loan_grade', y='loan_amnt', color='loan_intent', 
                        labels={"loan_intent": "Loan Intent", "loan_amnt": "Loan Amount", "loan_grade": "Loan Grade"},
                         category_orders = {"loan_grade": ["A", "B", "C", "D", "E", "F", "G"]} )
col3.plotly_chart(fig_loan_grade, use_container_width=True)

fig_taxa_intencao = px.scatter(nova_base, x = 'person_age', y = 'loan_int_rate',
                             color = 'loan_intent', size= 'loan_int_rate',
                             title="Taxa de Juros X Idade")
col4.plotly_chart(fig_taxa_intencao, use_container_width=True)