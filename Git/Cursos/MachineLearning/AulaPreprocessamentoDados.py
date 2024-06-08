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

st.set_page_config(layout='wide')
st.title('Estudo sobre Machine Learning')
st.header('Aula Pré-Processamento')

base_credit = pd.read_csv('credit_risk_dataset.csv', sep=',')
base_credit