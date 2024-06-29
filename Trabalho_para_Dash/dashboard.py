import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# PENSAR CALCULOS IMPORTANTES PARA O DASH
# CONSTRUINDO DASH

path = r'F:\Git\Git\Trabalho_para_Dash\arquivos_favorito'
df1 = pd.read_csv(path + r'\vendas_ecommerce_6meses.csv')

categoria = st.sidebar.selectbox("Categoria", df1['Categoria'].unique() )