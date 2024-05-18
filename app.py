import streamlit as st
import pandas as pd
import joblib
from joblib import load
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta


st.write('# Previsão do Preço do Petróleo ')
st.write('### Conforme maior o espaço entre as datas, maior a chance de erro.')
input_data_ini = st.date_input(label='Data Inicial', min_value=datetime.today() + timedelta(1), format='DD/MM/YYYY')
input_data_fim = st.date_input(label='Data Final', min_value=datetime.today() + timedelta(1), format='DD/MM/YYYY')



if st.button('Enviar'):
    dados = pd.read_csv('https://raw.githubusercontent.com/willcastr/analise_ipeadata/main/Dados%20Ipea.csv', sep=";")
    model = joblib.load('modelo_prophet.joblib')
    future_dates = pd.DataFrame({
        'ds': pd.date_range(start=input_data_ini, end=input_data_fim)
    })

    def format_data(val):
        return datetime.strftime(val, '%d/%m/%Y')

    # Função para formatar a coluna 'Quantidade'
    def format_valor_dol(val):
        return f'US$ {val} unidades'

    

    final_pred = model.predict(future_dates)
    final_pred = final_pred.rename(columns={'ds': 'Data', 'yhat': 'Previsão (US$)' ,'yhat_upper': 'Valor Máximo (US$)', 'yhat_lower':'Valor Minímo (US$)' })
    
    final_pred['Data'] = final_pred['Data'].apply(format_data)
    # final_pred.style.format({
    #     'Data': format_data,
    #     'Previsão': format_valor_dol,
    #     'Minímo': format_valor_dol,
    #     'Valor': format_valor_dol,

    # })
    st.write(final_pred[['Data', 'Previsão (US$)', 'Valor Máximo (US$)', 'Valor Minímo (US$)']])
