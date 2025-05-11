import os
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY")
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")

def get_etf_sector_data(ticker):
    """Obtiene datos sectoriales del ETF desde Finviz"""
    url = f"https://elite.finviz.com/export.ashx?v=111&f=sym_{ticker}&auth={FINVIZ_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(response.text)
        return df
    else:
        raise Exception(f"Error al obtener datos de Finviz para {ticker}")

def get_historical_data(tickers, period="10y", interval="1mo"):
    """Obtiene datos históricos de Yahoo Finance"""
    data = yf.download(tickers, period=period, interval=interval)
    return data['Adj Close']

def calculate_returns(prices):
    """Calcula retornos mensuales"""
    returns = prices.pct_change().dropna()
    return returns

def get_risk_free_rate():
    """Obtiene la tasa libre de riesgo de Cetes 28 días desde Banxico"""
    url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF61745/datos/oportuno"
    headers = {"Bmx-Token": BANXICO_TOKEN}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        rate = float(data['bmx']['series'][0]['datos'][0]['dato']) / 100
        return rate
    else:
        raise Exception("Error al obtener la tasa libre de riesgo de Banxico")