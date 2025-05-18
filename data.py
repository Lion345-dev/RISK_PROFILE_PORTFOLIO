import os
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY")
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")

def get_etf_sector_data(ticker):
    """Obtiene datos sectoriales del ETF desde Finviz"""
    try:
        url = f"https://elite.finviz.com/export.ashx?v=111&f=sym_{ticker}&auth={FINVIZ_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(response.text)
            logger.info(f"Datos sectoriales obtenidos para {ticker}")
            return df
        else:
            logger.error(f"Error al obtener datos de Finviz para {ticker}: {response.status_code}")
            raise Exception(f"Error al obtener datos de Finviz para {ticker}")
    except Exception as e:
        logger.error(f"Excepción al obtener datos de Finviz para {ticker}: {str(e)}")
        raise

def get_historical_data(tickers, period="5y", interval="1mo"):
    """Obtiene datos históricos de Yahoo Finance"""
    try:
        data = yf.download(tickers, period=period, interval=interval)
        if 'Close' in data:
            logger.info(f"Datos históricos obtenidos para {tickers}")
            return data['Close']
        else:
            logger.error(f"No se encontraron datos 'Close' para {tickers}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al obtener datos históricos para {tickers}: {str(e)}")
        return pd.DataFrame()

def calculate_returns(prices):
    """Calcula retornos mensuales"""
    if prices.empty or len(prices) < 2:
        logger.error("DataFrame de precios vacío o con menos de 2 filas")
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    if returns.empty:
        logger.error("No se pudieron calcular retornos: todos los valores son NaN")
    return returns

def get_risk_free_rate():
    """Obtiene la tasa libre de riesgo de Cetes 28 días desde Banxico"""
    try:
        url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF61745/datos/oportuno"
        headers = {"Bmx-Token": BANXICO_TOKEN}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            rate = float(data['bmx']['series'][0]['datos'][0]['dato']) / 100
            logger.info(f"Tasa libre de riesgo obtenida: {rate}")
            return rate
        else:
            logger.error(f"Error al obtener tasa de Banxico: {response.status_code}")
            return 0.05  # Respaldo: 5%
    except Exception as e:
        logger.error(f"Excepción al obtener tasa de Banxico: {str(e)}")
        return 0.05  # Respaldo: 5%