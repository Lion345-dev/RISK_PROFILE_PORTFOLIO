import os
import pandas as pd
from yahooquery import Ticker  # Nueva librería
import time

# Lista de tickers a descargar
tickers = [
    "ACWI", "SHY", "IVV", "EZU", "EPP", "EEM", "BKF", "ILF", "SPY", "EWC",
    "EWZ", "EWG", "EWQ", "EWU", "FXI", "INDI", "EWH", "EWJ", "EWT", "EWY",
    "EWA", "DIA", "IWM", "ITB", "IYH", "IYF", "XLF", "IEO", "QQQ", "IAU",
    "SLV", "DBO", "AGG", "BOND", "EWW", "TIP", "LQD", "EMLC", "VGSH", "NEAR", "BIL"
]

# Carpeta donde se guardarán los archivos individuales
output_folder = "precios_close"
os.makedirs(output_folder, exist_ok=True)

# Paso 1: Descargar y guardar archivos individuales
for ticker in tickers:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"⏳ Descargando {ticker} (Intento {attempt + 1}/{max_retries})...")
            ticker_obj = Ticker(ticker, asynchronous=True)
            data = ticker_obj.history(period="10y", interval="1d")

            if data is not None and not data.empty and 'adjclose' in data.columns:
                # Usar 'adjclose' como proxy para 'Close' y crear un DataFrame con la fecha como índice
                close_data = data[['adjclose']].rename(columns={'adjclose': 'Close'})
                close_data.index.name = 'Date'
                output_path = os.path.join(output_folder, f"{ticker}_close.csv")

                if os.path.exists(output_path):
                    os.remove(output_path)

                close_data.to_csv(output_path)
                print(f"✅ {ticker} guardado correctamente.\n")
                break
            else:
                print(f"⚠️ No se encontraron datos válidos para {ticker} con 'adjclose'. Saltando.\n")
                break

        except Exception as e:
            print(f"❌ Error al descargar {ticker} (Intento {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print(f"⚠️ Fallo tras {max_retries} intentos para {ticker}. Saltando.\n")
            else:
                time.sleep(2 ** attempt)  # Espera exponencial entre reintentos

# Paso 2: Unir todos los archivos en una sola tabla
tabla_unida = pd.DataFrame()

for archivo in os.listdir(output_folder):
    if archivo.endswith("_close.csv"):
        ticker = archivo.split("_")[0]
        ruta = os.path.join(output_folder, archivo)

        try:
            # Leer el archivo
            df = pd.read_csv(ruta)

            # Asegurarse que la primera columna se llama 'Date'
            if 'Date' not in df.columns:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

            # Convertir la columna 'Date' a datetime y establecerla como índice
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index("Date", inplace=True)
            df.rename(columns={"Close": ticker}, inplace=True)

            if tabla_unida.empty:
                tabla_unida = df
            else:
                tabla_unida = tabla_unida.join(df, how='outer')

        except Exception as e:
            print(f"❌ Error al procesar {archivo}: {e}\n")

# Ordenar por fecha y guardar la tabla final
if not tabla_unida.empty:
    tabla_unida.sort_index(inplace=True)
    tabla_unida.to_csv("precios_close_unificados.csv")
    print("✅ Todos los precios han sido unidos en 'precios_close_unificados.csv'")
else:
    print("⚠️ No se pudo crear la tabla unificada porque no hay datos válidos.")