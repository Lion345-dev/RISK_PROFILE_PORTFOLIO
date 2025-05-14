import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from portfolio import calculate_risk_profile, optimize_portfolio, sharpe_ratio_maximization
from data import get_risk_free_rate
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")

# Configuración de la página
st.set_page_config(page_title="Perfil de Riesgo y Recomendación de Portafolios", layout="wide")

# Logo y título
st.image("LogoAllianz.jpeg", width=150)
st.title("Simulador de Perfil de Riesgo y Portafolios")

# Mapeo de tickers a nombres de ETFs y ajuste de tickers para yfinance
ticker_to_name = {
    "CETETRCISHRS.MX": "Allianz ETF Conservador Pesos", "NAFTRACISHRS.MX": "Allianz ETF Dinámico Pesos", 
    "UDITRACISHRS.MX": "Allianz ETF Real Pesos", "SHY": "Allianz ETF Conservador Dólares", 
    "IVV": "Allianz ETF Dinámico Dólares", "EUNL.MI": "Allianz ETF Conservador Euros", 
    "EZU": "Allianz ETF Dinámico Euros", "ACWI": "MSCI ACWI INDEX FUND", 
    "EPP": "AZ MSCI ASIA PACIFIC ex JAPAN", "EEM": "MSCI EMERGING MKT IN", 
    "BKF": "BRIC INDEX FUND", "ILF": "S&P LATIN AMERICA 40", "SPY": "S&P500 INDEX FUND",
    "EWC": "MSCI CANADA", "LCTRACISHRS.MX": "IPC LARGE CAP T R TR", "EWZ": "MSCI BRAZIL",
    "EWG": "MSCI GERMANY INDEX", "EWQ": "MSCI FRANCE INDEX FD", "EWU": "MSCI UNITED KINGDOM",
    "FXI": "FTSE/XINHUA CHINA 25", "INDA": "AZ INDIA INDEX FUND", "EWH": "MSCI HONG KONG INDEX",
    "EWJ": "MSCI JAPAN INDEX FD", "EWT": "MSCI TAIWAN INDEX FD", "EWY": "MSCI SOUTH KOREA IND",
    "EWA": "MSCI AUSTRALIA INDEX", "DIA": "SPDR DJIA TRUST", "IWM": "AZ RUSSEL 2000",
    "ITB": "AZ DJ US HOME CONSTRUCT", "IYH": "HEALTH CARE SELECT SECTOR", "IYF": "DJ US FINANCIAL SECT",
    "XLF": "FINANCIAL SELECT SECTOR SPDR", "IEO": "DJ US OIL & GAS EXPL", "QQQ": "AZ QQQ NASDAQ 100",
    "IAU": "GOLD TRUST", "SLV": "SILVER TRUST", "DBO": "AZ OIL TRUST FUND",
    "M10TRACISHRS.MX": "LATIXX MEX M10TRAC", "M5TRACISHRS.MX": "LATIXX MEX M5TRAC", "AGG": "BARCLAYS AGGREGATE"
}

# Mapeo de tickers originales a los ajustados para yfinance
ticker_mapping = {
    "CETETRAC": "CETETRCISHRS.MX", "NAFTRAC": "NAFTRACISHRS.MX", "UDITRAC": "UDITRACISHRS.MX",
    "LCTRAC": "LCTRACISHRS.MX", "M10TRAC": "M10TRACISHRS.MX", "M5TRAC": "M5TRACISHRS.MX",
    "SHY": "SHY", "IVV": "IVV", "IBGS": "EUNL.MI", "EZU": "EZU", "ACWI": "ACWI", "EPP": "EPP",
    "EEM": "EEM", "BKF": "BKF", "ILF": "ILF", "SPY": "SPY", "EWC": "EWC", "EWZ": "EWZ",
    "EWG": "EWG", "EWQ": "EWQ", "EWU": "EWU", "FXI": "FXI", "INDI": "INDA", "EWH": "EWH",
    "EWJ": "EWJ", "EWT": "EWT", "EWY": "EWY", "EWA": "EWA", "DIA": "DIA", "IWM": "IWM",
    "ITB": "ITB", "IYH": "IYH", "IYF": "IYF", "XLF": "XLF", "IEO": "IEO", "QQQ": "QQQ",
    "IAU": "IAU", "SLV": "SLV", "DBO": "DBO", "AGG": "AGG"
}

# Cuestionario de Perfilamiento
st.header("Cuestionario de Perfilamiento")
education = st.selectbox("1. Nivel de Estudios", ["Primaria/Secundaria", "Preparatoria", "Licenciatura", "Posgrado"], index=0)
age = st.selectbox("2. Rango de Edad", ["Menos de 30 años", "30-50 años", "Más de 50 años"], index=0)
investment_horizon = st.selectbox("3. Horizonte de Inversión", ["Menos de 1 año", "1 a 5 años", "Más de 5 años"], index=0)
financial_knowledge = st.selectbox("4. Conocimiento Financiero", ["Ninguno", "Básico", "Intermedio", "Avanzado"], index=0)
risk_tolerance = st.selectbox("5. Tolerancia al Riesgo", ["Conservador", "Moderado", "Agresivo"], index=0)

# Mapear respuestas a puntos
education_points = {"Primaria/Secundaria": 1, "Preparatoria": 2, "Licenciatura": 3, "Posgrado": 4}[education]
age_points = {"Menos de 30 años": 3, "30-50 años": 2, "Más de 50 años": 1}[age]
horizon_points = {"Menos de 1 año": 1, "1 a 5 años": 2, "Más de 5 años": 3}[investment_horizon]
knowledge_points = {"Ninguno": 0, "Básico": 1, "Intermedio": 2, "Avanzado": 3}[financial_knowledge]
tolerance_points = {"Conservador": 1, "Moderado": 2, "Agresivo": 3}[risk_tolerance]

responses = [education_points, age_points, horizon_points, knowledge_points, tolerance_points]
profile, score = calculate_risk_profile(responses)

st.write(f"Tu perfil de riesgo es: {profile} (Puntaje: {score})")

# Selección de moneda
currency = st.multiselect("Selecciona las monedas para tu portafolio", ["MXN", "USD", "EUR"], default=["MXN"])

# Función para cargar datos históricos con reintentos y Alpha Vantage como respaldo
@st.cache_data(ttl=3600)
def load_ticker_data(tickers, period="10y", interval="1mo"):
    adjusted_tickers = [ticker_mapping.get(ticker, ticker) for ticker in tickers]
    data_dict = {}
    max_retries = 3
    retry_delay = 5  # segundos

    for ticker, orig_ticker in zip(adjusted_tickers, tickers):
        # Intentar con Yahoo Finance
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
                if not data.empty and 'Adj Close' in data:
                    data = data['Adj Close'].dropna()
                    if not data.empty:
                        data_dict[orig_ticker] = data
                        break
                time.sleep(retry_delay)
            except Exception as e:
                if "Rate limited" in str(e):
                    st.warning(f"Rate limit alcanzado para {ticker}. Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                else:
                    st.warning(f"Error al cargar datos de Yahoo Finance para {ticker}: {str(e)}")
                    break

        # Si falla Yahoo Finance, intentar con Alpha Vantage para tickers MX
        if orig_ticker not in data_dict and ticker.endswith('.MX'):
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url).json()
                if "Monthly Adjusted Time Series" in response:
                    time_series = response["Monthly Adjusted Time Series"]
                    dates = []
                    prices = []
                    for date, values in time_series.items():
                        dates.append(pd.to_datetime(date))
                        prices.append(float(values["5. adjusted close"]))
                    data = pd.Series(prices, index=dates, name=orig_ticker).sort_index()
                    # Filtrar para el período de 10 años
                    end_date = datetime.now()
                    start_date = end_date - pd.Timedelta(days=10*365)
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
                    if not data.empty:
                        data_dict[orig_ticker] = data
                else:
                    st.warning(f"No se encontraron datos en Alpha Vantage para {ticker}")
            except Exception as e:
                st.warning(f"Error al cargar datos de Alpha Vantage para {ticker}: {str(e)}")

    if not data_dict:
        return pd.DataFrame()

    # Combinar datos en un solo DataFrame
    combined_data = pd.DataFrame(data_dict)
    return combined_data.dropna()

# Función para calcular retornos
def calculate_returns(prices):
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    return returns

# Función para clasificar activos
def classify_asset(ticker):
    adjusted_ticker = ticker_mapping.get(ticker, ticker)
    try:
        url = f"https://elite.finviz.com/export.ashx?v=111&f=sym_{adjusted_ticker}&auth={FINVIZ_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return "Acción"
    except:
        pass

    if ticker in ["CETETRAC", "M10TRAC", "M5TRAC"]:
        return "Bono Gubernamental"

    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={adjusted_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url).json()
        if "Name" in response and "ETF" in response["Name"]:
            return "ETF"
        elif "Sector" in response:
            return "Acción"
        else:
            return "Fondo de Inversión"
    except:
        return "Fondo de Inversión"

# Ajustar pesos según perfil de riesgo
def adjust_weights(weights, tickers, profile):
    if len(weights) == 0 or len(tickers) == 0:
        return []
    adjusted_weights = np.array(weights)
    for i, ticker in enumerate(tickers):
        asset_type = classify_asset(ticker)
        if profile == "Conservador":
            if asset_type in ["Bono Gubernamental", "ETF"] and ticker in ["CETETRAC", "SHY", "IBGS"]:
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
        elif profile == "Moderado":
            if asset_type in ["Bono Gubernamental", "ETF"] and ticker in ["CETETRAC", "SHY", "IBGS"]:
                adjusted_weights[i] *= 1.0
            else:
                adjusted_weights[i] *= 1.0
        else:  # Agresivo
            if asset_type == "Acción" or ticker in ["NAFTRAC", "IVV", "EZU"]:
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
    adjusted_weights = adjusted_weights / adjusted_weights.sum() if adjusted_weights.sum() != 0 else adjusted_weights
    return adjusted_weights

# Botón para calcular portafolios
if st.button("Calcular Portafolios"):
    # Definir tickers
    tickers_basic = ["CETETRAC", "NAFTRAC", "UDITRAC", "SHY", "IVV", "IBGS", "EZU"]
    tickers_premier = ["ACWI", "EPP", "EEM", "BKF", "ILF", "EZU", "SPY", "EWC", "LCTRAC", "EWZ", "EWG", "EWQ", "EWU", "FXI", "INDI", "EWH", "EWJ", "EWT", "EWY", "EWA", "DIA", "IWM", "ITB", "ITB", "IYH", "IYF", "XLF", "IEO", "QQQ", "IAU", "SLV", "DBO", "UDITRAC", "M10TRAC", "M5TRAC", "IBGS", "SHY", "AGG"]

    # Filtrar tickers según moneda seleccionada
    def filter_tickers_by_currency(tickers, currencies):
        mxn_tickers = ["CETETRAC", "NAFTRAC", "UDITRAC", "LCTRAC", "M10TRAC", "M5TRAC"]
        usd_tickers = ["SHY", "IVV", "ACWI", "EPP", "EEM", "BKF", "ILF", "SPY", "EWC", "EWZ", "EWG", "EWQ", "EWU", "FXI", "INDI", "EWH", "EWJ", "EWT", "EWY", "EWA", "DIA", "IWM", "ITB", "IYH", "IYF", "XLF", "IEO", "QQQ", "IAU", "SLV", "DBO", "AGG"]
        eur_tickers = ["IBGS", "EZU"]
        filtered = []
        for ticker in tickers:
            if "MXN" in currencies and ticker in mxn_tickers:
                filtered.append(ticker)
            if "USD" in currencies and ticker in usd_tickers:
                filtered.append(ticker)
            if "EUR" in currencies and ticker in eur_tickers:
                filtered.append(ticker)
        return filtered

    filtered_tickers_basic = filter_tickers_by_currency(tickers_basic, currency)
    filtered_tickers_premier = filter_tickers_by_currency(tickers_premier, currency)

    st.write("Tickers Básicos Filtrados:", filtered_tickers_basic)
    st.write("Tickers Premier Filtrados:", filtered_tickers_premier)

    if not filtered_tickers_basic or not filtered_tickers_premier:
        st.error("No hay tickers disponibles para las monedas seleccionadas.")
    else:
        # Obtener datos históricos
        prices_basic = load_ticker_data(filtered_tickers_basic)
        st.write("Precios Básicos:", prices_basic)
        returns_basic = calculate_returns(prices_basic)
        st.write("Retornos Básicos:", returns_basic)

        prices_premier = load_ticker_data(filtered_tickers_premier)
        st.write("Precios Premier:", prices_premier)
        returns_premier = calculate_returns(prices_premier)
        st.write("Retornos Premier:", returns_premier)

        if returns_basic.empty or returns_premier.empty:
            st.error("No se pudieron calcular los retornos debido a datos insuficientes.")
        else:
            risk_free_rate = get_risk_free_rate()

            # Optimizar portafolios
            try:
                weights_basic = optimize_portfolio(returns_basic, risk_free_rate)
                if len(weights_basic) == 0:
                    st.error("No se pudieron optimizar los pesos del portafolio Básico.")
                    weights_basic = np.zeros(len(filtered_tickers_basic))
            except Exception as e:
                st.error(f"Error al optimizar el portafolio Básico: {str(e)}")
                weights_basic = np.zeros(len(filtered_tickers_basic))

            try:
                weights_premier = sharpe_ratio_maximization(returns_premier, risk_free_rate)
                if len(weights_premier) == 0:
                    st.error("No se pudieron optimizar los pesos del portafolio Premier.")
                    weights_premier = np.zeros(len(filtered_tickers_premier))
            except Exception as e:
                st.error(f"Error al optimizar el portafolio Premier: {str(e)}")
                weights_premier = np.zeros(len(filtered_tickers_premier))

            # Ajustar pesos según perfil de riesgo
            weights_basic = adjust_weights(weights_basic, filtered_tickers_basic, profile)
            weights_premier = adjust_weights(weights_premier, filtered_tickers_premier, profile)

            # Calcular métricas para "Comparativo Criterios"
            def calculate_criteria_scores(tickers, weights, prices):
                scores = []
                for ticker, weight in zip(tickers, weights):
                    if ticker not in prices.columns:
                        st.warning(f"Datos no disponibles para {ticker}")
                        continue
                    data = prices[ticker]
                    if data.empty or len(data) < 1:
                        st.warning(f"Datos insuficientes para {ticker}")
                        continue
                    # Rentabilidad (CAGR)
                    initial_price = data.iloc[0]
                    final_price = data.iloc[-1]
                    num_years = 10
                    cagr = ((final_price / initial_price) ** (1 / num_years) - 1) * 100 if initial_price > 0 else 0
                    rentabilidad = min(max((cagr + 10) / 2, 1), 10)
                    # Riesgo (Volatilidad)
                    returns = data.pct_change().dropna()
                    vol = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
                    riesgo = min(max((15 - vol) / 1.5, 1), 10)
                    # Diversificación
                    asset_type = classify_asset(ticker)
                    diversificacion = 8 if asset_type == "ETF" else 5
                    # Liquidez
                    stock = yf.Ticker(ticker_mapping.get(ticker, ticker))
                    info = stock.info
                    volume = info.get("averageVolume", 0)
                    liquidez = min(max(volume // 1000000, 1), 10)
                    # Alineación
                    alineacion = 8 if (profile == "Conservador" and asset_type == "Bono Gubernamental") or \
                                     (profile == "Moderado" and asset_type in ["ETF", "Acción"]) or \
                                     (profile == "Agresivo" and asset_type == "Acción") else 5
                    scores.append({
                        "Ticker": ticker_to_name.get(ticker_mapping.get(ticker, ticker), ticker),
                        "Rentabilidad": rentabilidad,
                        "Riesgo": riesgo,
                        "Diversificación": diversificacion,
                        "Liquidez": liquidez,
                        "Alineación": alineacion
                    })
                return pd.DataFrame(scores) if scores else pd.DataFrame()

            # Ponderaciones según perfil
            if profile == "Conservador":
                weights_criteria = {"Rentabilidad": 0.2, "Riesgo": 0.5, "Diversificación": 0.1, "Liquidez": 0.1, "Alineación": 0.1}
            elif profile == "Moderado":
                weights_criteria = {"Rentabilidad": 0.3, "Riesgo": 0.3, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1}
            else:
                weights_criteria = {"Rentabilidad": 0.4, "Riesgo": 0.2, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1}

            scores_basic = calculate_criteria_scores(filtered_tickers_basic, weights_basic, prices_basic)
            scores_premier = calculate_criteria_scores(filtered_tickers_premier, weights_premier, prices_premier)

            # Calcular puntuación total
            for df in [scores_basic, scores_premier]:
                if not df.empty:
                    df["Puntuación Total"] = sum(df[crit] * weight for crit, weight in weights_criteria.items())

            # Gráficos de pastel
            def plot_pie_chart(weights, tickers):
                if len(weights) == 0 or len(tickers) == 0:
                    return px.pie(names=["Sin Datos"], values=[1], title="Asignación por Tipo de Activo")
                asset_types = [classify_asset(ticker) for ticker in tickers]
                df = pd.DataFrame({"Ticker": tickers, "Peso": weights, "Tipo": asset_types})
                grouped = df.groupby("Tipo")["Peso"].sum().reset_index()
                fig = px.pie(grouped, values="Peso", names="Tipo", title="Asignación por Tipo de Activo")
                fig.update_traces(marker=dict(colors=[f"rgb({r},{g},{b})" for r, g, b in [
                    (128, 179, 135), (0, 55, 129), (255, 255, 255), (201, 202, 204), (107, 149, 177)
                ]]))
                return fig

            fig_pie_basic = plot_pie_chart(weights_basic, filtered_tickers_basic)
            fig_pie_premier = plot_pie_chart(weights_premier, filtered_tickers_premier)

            # Gráfico de líneas
            def plot_historical_performance(prices, tickers, weights, benchmark_ticker):
                if prices.empty or len(weights) == 0:
                    return None
                portfolio_value = (prices * weights).sum(axis=1)
                benchmark = load_ticker_data([benchmark_ticker])
                if benchmark_ticker in benchmark.columns:
                    benchmark = benchmark[benchmark_ticker]
                    benchmark = benchmark.reindex(portfolio_value.index, method="ffill")
                    df = pd.DataFrame({
                        "Portafolio": portfolio_value / portfolio_value.iloc[0] * 100,
                        "Benchmark": benchmark / benchmark.iloc[0] * 100
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df["Portafolio"], name="Portafolio"))
                    fig.add_trace(go.Scatter(x=df.index, y=df["Benchmark"], name=benchmark_ticker))
                    fig.update_layout(
                        title="Rendimiento Histórico (10 años)",
                        yaxis_title="Valor Normalizado (%)",
                        template="plotly_white",
                        height=400,
                        plot_bgcolor="rgb(255,255,255)",
                        paper_bgcolor="rgb(255,255,255)"
                    )
                    return fig
                return None

            benchmark = "IPC.MX" if "MXN" in currency else "SPY"
            fig_line_basic = plot_historical_performance(prices_basic, filtered_tickers_basic, weights_basic, benchmark)
            fig_line_premier = plot_historical_performance(prices_premier, filtered_tickers_premier, weights_premier, benchmark)

            # Mostrar resultados
            st.subheader("Portafolio Básico")
            st.write({ticker_to_name.get(ticker_mapping.get(t, t), t): w for t, w in zip(filtered_tickers_basic, weights_basic)})
            st.plotly_chart(fig_pie_basic)
            if fig_line_basic:
                st.plotly_chart(fig_line_basic)

            st.subheader("Portafolio Premier")
            st.write({ticker_to_name.get(ticker_mapping.get(t, t), t): w for t, w in zip(filtered_tickers_premier, weights_premier)})
            st.plotly_chart(fig_pie_premier)
            if fig_line_premier:
                st.plotly_chart(fig_line_premier)

            # Resumen con Grok
            prompt = f"""
            Explica en términos sencillos el perfil de riesgo del usuario basado en su puntaje {score}.
            Usa un lenguaje fácil de entender, como si hablaras con alguien que no sabe de finanzas.
            Evita palabras técnicas. Resume los portafolios Básico y Premier, explicando cómo se dividen las inversiones
            y por qué se ajustan al perfil del usuario ({profile}).
            """
            try:
                headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
                response = requests.post("https://api.x.ai/v1/grok", json={"prompt": prompt}, headers=headers)
                summary = response.json().get("text", "No se pudo generar el resumen.")
            except:
                summary = "No se pudo generar el resumen debido a un error en la API de Grok."

            st.subheader("Resumen de Resultados")
            st.write(summary)

            # Generar PDF
            def generate_pdf_report():
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = []

                elements.append(Image("LogoAllianz.jpeg", width=100, height=50))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Reporte de Perfil de Riesgo y Portafolios", styles['Heading1']))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph(f"Perfil de Riesgo: {profile} (Puntaje: {score})", styles['Heading2']))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Respuestas del Cuestionario:", styles['Heading3']))
                questionnaire_data = [
                    ["Pregunta", "Respuesta"],
                    ["Nivel de Estudios", education],
                    ["Rango de Edad", age],
                    ["Horizonte de Inversión", investment_horizon],
                    ["Conocimiento Financiero", financial_knowledge],
                    ["Tolerancia al Riesgo", risk_tolerance]
                ]
                questionnaire_table = Table(questionnaire_data, colWidths=[200, 200])
                questionnaire_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0, 55/255, 129/255)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                elements.append(questionnaire_table)
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Portafolio Básico", styles['Heading2']))
                basic_data = [["Ticker", "Nombre", "Peso (%)"]] + [
                    [ticker, ticker_to_name.get(ticker_mapping.get(ticker, ticker), ticker), f"{weight*100:.2f}"]
                    for ticker, weight in zip(filtered_tickers_basic, weights_basic)
                ]
                basic_table = Table(basic_data, colWidths=[100, 200, 100])
                basic_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                elements.append(basic_table)
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Comparativo Criterios - Portafolio Básico", styles['Heading3']))
                scores_basic_data = [["Ticker"] + list(weights_criteria.keys()) + ["Puntuación Total"]]
                for _, row in scores_basic.iterrows():
                    scores_basic_data.append([
                        row["Ticker"],
                        f"{row['Rentabilidad']:.1f}",
                        f"{row['Riesgo']:.1f}",
                        f"{row['Diversificación']:.1f}",
                        f"{row['Liquidez']:.1f}",
                        f"{row['Alineación']:.1f}",
                        f"{row['Puntuación Total']:.1f}"
                    ])
                scores_basic_table = Table(scores_basic_data, colWidths=[150] + [60]*6)
                scores_basic_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                elements.append(scores_basic_table)
                elements.append(Spacer(1, 12))

                pie_basic_img = io.BytesIO()
                fig_pie_basic.write_image(pie_basic_img, format='png')
                elements.append(Image(pie_basic_img, width=300, height=200))
                if fig_line_basic:
                    line_basic_img = io.BytesIO()
                    fig_line_basic.write_image(line_basic_img, format='png')
                    elements.append(Image(line_basic_img, width=300, height=200))
                elements.append(PageBreak())

                elements.append(Paragraph("Portafolio Premier", styles['Heading2']))
                premier_data = [["Ticker", "Nombre", "Peso (%)"]] + [
                    [ticker, ticker_to_name.get(ticker_mapping.get(ticker, ticker), ticker), f"{weight*100:.2f}"]
                    for ticker, weight in zip(filtered_tickers_premier, weights_premier)
                ]
                premier_table = Table(premier_data, colWidths=[100, 200, 100])
                premier_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                elements.append(premier_table)
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Comparativo Criterios - Portafolio Premier", styles['Heading3']))
                scores_premier_data = [["Ticker"] + list(weights_criteria.keys()) + ["Puntuación Total"]]
                for _, row in scores_premier.iterrows():
                    scores_premier_data.append([
                        row["Ticker"],
                        f"{row['Rentabilidad']:.1f}",
                        f"{row['Riesgo']:.1f}",
                        f"{row['Diversificación']:.1f}",
                        f"{row['Liquidez']:.1f}",
                        f"{row['Alineación']:.1f}",
                        f"{row['Puntuación Total']:.1f}"
                    ])
                scores_premier_table = Table(scores_premier_data, colWidths=[150] + [60]*6)
                scores_premier_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                elements.append(scores_premier_table)
                elements.append(Spacer(1, 12))

                pie_premier_img = io.BytesIO()
                fig_pie_premier.write_image(pie_premier_img, format='png')
                elements.append(Image(pie_premier_img, width=300, height=200))
                if fig_line_premier:
                    line_premier_img = io.BytesIO()
                    fig_line_premier.write_image(line_premier_img, format='png')
                    elements.append(Image(line_premier_img, width=300, height=200))
                elements.append(PageBreak())

                elements.append(Paragraph("Resumen de Resultados", styles['Heading2']))
                elements.append(Paragraph(summary, styles['BodyText']))
                elements.append(Spacer(1, 12))

                doc.build(elements)
                pdf_buffer.seek(0)
                with open("report.pdf", "wb") as f:
                    f.write(pdf_buffer.read())

            generate_pdf_report()
            with open("report.pdf", "rb") as file:
                st.download_button("Descargar Informe PDF", file, file_name="report.pdf")