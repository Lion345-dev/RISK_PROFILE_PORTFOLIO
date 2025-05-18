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
import logging
from anthropic import Anthropic, APIError

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Verificar clave de Anthropic
if not ANTHROPIC_API_KEY:
    st.error("No se encontró la clave API de Anthropic. Configura ANTHROPIC_API_KEY en el archivo .env.")
    st.stop()

# Inicializar cliente de Anthropic
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Configuración de la página
st.set_page_config(page_title="Perfil de Riesgo y Recomendación de Portafolios", layout="wide")

# Logo y título
st.image("LogoAllianz.jpeg", width=150)
st.title("Simulador de Perfil de Riesgo y Portafolios")

# Mapeo de tickers a nombres y para filtrado
ticker_to_name = {
    "BOND": "PIMCO Active Bond ETF",
    "EWW": "iShares MSCI Mexico ETF",
    "TIP": "iShares TIPS Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "EMLC": "VanEck J.P. Morgan EM Local Currency Bond ETF",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "NEAR": "BlackRock Short Duration Bond ETF",
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "EZU": "iShares MSCI Eurozone ETF",
}

ticker_mapping = {
    "BONDDIAA": "BOND",
    "NAFTRAC": "EWW",
    "UDITRAC": "TIP",
    "IBGS": "LQD",
    "IPCLARGECAP": "EMLC",
    "VGIT": "VGSH",
    "VGSH": "NEAR",
    "BONDDIAA2": "BIL",  # Manejar duplicado de BONDDIAA.MX
    "SPY": "SPY",
    "EZU": "EZU",
}

# Cuestionario de Perfilamiento
st.header("Cuestionario de Perfilamiento")
risk_tolerance = st.selectbox("1. Tolerancia al riesgo", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"], index=0)
investment_horizon = st.selectbox("2. Horizonte de inversión", ["Corto plazo (<1 año)", "Mediano plazo (1-5 años)", "Largo plazo (>5 años)"], index=0)
financial_objective = st.selectbox("3. Objetivo financiero", ["Crecimiento de capital", "Ingresos regulares", "Preservación del capital"], index=0)
financial_situation = st.selectbox("4. Situación financiera personal", ["Muy estable", "Estable", "Moderada", "Inestable"], index=0)
market_knowledge = st.selectbox("5. Conocimiento del mercado", ["Ninguno", "Básico", "Intermedio", "Avanzado"], index=0)
liquidity_need = st.selectbox("6. Necesidad de liquidez", ["Inmediato", "1-6 meses", ">6 meses"], index=0)
age = st.selectbox("7. Edad", ["<30 años", "30-50 años", ">50 años"], index=0)

# Calcular puntaje
total_score = (
    {"Muy bajo": 1, "Bajo": 2, "Moderado": 3, "Alto": 4, "Muy alto": 5}[risk_tolerance] +
    {"Corto plazo (<1 año)": 1, "Mediano plazo (1-5 años)": 2, "Largo plazo (>5 años)": 3}[investment_horizon] +
    {"Crecimiento de capital": 2, "Ingresos regulares": 1, "Preservación del capital": 0}[financial_objective] +
    {"Muy estable": 3, "Estable": 2, "Moderada": 1, "Inestable": 0}[financial_situation] +
    {"Ninguno": 0, "Básico": 1, "Intermedio": 2, "Avanzado": 3}[market_knowledge] +
    {"Inmediato": 0, "1-6 meses": 1, ">6 meses": 2}[liquidity_need] +
    {"<30 años": 3, "30-50 años": 2, ">50 años": 1}[age]
)

# Determinar perfil
if total_score <= 6:
    profile = "Conservador"
elif total_score <= 12:
    profile = "Moderado"
elif total_score <= 18:
    profile = "Balanceado"
elif total_score <= 24:
    profile = "Crecimiento"
else:
    profile = "Oportunidad"

st.success(f"Tu perfil de inversionista es: {profile} (Puntaje: {total_score})")

# Selección de moneda
currency = st.multiselect("Selecciona las monedas para tu portafolio", ["MXN", "USD", "EUR"], default=["MXN", "USD", "EUR"])

# Función para recomendar tickers con Claude
def recomendar_activos_con_claude(perfil, portfolio_type="basic", currencies=["MXN", "USD", "EUR"]):
    currency_prompt = ", ".join(
        [f"{c} ({'.MX' if c == 'MXN' else '.DE' if c == 'EUR' else 'sin sufijo'})" for c in currencies]
    )
    prompt = f"""
    Eres un asesor financiero experto. Según el perfil de riesgo '{perfil}', recomienda 5 activos internacionales diversificados
    (acciones, ETFs o fondos) adecuados para un portafolio {'básico' if portfolio_type == 'basic' else 'premier'}.
    Los activos deben estar listados en bolsas compatibles con las monedas: {currency_prompt}.
    Ejemplo de tickers: ["BOND", "EWW", "TIP"]. Devuelve sólo una lista de Python con los tickers.
    """
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.7,
            system="Eres un asesor financiero experto.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response.content[0].text
        tickers = eval(content.strip())
        # Filtrar tickers por moneda
        valid_tickers = []
        for ticker in tickers:
            if "MXN" in currencies and ticker.endswith('.MX'):
                valid_tickers.append(ticker)
            elif "USD" in currencies and not ticker.endswith('.MX') and not ticker.endswith('.DE'):
                valid_tickers.append(ticker)
            elif "EUR" in currencies and ticker.endswith('.DE'):
                valid_tickers.append(ticker)
        return valid_tickers[:5]
    except APIError as e:
        if e.status_code == 429:
            logger.error("Límite de créditos de Anthropic alcanzado.")
            return []
        logger.error(f"Error en la solicitud a Anthropic: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error interpretando respuesta de Anthropic: {str(e)}")
        return []

# Función para verificar tickers válidos
@st.cache_data(ttl=86400)
def validate_tickers(tickers):
    valid_tickers = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info.get("regularMarketPrice") is not None:
                valid_tickers.append(ticker)
                logger.info(f"Ticker {ticker} validado correctamente.")
            else:
                logger.warning(f"Ticker {ticker} no tiene datos válidos.")
            time.sleep(3)
        except Exception as e:
            logger.warning(f"Error al validar {ticker}: {str(e)}")
            time.sleep(3)
    st.write(f"Tickers validados: {valid_tickers}")
    return valid_tickers

# Función para cargar datos históricos
@st.cache_data(ttl=3600)
def load_ticker_data(tickers, period="5y", interval="1mo"):
    data_dict = {}
    max_retries = 2
    retry_delay = 10
    alpha_vantage_requests = 0
    alpha_vantage_limit = 5

    for ticker in tickers:
        # Intentar con Yahoo Finance
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
                if not data.empty and 'Close' in data:
                    data = data['Close'].dropna()
                    if len(data) >= 2:
                        data_dict[ticker] = data
                        logger.info(f"{ticker}: {data.shape} filas descargadas.")
                        break
                    else:
                        logger.warning(f"{ticker}: Datos insuficientes ({len(data)} filas).")
                else:
                    logger.warning(f"{ticker}: DataFrame vacío o sin 'Close'.")
                time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"{ticker}: Error en yfinance - {str(e)}")
                time.sleep(retry_delay)

        # Intentar con Alpha Vantage
        if ticker not in data_dict:
            if alpha_vantage_requests >= alpha_vantage_limit:
                logger.warning(f"Límite de Alpha Vantage alcanzado ({alpha_vantage_limit} solicitudes).")
                continue
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url).json()
                alpha_vantage_requests += 1
                if "Monthly Time Series" in response:
                    time_series = response["Monthly Time Series"]
                    dates = []
                    prices = []
                    for date, values in time_series.items():
                        dates.append(pd.to_datetime(date))
                        prices.append(float(values["4. close"]))
                    data = pd.Series(prices, index=dates, name=ticker).sort_index()
                    end_date = datetime.now()
                    start_date = end_date - pd.Timedelta(days=5*365)
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
                    if len(data) >= 2:
                        data_dict[ticker] = data
                        logger.info(f"{ticker}: {data.shape} filas desde Alpha Vantage.")
                    else:
                        logger.warning(f"{ticker}: Datos insuficientes desde Alpha Vantage ({len(data)} filas).")
                else:
                    logger.warning(f"No se encontraron datos en Alpha Vantage para {ticker}.")
                time.sleep(12)
            except Exception as e:
                logger.warning(f"Error al cargar datos de Alpha Vantage para {ticker}: {str(e)}")

    if not data_dict:
        logger.error("No se obtuvieron datos para ningún ticker.")
        st.error("No se pudieron obtener datos para los tickers seleccionados. Verifica tu conexión o intenta más tarde.")
        return pd.DataFrame()

    combined_data = pd.DataFrame(data_dict)
    return combined_data.dropna()

# Función para calcular retornos
def calculate_returns(prices):
    if prices.empty or len(prices) < 2:
        logger.error("DataFrame de precios vacío o con menos de 2 filas.")
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    if returns.empty:
        logger.error("No se pudieron calcular retornos: todos los valores son NaN o insuficientes.")
    return returns

# Función para clasificar activos
def classify_asset(ticker):
    try:
        url = f"https://elite.finviz.com/export.ashx?v=111&f=sym_{ticker}&auth={FINVIZ_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return "Acción"
    except:
        pass

    if ticker.endswith('.MX'):
        return "Bono Gubernamental"

    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
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
        if profile in ["Conservador", "Moderado"]:
            if asset_type in ["Bono Gubernamental", "ETF"] and (ticker in ["BOND", "TIP", "LQD", "EMLC", "VGSH", "NEAR", "BIL"]):
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
        elif profile == "Balanceado":
            adjusted_weights[i] *= 1.0
        else:  # Crecimiento, Oportunidad
            if asset_type == "Acción" or ticker in ["EWW", "SPY", "EZU"]:
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
    adjusted_weights = adjusted_weights / adjusted_weights.sum() if adjusted_weights.sum() != 0 else adjusted_weights
    return adjusted_weights

# Botón para calcular portafolios
if st.button("Calcular Portafolios"):
    with st.spinner("Obteniendo recomendaciones de Claude..."):
        # Obtener tickers recomendados por Claude
        tickers_basic = recomendar_activos_con_claude(profile, "basic", currency)
        tickers_premier = recomendar_activos_con_claude(profile, "premier", currency)

        # Validar tickers
        tickers_basic = validate_tickers(tickers_basic)
        tickers_premier = validate_tickers(tickers_premier)

        # Filtrar tickers según moneda seleccionada
        def filter_tickers_by_currency(tickers, currencies):
            filtered = []
            mxn_tickers = []
            usd_tickers = ["BOND", "EWW", "TIP", "LQD", "EMLC", "VGSH", "NEAR", "BIL", "SPY"]
            eur_tickers = ["EZU"]
            for ticker in tickers:
                mapped_ticker = ticker_mapping.get(ticker, ticker)
                if "MXN" in currencies and (ticker.endswith('.MX') or ticker in mxn_tickers):
                    filtered.append(mapped_ticker)
                elif "USD" in currencies and (not ticker.endswith('.MX') and not ticker.endswith('.DE') or ticker in usd_tickers):
                    filtered.append(mapped_ticker)
                elif "EUR" in currencies and (ticker.endswith('.DE') or ticker in eur_tickers):
                    filtered.append(mapped_ticker)
            return filtered

        filtered_tickers_basic = filter_tickers_by_currency(tickers_basic, currency)
        filtered_tickers_premier = filter_tickers_by_currency(tickers_premier, currency)

        # Fallback a tickers predeterminados
        if not filtered_tickers_basic:
            default_basic = ["BOND", "EWW", "TIP", "LQD", "EMLC", "VGSH", "NEAR", "SPY"]
            default_basic_mapped = [ticker_mapping.get(t, t) for t in default_basic]
            filtered_tickers_basic = filter_tickers_by_currency(validate_tickers(default_basic_mapped), currency)
        if not filtered_tickers_premier:
            default_premier = ["BIL", "EWW", "LQD", "VGSH", "NEAR", "SPY", "EZU"]
            default_premier_mapped = [ticker_mapping.get(t, t) for t in default_premier]
            filtered_tickers_premier = filter_tickers_by_currency(validate_tickers(default_premier_mapped), currency)

        st.write("Tickers Básicos Filtrados:", filtered_tickers_basic)
        st.write("Tickers Premier Filtrados:", filtered_tickers_premier)

        if not filtered_tickers_basic or not filtered_tickers_premier:
            st.error("No se encontraron tickers válidos para las monedas seleccionadas. Posibles causas: "
                     "1) Claude no devolvió tickers compatibles, 2) Yahoo Finance/Alpha Vantage no tienen datos, "
                     "3) Límite de tasa alcanzado. Intenta de nuevo más tarde o usa solo USD.")
        else:
            # Obtener datos históricos
            prices_basic = load_ticker_data(filtered_tickers_basic)
            returns_basic = calculate_returns(prices_basic)

            prices_premier = load_ticker_data(filtered_tickers_premier)
            returns_premier = calculate_returns(prices_premier)

            if returns_basic.empty or returns_premier.empty:
                st.error("No se pudieron calcular los retornos debido a datos insuficientes. "
                         "Verifica los logs para detalles o intenta con menos tickers.")
                weights_basic = np.zeros(len(filtered_tickers_basic))
                weights_premier = np.zeros(len(filtered_tickers_premier))
                scores_basic = pd.DataFrame()
                scores_premier = pd.DataFrame()
            else:
                risk_free_rate = get_risk_free_rate()

                # Optimizar portafolios
                try:
                    weights_basic = optimize_portfolio(returns_basic, risk_free_rate)
                    if len(weights_basic) == 0:
                        logger.error("No se pudieron optimizar los pesos del portafolio Básico.")
                        weights_basic = np.zeros(len(filtered_tickers_basic))
                except Exception as e:
                    logger.error(f"Error al optimizar el portafolio Básico: {str(e)}")
                    weights_basic = np.zeros(len(filtered_tickers_basic))

                try:
                    weights_premier = sharpe_ratio_maximization(returns_premier, risk_free_rate)
                    if len(weights_premier) == 0:
                        logger.error("No se pudieron optimizar los pesos del portafolio Premier.")
                        weights_premier = np.zeros(len(filtered_tickers_premier))
                except Exception as e:
                    logger.error(f"Error al optimizar el portafolio Premier: {str(e)}")
                    weights_premier = np.zeros(len(filtered_tickers_premier))

                # Ajustar pesos según perfil
                weights_basic = adjust_weights(weights_basic, filtered_tickers_basic, profile)
                weights_premier = adjust_weights(weights_premier, filtered_tickers_premier, profile)

                # Calcular métricas para "Comparativo Criterios"
                def calculate_criteria_scores(tickers, weights, prices):
                    scores = []
                    for ticker, weight in zip(tickers, weights):
                        if ticker not in prices.columns:
                            logger.warning(f"Datos no disponibles para {ticker}")
                            continue
                        data = prices[ticker]
                        if data.empty or len(data) < 2:
                            logger.warning(f"Datos insuficientes para {ticker}")
                            continue
                        initial_price = data.iloc[0]
                        final_price = data.iloc[-1]
                        num_years = 5
                        cagr = ((final_price / initial_price) ** (1 / num_years) - 1) * 100 if initial_price > 0 else 0
                        rentabilidad = min(max((cagr + 10) / 2, 1), 10)
                        returns = data.pct_change().dropna()
                        vol = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
                        riesgo = min(max((15 - vol) / 1.5, 1), 10)
                        asset_type = classify_asset(ticker)
                        diversificacion = 8 if asset_type == "ETF" else 5
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        volume = info.get("averageVolume", 0)
                        liquidez = min(max(volume // 1000000, 1), 10)
                        alineacion = 8 if (profile in ["Conservador", "Moderado"] and asset_type == "Bono Gubernamental") or \
                                        (profile == "Balanceado" and asset_type in ["ETF", "Acción"]) or \
                                        (profile in ["Crecimiento", "Oportunidad"] and asset_type == "Acción") else 5
                        scores.append({
                            "Ticker": ticker_to_name.get(ticker, ticker),
                            "Rentabilidad": rentabilidad,
                            "Riesgo": riesgo,
                            "Diversificación": diversificacion,
                            "Liquidez": liquidez,
                            "Alineación": alineacion
                        })
                    return pd.DataFrame(scores) if scores else pd.DataFrame()

                weights_criteria = {
                    "Conservador": {"Rentabilidad": 0.2, "Riesgo": 0.5, "Diversificación": 0.1, "Liquidez": 0.1, "Alineación": 0.1},
                    "Moderado": {"Rentabilidad": 0.3, "Riesgo": 0.3, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
                    "Balanceado": {"Rentabilidad": 0.35, "Riesgo": 0.25, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
                    "Crecimiento": {"Rentabilidad": 0.4, "Riesgo": 0.2, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
                    "Oportunidad": {"Rentabilidad": 0.45, "Riesgo": 0.15, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1}
                }[profile]

                scores_basic = calculate_criteria_scores(filtered_tickers_basic, weights_basic, prices_basic)
                scores_premier = calculate_criteria_scores(filtered_tickers_premier, weights_premier, prices_premier)

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
                        title="Rendimiento Histórico (5 años)",
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

            # Almacenar datos para el reporte
            st.session_state['report_data'] = {
                'profile': profile,
                'score': total_score,
                'tickers_basic': filtered_tickers_basic,
                'tickers_premier': filtered_tickers_premier,
                'prices_basic': prices_basic,
                'prices_premier': prices_premier,
                'returns_basic': returns_basic,
                'returns_premier': returns_premier,
                'weights_basic': weights_basic,
                'weights_premier': weights_premier,
                'scores_basic': scores_basic,
                'scores_premier': scores_premier,
                'fig_pie_basic': fig_pie_basic,
                'fig_pie_premier': fig_pie_premier,
                'fig_line_basic': fig_line_basic,
                'fig_line_premier': fig_line_premier
            }

            # Mostrar resultados
            st.subheader("Portafolio Básico")
            st.write({ticker_to_name.get(t, t): w for t, w in zip(filtered_tickers_basic, weights_basic)})
            st.plotly_chart(fig_pie_basic)
            if fig_line_basic:
                st.plotly_chart(fig_line_basic)

            st.subheader("Portafolio Premier")
            st.write({ticker_to_name.get(t, t): w for t, w in zip(filtered_tickers_premier, weights_premier)})
            st.plotly_chart(fig_pie_premier)
            if fig_line_premier:
                st.plotly_chart(fig_line_premier)

            # Resumen con Grok
            prompt = f"""
            Explica en términos sencillos el perfil de riesgo del usuario basado en su puntaje {total_score}.
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
                logger.error("Error en la API de Grok.")

            st.session_state['report_data']['summary'] = summary
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

                elements.append(Paragraph(f"Perfil de Riesgo: {profile} (Puntaje: {total_score})", styles['Heading2']))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Respuestas del Cuestionario:", styles['Heading3']))
                questionnaire_data = [
                    ["Pregunta", "Respuesta"],
                    ["Tolerancia al riesgo", risk_tolerance],
                    ["Horizonte de inversión", investment_horizon],
                    ["Objetivo financiero", financial_objective],
                    ["Situación financiera", financial_situation],
                    ["Conocimiento del mercado", market_knowledge],
                    ["Necesidad de liquidez", liquidity_need],
                    ["Edad", age]
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
                    [ticker, ticker_to_name.get(ticker, ticker), f"{weight*100:.2f}"]
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
                    [ticker, ticker_to_name.get(ticker, ticker), f"{weight*100:.2f}"]
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