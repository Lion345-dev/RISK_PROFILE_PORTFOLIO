import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import time
import random
from yahooquery import Ticker
import logging
from anthropic import Anthropic, APIError
import visuals  # Importar el módulo visuals para gráficos
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Verificar claves de API
FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY")
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    st.error("No se encontró la clave API de Anthropic. Configura ANTHROPIC_API_KEY en el archivo .env.")
    st.stop()

# Inicializar cliente de Anthropic
client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# Configuración de la página
st.set_page_config(page_title="Perfil de Riesgo y Recomendación de Portafolios", layout="wide")

# Logo y título
try:
    st.image("LogoAllianz.jpeg", width=150)
except Exception as e:
    st.warning(f"No se pudo cargar el logo 'LogoAllianz.jpeg': {str(e)}")
st.title("Simulador de Perfil de Riesgo y Portafolios")

# Mapeo de tickers a nombres
ticker_to_name = {
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
    "EWW": "iShares MSCI Mexico ETF",
    "TIP": "iShares TIPS Bond ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "IBGS.AS": "iShares Global Govt Bond UCITS ETF",
    "EZU": "iShares MSCI Eurozone ETF",
    "EPP": "iShares MSCI Pacific ex Japan ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "BKF": "iShares MSCI BRIC ETF",
    "ILF": "iShares Latin America 40 ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "EWC": "iShares MSCI Canada ETF",
    "EWZ": "iShares MSCI Brazil ETF",
    "EWG": "iShares MSCI Germany ETF",
    "EWQ": "iShares MSCI France ETF",
    "EWU": "iShares MSCI United Kingdom ETF",
    "FXI": "iShares China Large-Cap ETF",
    "EWH": "iShares MSCI Hong Kong ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWT": "iShares MSCI Taiwan ETF",
    "EWY": "iShares MSCI South Korea ETF",
    "EWA": "iShares MSCI Australia ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
    "IWM": "iShares Russell 2000 ETF",
    "ITB": "iShares U.S. Home Construction ETF",
    "IYH": "iShares U.S. Healthcare ETF",
    "IYF": "iShares U.S. Financials ETF",
    "XLF": "Financial Select Sector SPDR Fund",
    "IEO": "iShares U.S. Oil & Gas Exploration & Production ETF",
    "QQQ": "Invesco QQQ Trust",
    "IAU": "iShares Gold Trust",
    "SLV": "iShares Silver Trust",
    "DBO": "Invesco DB Oil Fund",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "VGIT": "Vanguard Intermediate-Term Treasury ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
}

ticker_mapping = {
    "SPY": "SPY",
    "EZU": "EZU",
}

# Definir portafolios
portafolio_basico = ["BIL", "EWW", "TIP", "ACWI", "SHY", "IVV", "IBGS.AS", "EZU"]
portafolio_premier = [
    "ACWI", "EPP", "EEM", "BKF", "ILF", "EZU", "SPY", "EWC", "EWZ", "EWG", "EWQ", "EWU", "FXI",
    "EWH", "EWJ", "EWT", "EWY", "EWA", "DIA", "IWM", "ITB", "IYH", "IYF", "XLF", "IEO",
    "QQQ", "IAU", "SLV", "DBO", "TIP", "VGSH", "VGIT", "IBGS.AS", "SHY", "BIL", "AGG"
]

# Tickers de respaldo confiables
fallback_tickers = ["SPY", "IVV", "QQQ", "AGG", "TIP"]

# Cuestionario de Perfilamiento
st.header("Cuestionario de Perfilamiento")
risk_tolerance = st.selectbox("1. Tolerancia al riesgo", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"], index=0)
investment_horizon = st.selectbox("2. Horizonte de inversión", ["Corto plazo (<1 año)", "Mediano plazo (1-5 años)", "Largo plazo (>5 años)"], index=0)
financial_objective = st.selectbox("3. Objetivo financiero", ["Crecimiento de capital", "Ingresos regulares", "Preservación del capital"], index=0)
financial_situation = st.selectbox("4. Situación financiera personal", ["Muy estable", "Estable", "Moderada", "Inestable"], index=0)
market_knowledge = st.selectbox("5. Conocimiento del mercado", ["Ninguno", "Básico", "Intermedio", "Avanzado"], index=0)
liquidity_need = st.selectbox("6. Necesidad de liquidez", ["Inmediato", "1-6 meses", ">6 meses"], index=0)
age = st.selectbox("7. Edad", ["<30 años", "30-50 años", ">50 años"], index=0)

# Calcular puntaje y perfil
def calculate_risk_profile(responses):
    score = sum(responses)
    if score <= 6:
        return "Conservador", score
    elif score <= 12:
        return "Moderado", score
    elif score <= 18:
        return "Balanceado", score
    elif score <= 24:
        return "Crecimiento", score
    else:
        return "Oportunidad", score

responses = [
    {"Muy bajo": 1, "Bajo": 2, "Moderado": 3, "Alto": 4, "Muy alto": 5}[risk_tolerance],
    {"Corto plazo (<1 año)": 1, "Mediano plazo (1-5 años)": 2, "Largo plazo (>5 años)": 3}[investment_horizon],
    {"Crecimiento de capital": 2, "Ingresos regulares": 1, "Preservación del capital": 0}[financial_objective],
    {"Muy estable": 3, "Estable": 2, "Moderada": 1, "Inestable": 0}[financial_situation],
    {"Ninguno": 0, "Básico": 1, "Intermedio": 2, "Avanzado": 3}[market_knowledge],
    {"Inmediato": 0, "1-6 meses": 1, ">6 meses": 2}[liquidity_need],
    {"<30 años": 3, "30-50 años": 2, ">50 años": 1}[age]
]

profile, total_score = calculate_risk_profile(responses)
st.success(f"Tu perfil de inversionista es: {profile} (Puntaje: {total_score})")

# Selección de moneda
currency = st.multiselect("Selecciona las monedas para tu portafolio", ["MXN", "USD", "EUR"], default=["MXN", "USD", "EUR"])

# Función auxiliar para validar DataFrames
def validate_dataframe(df, name="DataFrame"):
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{name} no es un DataFrame, es {type(df)}")
        return False
    if df.empty:
        logger.error(f"{name} está vacío")
        return False
    if len(df.columns) == 0:
        logger.error(f"{name} no tiene columnas")
        return False
    if all(df[col].isna().all() for col in df.columns):
        logger.error(f"{name} contiene solo valores NaN en todas las columnas")
        return False
    logger.info(f"{name} validado correctamente: {df.shape} filas y columnas")
    return True

# Función para obtener datos históricos con yahooquery
def fetch_historical_data(tickers, period="2y", interval="1d", max_retries=3):
    """Obtiene datos históricos de cierre ajustados usando yahooquery con reintentos"""
    if not tickers:
        logger.error("Lista de tickers vacía")
        return pd.DataFrame()

    prices_df = pd.DataFrame()
    valid_tickers = []
    failed_tickers = []

    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                logger.info(f"⏳ Descargando datos de {ticker} (Intento {attempt + 1}/{max_retries})...")
                ticker_obj = Ticker(ticker, asynchronous=True)
                data = ticker_obj.history(period=period, interval=interval)
                if data is not None and not data.empty and 'adjclose' in data.columns:
                    daily_data = data.groupby(data.index)['adjclose'].last().to_frame(name=ticker)
                    prices_df = pd.concat([prices_df, daily_data], axis=1) if not prices_df.empty else daily_data
                    valid_tickers.append(ticker)
                    logger.info(f"✅ Datos de {ticker} obtenidos: {len(daily_data)} filas")
                    break
                else:
                    logger.warning(f"⚠️ No se encontraron datos válidos para {ticker} con {period}, intentando 1mo...")
                    data_short = ticker_obj.history(period="1mo", interval="1d")
                    if data_short is not None and not data_short.empty and 'adjclose' in data_short.columns:
                        daily_data_short = data_short.groupby(data_short.index)['adjclose'].last().to_frame(name=ticker)
                        prices_df = pd.concat([prices_df, daily_data_short], axis=1) if not prices_df.empty else daily_data_short
                        valid_tickers.append(ticker)
                        logger.info(f"✅ Datos de {ticker} obtenidos con período corto: {len(daily_data_short)} filas")
                        break
            except Exception as e:
                logger.error(f"❌ Error al descargar datos de {ticker} (Intento {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    failed_tickers.append(ticker)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))  # Espera exponencial con jitter

    if failed_tickers:
        st.warning(f"No se encontraron datos históricos para los siguientes tickers: {failed_tickers}")
        if not valid_tickers and len(failed_tickers) == len(tickers):
            logger.warning("Todos los tickers fallaron, usando tickers de respaldo")
            for fb_ticker in fallback_tickers:
                for attempt in range(max_retries):
                    try:
                        ticker_obj = Ticker(fb_ticker, asynchronous=True)
                        data_fb = ticker_obj.history(period=period, interval=interval)
                        if data_fb is not None and not data_fb.empty and 'adjclose' in data_fb.columns:
                            daily_data_fb = data_fb.groupby(data_fb.index)['adjclose'].last().to_frame(name=fb_ticker)
                            prices_df = pd.concat([prices_df, daily_data_fb], axis=1) if not prices_df.empty else daily_data_fb
                            valid_tickers.append(fb_ticker)
                            logger.info(f"✅ Datos de respaldo {fb_ticker} obtenidos: {len(daily_data_fb)} filas")
                            break
                    except Exception as e:
                        logger.error(f"❌ Error con ticker de respaldo {fb_ticker}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + random.uniform(0, 1))

    if prices_df.empty:
        logger.error("DataFrame inicial vacío antes de dropna")
        return pd.DataFrame()

    logger.info(f"Estado de prices_df antes de dropna: {prices_df.shape}, NaN total: {prices_df.isna().sum().sum()}")
    prices_df = prices_df.dropna(how='all')
    logger.info(f"Estado de prices_df después de dropna: {prices_df.shape}, NaN total: {prices_df.isna().sum().sum()}")

    if prices_df.empty:
        logger.error("DataFrame vacío después de dropna")
        return pd.DataFrame()

    prices_df = prices_df.apply(pd.to_numeric, errors='coerce')

    if not validate_dataframe(prices_df, f"Datos históricos para {valid_tickers}"):
        logger.error(f"Datos históricos no válidos para {valid_tickers}")
        return pd.DataFrame()

    logger.info(f"Datos históricos obtenidos para {valid_tickers}: {prices_df.shape}")
    return prices_df

def get_etf_sector_data(ticker, max_retries=3):
    """Obtiene datos sectoriales del ETF desde Finviz con reintentos"""
    if not FINVIZ_API_KEY:
        logger.warning("No se proporcionó FINVIZ_API_KEY, usando fallback")
        return pd.DataFrame()

    for attempt in range(max_retries):
        try:
            url = f"https://elite.finviz.com/export.ashx?v=111&f=sym_{ticker}&auth={FINVIZ_API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                logger.info(f"Datos sectoriales obtenidos para {ticker}")
                return df
            elif response.status_code == 429:
                logger.warning(f"Límite de tasa alcanzado en Finviz para {ticker}, reintentando...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                else:
                    logger.error(f"Error 429 persistente para {ticker}, usando fallback")
                    return pd.DataFrame()
            else:
                logger.error(f"Error al obtener datos de Finviz para {ticker}: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Excepción al obtener datos de Finviz para {ticker}: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

def calculate_returns(prices):
    """Calcula retornos diarios"""
    if not validate_dataframe(prices, "Precios"):
        logger.error("No se pueden calcular retornos: precios inválidos")
        return pd.DataFrame()
    if len(prices) < 2:
        logger.error("No hay suficientes datos para calcular retornos (mínimo 2 filas requeridas)")
        return pd.DataFrame()
    try:
        returns = prices.pct_change().dropna(how='all')
        if not validate_dataframe(returns, "Retornos"):
            logger.error("Retornos calculados no válidos")
            return pd.DataFrame()
        logger.info(f"Retornos calculados para {list(prices.columns)} con {len(returns)} filas")
        return returns
    except Exception as e:
        logger.error(f"Error al calcular retornos: {str(e)}")
        return pd.DataFrame()

def get_risk_free_rate(max_retries=3):
    """Obtiene la tasa libre de riesgo de Cetes 28 días desde Banxico"""
    if not BANXICO_TOKEN:
        logger.warning("No se proporcionó BANXICO_TOKEN, usando tasa por defecto: 0.05")
        return 0.05

    for attempt in range(max_retries):
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
                if attempt == max_retries - 1:
                    logger.warning("Usando tasa por defecto: 0.05")
                    return 0.05
        except Exception as e:
            logger.error(f"Excepción al obtener tasa de Banxico: {str(e)}")
            if attempt == max_retries - 1:
                logger.warning("Usando tasa por defecto: 0.05")
                return 0.05
        time.sleep(2 ** attempt + random.uniform(0, 1))
    return 0.05

def optimize_portfolio(returns, risk_free_rate):
    """Maximiza el ratio de Sharpe para el portafolio (Markowitz)"""
    if not validate_dataframe(returns, "Retornos"):
        logger.error("No se puede optimizar el portafolio: retornos inválidos")
        return np.array([])

    num_assets = len(returns.columns)
    if num_assets == 0:
        logger.error("No hay activos para optimizar")
        return np.array([])

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol != 0 else np.inf

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x},
    )
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    try:
        result = minimize(
            neg_sharpe_ratio,
            initial_guess,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            logger.info("Maximización de Sharpe exitosa")
            return result.x
        else:
            logger.error("Maximización de Sharpe fallida: " + result.message)
            return np.zeros(num_assets)
    except Exception as e:
        logger.error(f"Error en maximización de Sharpe: {str(e)}")
        return np.zeros(num_assets)

@st.cache_data(ttl=3600)
def validate_tickers(tickers, max_retries=3):
    """Valida que los tickers tengan datos disponibles usando yahooquery"""
    valid_tickers = []
    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                ticker_obj = Ticker(ticker, asynchronous=True)
                data = ticker_obj.history(period="1d", interval="1d")
                if data is not None and not data.empty and 'adjclose' in data.columns:
                    valid_tickers.append(ticker)
                    logger.info(f"Ticker {ticker} validado correctamente.")
                    break
                else:
                    logger.warning(f"Ticker {ticker} no tiene datos válidos con 1d, intentando 1mo...")
                    data_short = ticker_obj.history(period="1mo", interval="1d")
                    if data_short is not None and not data_short.empty and 'adjclose' in data_short.columns:
                        valid_tickers.append(ticker)
                        logger.info(f"Ticker {ticker} validado con período corto.")
                        break
            except Exception as e:
                logger.warning(f"Error al validar ticker {ticker}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    return valid_tickers

def recomendar_activos_con_claude(perfil, portfolio_type="basic", currencies=["MXN", "USD", "EUR"], max_retries=3):
    """Recomienda tickers usando Claude"""
    available_tickers = portafolio_basico if portfolio_type == "basic" else portafolio_premier
    tickers_string = ", ".join([f"'{t}'" for t in available_tickers])
    currency_prompt = ", ".join([f"{c} ({'.MX' if c == 'MXN' else '.DE' if c == 'EUR' else 'sin sufijo'})" for c in currencies])
    prompt = f"""
    Eres un asesor financiero experto. Según el perfil de riesgo '{perfil}', recomienda 5 activos de entre los tickers disponibles: {tickers_string}.
    Los activos deben estar listados en bolsas compatibles con las monedas: {currency_prompt}.
    Devuelve sólo una lista de Python con los tickers, seleccionando únicamente de los proporcionados.
    """
    for attempt in range(max_retries):
        try:
            response = client_anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.7,
                system="Eres un asesor financiero experto.",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            tickers = eval(content)
            if not isinstance(tickers, list) or len(tickers) != 5:
                logger.warning("Respuesta de Claude no válida, usando tickers por defecto")
                return available_tickers[:5]
            valid_tickers = [t for t in tickers if t in available_tickers]
            return valid_tickers[:5] if len(valid_tickers) >= 5 else available_tickers[:5]
        except APIError as e:
            if "429" in str(e):
                logger.warning(f"Límite de tasa alcanzado en Anthropic, reintentando en {2 ** attempt + random.uniform(0, 1)} segundos...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                else:
                    logger.error(f"Error persistente 429 en Anthropic: {str(e)}")
                    return available_tickers[:5]
            else:
                logger.error(f"Error en la solicitud a Anthropic: {str(e)}")
                return available_tickers[:5]
        except Exception as e:
            logger.error(f"Error interpretando respuesta de Anthropic: {str(e)}")
            return available_tickers[:5]
    return available_tickers[:5]

def classify_asset(ticker):
    """Clasifica el tipo de activo con fallback si Finviz falla"""
    sector_data = get_etf_sector_data(ticker)
    if sector_data.empty:
        logger.warning(f"Fallo al clasificar {ticker} con Finviz, usando fallback")
        return "Bono Gubernamental" if ticker.endswith('.MX') else "Fondo de Inversión"
    return "Acción"

def adjust_weights(weights, tickers, profile):
    """Ajusta pesos según el perfil de riesgo"""
    if len(weights) == 0 or len(tickers) == 0 or all(w == 0 for w in weights):
        logger.warning("Pesos o tickers inválidos, asignando pesos uniformes")
        return np.ones(len(tickers)) / len(tickers) if len(tickers) > 0 else np.array([])
    adjusted_weights = np.array(weights)
    for i, ticker in enumerate(tickers):
        asset_type = classify_asset(ticker)
        if profile in ["Conservador", "Moderado"]:
            if asset_type == "Bono Gubernamental" or ticker in ["TIP", "VGSH", "BIL", "VGIT", "AGG", "SHY", "IBGS.AS"]:
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
        elif profile == "Balanceado":
            adjusted_weights[i] *= 1.0
        else:
            if asset_type == "Acción" or ticker in ["EWW", "SPY", "EZU", "IVV", "ACWI"]:
                adjusted_weights[i] *= 1.5
            else:
                adjusted_weights[i] *= 0.5
    if adjusted_weights.sum() == 0:
        logger.warning("Suma de pesos ajustados es cero, asignando pesos uniformes")
        return np.ones(len(tickers)) / len(tickers)
    return adjusted_weights / adjusted_weights.sum()

def filter_tickers_by_currency(tickers, currencies):
    """Filtra tickers según las monedas seleccionadas"""
    filtered = []
    for ticker in tickers:
        mapped_ticker = ticker_mapping.get(ticker, ticker)
        if "USD" in currencies and not (ticker.endswith('.MX') or ticker.endswith('.DE')):
            filtered.append(mapped_ticker)
        elif "MXN" in currencies and ticker.endswith('.MX'):
            filtered.append(mapped_ticker)
        elif "EUR" in currencies and ticker.endswith('.DE'):
            filtered.append(mapped_ticker)
    logger.info(f"Tickers después de filtrar por monedas {currencies}: {filtered}")
    return filtered

def ensure_valid_tickers(tickers, portfolio_type="basic"):
    """Asegura que los tickers tengan datos históricos válidos; usa tickers de respaldo si es necesario"""
    valid_tickers = validate_tickers(tickers)
    if len(valid_tickers) < 3:  # Mínimo 3 tickers para un portafolio significativo
        logger.warning(f"No hay suficientes tickers válidos ({valid_tickers}) para {portfolio_type}, usando tickers de respaldo")
        fallback_valid = validate_tickers(fallback_tickers)
        valid_tickers.extend(fallback_valid[:5 - len(valid_tickers)])
        if len(valid_tickers) < 3:
            logger.error(f"No se encontraron suficientes tickers válidos incluso con respaldo para {portfolio_type}")
            st.error(f"No se encontraron suficientes tickers válidos para el portafolio {portfolio_type}, incluso con tickers de respaldo. Por favor, verifica tu conexión a internet o los tickers.")
            st.stop()
    return valid_tickers

def describe_allocation(weights, tickers):
    """Describe la asignación por tipo de activo"""
    if len(weights) == 0 or len(tickers) == 0 or all(w == 0 for w in weights):
        return "No se pudo calcular la asignación por tipo de activo debido a datos insuficientes."
    asset_types = [classify_asset(ticker) for ticker in tickers]
    df = pd.DataFrame({"Ticker": tickers, "Peso": weights, "Tipo": asset_types})
    grouped = df.groupby("Tipo")["Peso"].sum().reset_index()
    allocation_desc = ", ".join([f"{row['Tipo']}: {row['Peso']*100:.2f}%" for _, row in grouped.iterrows()])
    return f"Asignación por tipo de activo: {allocation_desc}"

def suggest_investment_amounts(profile):
    """Sugiere cantidades en MXN para invertir según el perfil"""
    suggestions = {
        "Conservador": 50000,
        "Moderado": 100000,
        "Balanceado": 150000,
        "Crecimiento": 200000,
        "Oportunidad": 300000
    }
    return suggestions.get(profile, 100000)

def generate_summary_with_claude(profile, tickers_basic, tickers_premier, scores_basic, scores_premier, max_retries=3):
    """Genera un resumen con la API de Anthropic (Claude)."""
    if scores_basic.empty or scores_premier.empty:
        logger.warning("Métricas vacías, usando resumen genérico")
        return f"""
        Tu perfil de riesgo es {profile} (puntaje: {total_score}). 
        **Portafolio Básico**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}. 
        Estos activos fueron seleccionados por su {'estabilidad y menor riesgo' if profile in ['Conservador', 'Moderado'] else 'equilibrio entre riesgo y retorno' if profile == 'Balanceado' else 'potencial de crecimiento'}.
        **Portafolio Premier**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}. 
        Este portafolio busca {'mayor seguridad' if profile in ['Conservador', 'Moderado'] else 'crecimiento moderado' if profile == 'Balanceado' else 'alto crecimiento con mayor riesgo'}.
        """

    prompt = f"""
    Eres Claude, un asesor financiero experto creado por Anthropic. Explica por qué los siguientes tickers fueron seleccionados para un inversionista con perfil de riesgo '{profile}' (puntaje: {total_score}):
    - Portafolio Básico: {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}.
    - Portafolio Premier: {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}.
    Basándote en las métricas de evaluación:
    - Portafolio Básico: {scores_basic.to_dict(orient='records')}.
    - Portafolio Premier: {scores_premier.to_dict(orient='records')}.
    Proporciona un resumen claro y conciso en 3-4 oraciones por portafolio, explicando cómo los tickers se alinean con el perfil de riesgo.
    """
    for attempt in range(max_retries):
        try:
            response = client_anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                system="Eres Claude, un asesor financiero experto creado por Anthropic.",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.content[0].text.strip()
            logger.info("Resumen generado exitosamente con Claude")
            return summary
        except APIError as e:
            if "429" in str(e):
                logger.warning(f"Límite de tasa alcanzado en Anthropic, reintentando en {2 ** attempt + random.uniform(0, 1)} segundos...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                else:
                    logger.error(f"Error persistente 429 en Anthropic: {str(e)}")
                    return f"""
                    Tu perfil de riesgo es {profile} (puntaje: {total_score}). 
                    **Portafolio Básico**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}. 
                    **Portafolio Premier**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}.
                    No se pudo generar un resumen detallado debido a problemas con la API.
                    """
            else:
                logger.error(f"Error en la solicitud a Anthropic: {str(e)}")
                return f"""
                Tu perfil de riesgo es {profile} (puntaje: {total_score}). 
                **Portafolio Básico**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}. 
                **Portafolio Premier**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}.
                No se pudo generar un resumen detallado debido a problemas con la API.
                """
        except Exception as e:
            logger.error(f"Error interpretando respuesta de Anthropic: {str(e)}")
            return f"""
            Tu perfil de riesgo es {profile} (puntaje: {total_score}). 
            **Portafolio Básico**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}. 
            **Portafolio Premier**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}.
            No se pudo generar un resumen detallado debido a problemas con la API.
            """
    return f"""
    Tu perfil de riesgo es {profile} (puntaje: {total_score}). 
    **Portafolio Básico**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_basic])}. 
    **Portafolio Premier**: Incluye {', '.join([ticker_to_name.get(t, t) for t in tickers_premier])}.
    No se pudo generar un resumen detallado debido a problemas con la API.
    """

# Botón para calcular portafolios
if st.button("Calcular Portafolios"):
    st.info("El proceso para cargar tus portafolios puede tardar un momento. Por favor, espera...")

    # Prueba de conexión inicial
    test_ticker = "SPY"
    test_data = None
    for attempt in range(3):
        try:
            test_data = Ticker(test_ticker).history(period="1mo", interval="1d")
            break
        except Exception as e:
            logger.error(f"Error al probar conexión con Yahoo Finance (intento {attempt + 1}/3): {str(e)}")
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    if test_data is None or test_data.empty or 'adjclose' not in test_data.columns:
        st.error("No se puede conectar a Yahoo Finance. Verifica tu conexión a internet o intenta más tarde.")
        st.stop()

    with st.spinner("Obteniendo recomendaciones de Claude..."):
        tickers_basic = recomendar_activos_con_claude(profile, "basic", currency)
        tickers_premier = recomendar_activos_con_claude(profile, "premier", currency)

        # Validar tickers
        combined_tickers = list(set(tickers_basic + tickers_premier))
        valid_tickers = validate_tickers(combined_tickers)
        tickers_basic = [t for t in tickers_basic if t in valid_tickers]
        tickers_premier = [t for t in tickers_premier if t in valid_tickers]

        # Filtrar por moneda
        filtered_tickers_basic = filter_tickers_by_currency(tickers_basic, currency)
        filtered_tickers_premier = filter_tickers_by_currency(tickers_premier, currency)

        # Fallback a tickers predeterminados
        if not filtered_tickers_basic:
            default_basic = [t for t in portafolio_basico if t in validate_tickers(portafolio_basico)]
            filtered_tickers_basic = filter_tickers_by_currency(default_basic, currency)
        if not filtered_tickers_premier:
            default_premier = [t for t in portafolio_premier if t in validate_tickers(portafolio_premier)]
            filtered_tickers_premier = filter_tickers_by_currency(default_premier, currency)

        # Asegurar que los tickers tengan datos históricos válidos
        filtered_tickers_basic = ensure_valid_tickers(filtered_tickers_basic, "basic")
        filtered_tickers_premier = ensure_valid_tickers(filtered_tickers_premier, "premier")

        st.write("Tickers Básicos Filtrados:", filtered_tickers_basic)
        st.write("Tickers Premier Filtrados:", filtered_tickers_premier)

        # Obtener datos históricos directamente de yahooquery
        benchmark_ticker = "SPY"
        with st.spinner("Descargando datos históricos desde yahooquery..."):
            prices_basic = fetch_historical_data(filtered_tickers_basic)
            prices_premier = fetch_historical_data(filtered_tickers_premier)
            prices_benchmark = fetch_historical_data([benchmark_ticker])

        # Depuración adicional
        if prices_basic.empty or prices_premier.empty:
            logger.warning(f"Datos vacíos - Básico: {len(prices_basic)}, Premier: {len(prices_premier)}")
            st.warning("Datos insuficientes detectados. Se intentarán tickers de respaldo.")

        # Calcular retornos
        returns_basic = calculate_returns(prices_basic)
        returns_premier = calculate_returns(prices_premier)

        # Definir weights_criteria
        weights_criteria = {
            "Conservador": {"Rentabilidad": 0.2, "Riesgo": 0.5, "Diversificación": 0.1, "Liquidez": 0.1, "Alineación": 0.1},
            "Moderado": {"Rentabilidad": 0.3, "Riesgo": 0.3, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
            "Balanceado": {"Rentabilidad": 0.35, "Riesgo": 0.25, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
            "Crecimiento": {"Rentabilidad": 0.4, "Riesgo": 0.2, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1},
            "Oportunidad": {"Rentabilidad": 0.45, "Riesgo": 0.15, "Diversificación": 0.2, "Liquidez": 0.1, "Alineación": 0.1}
        }[profile]

        if returns_basic.empty or returns_premier.empty:
            st.error("No se pudieron calcular los retornos debido a datos insuficientes. Revisa los logs para más detalles.")
            weights_basic = np.ones(len(filtered_tickers_basic)) / len(filtered_tickers_basic) if len(filtered_tickers_basic) > 0 else np.array([])
            weights_premier = np.ones(len(filtered_tickers_premier)) / len(filtered_tickers_premier) if len(filtered_tickers_premier) > 0 else np.array([])
            scores_basic = pd.DataFrame()
            scores_premier = pd.DataFrame()
        else:
            risk_free_rate = get_risk_free_rate()
            weights_basic = optimize_portfolio(returns_basic, risk_free_rate)
            weights_premier = optimize_portfolio(returns_premier, risk_free_rate)
            weights_basic = adjust_weights(weights_basic, filtered_tickers_basic, profile)
            weights_premier = adjust_weights(weights_premier, filtered_tickers_premier, profile)

            def calculate_criteria_scores(tickers, weights, prices):
                scores = []
                for ticker, weight in zip(tickers, weights):
                    if ticker not in prices.columns or weight == 0:
                        continue
                    data = prices[ticker].dropna()
                    if len(data) < 2:
                        logger.warning(f"Datos insuficientes para {ticker}")
                        continue
                    initial_price = data.iloc[0]
                    final_price = data.iloc[-1]
                    num_years = min(2, len(data) / 252)
                    cagr = ((final_price / initial_price) ** (1 / num_years) - 1) * 100 if initial_price > 0 else 0
                    rentabilidad = min(max((cagr + 10) / 2, 1), 10) if not np.isnan(cagr) else 1.0
                    returns = data.pct_change().dropna()
                    vol = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
                    riesgo = min(max((15 - vol) / 1.5, 1), 10) if not np.isnan(vol) else 8.0
                    asset_type = classify_asset(ticker)
                    diversificacion = 8 if asset_type == "Fondo de Inversión" else 5
                    liquidez = 5
                    alineacion = 8 if (profile in ["Conservador", "Moderado"] and asset_type == "Bono Gubernamental") or \
                                     (profile == "Balanceado" and asset_type in ["Fondo de Inversión", "Acción"]) or \
                                     (profile in ["Crecimiento", "Oportunidad"] and asset_type == "Acción") else 5
                    scores.append({
                        "Ticker": ticker_to_name.get(ticker, ticker),
                        "Rentabilidad": rentabilidad,
                        "Riesgo": riesgo,
                        "Diversificación": diversificacion,
                        "Liquidez": liquidez,
                        "Alineación": alineacion
                    })
                return pd.DataFrame(scores)

            scores_basic = calculate_criteria_scores(filtered_tickers_basic, weights_basic, prices_basic)
            scores_premier = calculate_criteria_scores(filtered_tickers_premier, weights_premier, prices_premier)

            for df in [scores_basic, scores_premier]:
                if not df.empty:
                    df["Puntuación Total"] = sum(df[crit] * weight for crit, weight in weights_criteria.items())

        # Gráficos usando visuals.py
        allocation_desc_basic = describe_allocation(weights_basic, filtered_tickers_basic)
        allocation_desc_premier = describe_allocation(weights_premier, filtered_tickers_premier)

        fig_pie_basic = visuals.plot_pie_chart(weights_basic, [ticker_to_name.get(t, t) for t in filtered_tickers_basic])
        fig_pie_premier = visuals.plot_pie_chart(weights_premier, [ticker_to_name.get(t, t) for t in filtered_tickers_premier])
        fig_line_basic = visuals.plot_historical_performance(prices_basic, weights_basic, prices_benchmark, benchmark_ticker)
        fig_line_premier = visuals.plot_historical_performance(prices_premier, weights_premier, prices_benchmark, benchmark_ticker)

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
            'allocation_desc_basic': allocation_desc_basic,
            'allocation_desc_premier': allocation_desc_premier,
            'weights_criteria': weights_criteria
        }

        # Generar resumen con Claude
        with st.spinner("Generando resumen con Claude..."):
            summary = generate_summary_with_claude(profile, filtered_tickers_basic, filtered_tickers_premier, scores_basic, scores_premier)
            st.session_state['report_data']['summary'] = summary

        # Mostrar resultados
        st.subheader("Portafolio Básico")
        st.write({ticker_to_name.get(t, t): f"{w*100:.2f}%" for t, w in zip(filtered_tickers_basic, weights_basic)})
        st.plotly_chart(fig_pie_basic, key="pie_chart_basic")
        st.plotly_chart(fig_line_basic, key="line_chart_basic")

        st.subheader("Portafolio Premier")
        st.write({ticker_to_name.get(t, t): f"{w*100:.2f}%" for t, w in zip(filtered_tickers_premier, weights_premier)})
        st.plotly_chart(fig_pie_premier, key="pie_chart_premier")
        st.plotly_chart(fig_line_premier, key="line_chart_premier")

        # Resumen
        st.subheader("Resumen de Resultados")
        st.write(summary)

        # Generar y ofrecer descarga del PDF
        def generate_pdf_report(report_data, risk_tolerance, investment_horizon, financial_objective,
                               financial_situation, market_knowledge, liquidity_need, age):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            profile = report_data.get('profile')
            total_score = report_data.get('score')
            tickers_basic = report_data.get('tickers_basic', [])
            tickers_premier = report_data.get('tickers_premier', [])
            weights_basic = report_data.get('weights_basic', [])
            weights_premier = report_data.get('weights_premier', [])
            scores_basic = report_data.get('scores_basic', pd.DataFrame())
            scores_premier = report_data.get('scores_premier', pd.DataFrame())
            allocation_desc_basic = report_data.get('allocation_desc_basic', "No disponible")
            allocation_desc_premier = report_data.get('allocation_desc_premier', "No disponible")
            summary = report_data.get('summary', '')
            weights_criteria = report_data.get('weights_criteria', {})

            try:
                elements.append(Image("LogoAllianz.jpeg", width=100, height=50))
            except Exception as e:
                logger.warning(f"No se pudo añadir el logo 'LogoAllianz.jpeg' al PDF: {str(e)}. Continuando sin logo.")
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Reporte de Perfil de Riesgo y Portafolios", styles['Heading1']))
            elements.append(Paragraph(f"Perfil de Riesgo: {profile} (Puntaje: {total_score})", styles['Heading2']))
            elements.append(Spacer(1, 12))

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

            suggested_amount = suggest_investment_amounts(profile)
            elements.append(Paragraph(f"Cantidad sugerida para invertir: {suggested_amount:,.2f} MXN", styles['Heading3']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Portafolio Básico", styles['Heading2']))
            basic_data = [["Ticker", "Nombre", "Peso (%)", "Monto (MXN)"]] + [
                [ticker, ticker_to_name.get(ticker, ticker), f"{weight*100:.2f}", f"{weight * suggested_amount:,.2f}"]
                for ticker, weight in zip(tickers_basic, weights_basic)
            ]
            basic_table = Table(basic_data, colWidths=[100, 200, 100, 100])
            basic_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            elements.append(basic_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Asignación por Tipo de Activo - Portafolio Básico", styles['Heading3']))
            elements.append(Paragraph(allocation_desc_basic, styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Comparativo Criterios - Portafolio Básico", styles['Heading3']))
            if not scores_basic.empty:
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
            else:
                elements.append(Paragraph("No hay datos de criterios disponibles para el Portafolio Básico.", styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Portafolio Premier", styles['Heading2']))
            premier_data = [["Ticker", "Nombre", "Peso (%)", "Monto (MXN)"]] + [
                [ticker, ticker_to_name.get(ticker, ticker), f"{weight*100:.2f}", f"{weight * suggested_amount:,.2f}"]
                for ticker, weight in zip(tickers_premier, weights_premier)
            ]
            premier_table = Table(premier_data, colWidths=[100, 200, 100, 100])
            premier_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(128/255, 179/255, 135/255)),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            elements.append(premier_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Asignación por Tipo de Activo - Portafolio Premier", styles['Heading3']))
            elements.append(Paragraph(allocation_desc_premier, styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Comparativo Criterios - Portafolio Premier", styles['Heading3']))
            if not scores_premier.empty:
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
            else:
                elements.append(Paragraph("No hay datos de criterios disponibles para el Portafolio Premier.", styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Resumen de Resultados", styles['Heading2']))
            elements.append(Paragraph(summary, styles['BodyText']))
            elements.append(Spacer(1, 12))

            doc.build(elements)
            pdf_buffer.seek(0)
            return pdf_buffer

        with st.spinner("Generando informe PDF..."):
            if 'report_data' in st.session_state:
                pdf_buffer = generate_pdf_report(
                    st.session_state['report_data'],
                    risk_tolerance,
                    investment_horizon,
                    financial_objective,
                    financial_situation,
                    market_knowledge,
                    liquidity_need,
                    age
                )
                st.download_button(
                    label="Descargar Informe PDF",
                    data=pdf_buffer,
                    file_name="reporte_portafolio.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("No se encontraron datos para generar el informe. Por favor, calcula los portafolios primero.")