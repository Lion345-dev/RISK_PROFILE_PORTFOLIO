import streamlit as st
from data import get_etf_sector_data, get_historical_data, calculate_returns, get_risk_free_rate
from portfolio import calculate_risk_profile, optimize_portfolio, sharpe_ratio_maximization
from visuals import plot_pie_chart, plot_historical_performance, generate_pdf_report

# Configuración de la página
st.set_page_config(page_title="Perfil de Riesgo y Recomendación de Portafolios", layout="wide")

# Logo y título
st.image("LogoAllianz.jpeg", width=150)
st.title("Simulador de Perfil de Riesgo y Portafolios")

# Cuestionario de Perfilamiento
st.header("Cuestionario de Perfilamiento")
education = st.selectbox("1. Nivel de Estudios", ["Primaria/Secundaria", "Preparatoria", "Licenciatura", "Posgrado"], index=0)
age = st.selectbox("2. Rango de Edad", ["Menos de 30 años", "30-50 años", "Más de 50 años"], index=0)
investment_horizon = st.selectbox("3. Horizonte de Inversión", ["Menos de 1 año", "1 a 5 años", "Más de 5 años"], index=0)
financial_knowledge = st.multiselect("4. Conocimiento Financiero", ["Ninguno", "Básico", "Intermedio", "Avanzado"])
risk_tolerance = st.selectbox("5. Tolerancia al Riesgo", ["Conservador", "Moderado", "Agresivo"], index=0)

# Mapear respuestas a puntos
education_points = {"Primaria/Secundaria": 1, "Preparatoria": 2, "Licenciatura": 3, "Posgrado": 4}[education]
age_points = {"Menos de 30 años": 3, "30-50 años": 2, "Más de 50 años": 1}[age]
horizon_points = {"Menos de 1 año": 1, "1 a 5 años": 2, "Más de 5 años": 3}[investment_horizon]
knowledge_points = sum([0 if "Ninguno" in financial_knowledge else 1 if "Básico" in financial_knowledge else 2 if "Intermedio" in financial_knowledge else 3])
tolerance_points = {"Conservador": 1, "Moderado": 2, "Agresivo": 3}[risk_tolerance]

responses = [education_points, age_points, horizon_points, knowledge_points, tolerance_points]
profile, score = calculate_risk_profile(responses)

st.write(f"Tu perfil de riesgo es: {profile} (Puntaje: {score})")

# Selección de moneda
currency = st.multiselect("Selecciona las monedas para tu portafolio", ["MXN", "USD", "EUR"], default=["MXN"])

# Botón para calcular portafolios
if st.button("Calcular Portafolios"):
    # Obtener datos (esto es un ejemplo, debes ajustar según los tickers reales)
    tickers_basic = ["CETETRAC", "NAFTRAC"]  # Reemplaza con tickers reales de Alternativas Básicas
    tickers_premier = ["ACWI", "EPP"]  # Reemplaza con tickers reales de Alternativas Premier
    
    prices_basic = get_historical_data(tickers_basic)
    returns_basic = calculate_returns(prices_basic)
    
    prices_premier = get_historical_data(tickers_premier)
    returns_premier = calculate_returns(prices_premier)
    
    risk_free_rate = get_risk_free_rate()
    
    # Optimizar portafolios
    weights_basic = optimize_portfolio(returns_basic, risk_free_rate)
    weights_premier = sharpe_ratio_maximization(returns_premier, risk_free_rate)
    
    # Mostrar resultados
    st.subheader("Portafolio Básico")
    st.write(weights_basic)
    fig_pie_basic = plot_pie_chart(weights_basic, tickers_basic)
    st.plotly_chart(fig_pie_basic)
    
    st.subheader("Portafolio Premier")
    st.write(weights_premier)
    fig_pie_premier = plot_pie_chart(weights_premier, tickers_premier)
    st.plotly_chart(fig_pie_premier)
    
    # Generar y descargar informe PDF
    generate_pdf_report(profile, dict(zip(tickers_basic, weights_basic)), dict(zip(tickers_premier, weights_premier)))
    with open("report.pdf", "rb") as file:
        st.download_button("Descargar Informe PDF", file, file_name="report.pdf")