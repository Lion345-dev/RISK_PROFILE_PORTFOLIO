import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def plot_pie_chart(weights, labels):
    """Genera un gráfico de pastel para la asignación de activos"""
    fig = px.pie(values=weights, names=labels, title="Asignación de Activos",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    return fig

def plot_historical_performance(prices, weights, benchmark_prices, benchmark_ticker="SPY", period="2y"):
    """Genera un gráfico de rendimiento histórico con normalización y comparación con benchmark"""
    if not isinstance(prices, pd.DataFrame) or prices.empty or all(w == 0 for w in weights):
        return px.line(x=[0], y=[0], title="Rendimiento Histórico (Datos Insuficientes)")

    # Calcular valor del portafolio normalizado
    portfolio_value = (prices * weights).sum(axis=1)
    portfolio_normalized = portfolio_value / portfolio_value.iloc[0] * 100

    # Normalizar benchmark
    if not isinstance(benchmark_prices, pd.DataFrame) or benchmark_prices.empty or benchmark_ticker not in benchmark_prices.columns:
        return px.line(x=portfolio_normalized.index, y=portfolio_normalized, title="Rendimiento Histórico (Sin Benchmark)")
    benchmark = benchmark_prices[benchmark_ticker].reindex(portfolio_normalized.index, method="ffill")
    benchmark_normalized = benchmark / benchmark.iloc[0] * 100

    # Crear figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_normalized.index, y=portfolio_normalized, name="Portafolio", line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=benchmark_normalized.index, y=benchmark_normalized, name=benchmark_ticker, line=dict(color='#ff7f0e')))
    fig.update_layout(
        title="Rendimiento Histórico (2 años)",
        xaxis_title="Fecha",
        yaxis_title="Valor Normalizado (%)",
        template="plotly_white",
        height=400
    )
    return fig

def generate_pdf_report(profile, portfolio_basic, portfolio_premier):
    """Genera un informe PDF con el perfil y los portafolios"""
    c = canvas.Canvas("report.pdf", pagesize=letter)
    c.drawString(100, 750, f"Perfil de Riesgo: {profile}")
    c.drawString(100, 730, "Portafolio Básico:")
    y_position = 710
    for asset, weight in portfolio_basic.items():
        c.drawString(120, y_position, f"{asset}: {weight:.2%}")
        y_position -= 20
    c.drawString(100, y_position, "Portafolio Premier:")
    y_position -= 20
    for asset, weight in portfolio_premier.items():
        c.drawString(120, y_position, f"{asset}: {weight:.2%}")
        y_position -= 20
    c.save()