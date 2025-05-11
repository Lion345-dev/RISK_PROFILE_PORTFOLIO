import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def plot_pie_chart(weights, labels):
    """Genera un gráfico de pastel para la asignación de activos"""
    fig = px.pie(values=weights, names=labels, title="Asignación de Activos")
    return fig

def plot_historical_performance(prices):
    """Genera un gráfico de rendimiento histórico"""
    fig = px.line(prices, title="Rendimiento Histórico")
    return fig

def generate_pdf_report(profile, portfolio_basic, portfolio_premier):
    """Genera un informe PDF con el perfil y los portafolios"""
    c = canvas.Canvas("report.pdf", pagesize=letter)
    c.drawString(100, 750, f"Perfil de Riesgo: {profile}")
    c.drawString(100, 730, "Portafolio Básico:")
    for asset, weight in portfolio_basic.items():
        c.drawString(120, 710, f"{asset}: {weight:.2%}")
    c.drawString(100, 690, "Portafolio Premier:")
    for asset, weight in portfolio_premier.items():
        c.drawString(120, 670, f"{asset}: {weight:.2%}")
    c.save()