import numpy as np
from scipy.optimize import minimize

def calculate_risk_profile(responses):
    score = sum(responses)
    if score <= 5:
        return "Conservador", score
    elif score <= 10:
        return "Moderado", score
    else:
        return "Agresivo", score

def optimize_portfolio(returns, risk_free_rate):
    if returns.empty:
        return np.array([])
    
    num_assets = len(returns.columns)
    if num_assets == 0:
        return np.array([])

    # Calcular retornos esperados y matriz de covarianza
    expected_returns = returns.mean() * 252  # Anualizar retornos
    cov_matrix = returns.cov() * 252  # Anualizar covarianza

    # Función objetivo: Minimizar la volatilidad del portafolio
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Restricciones
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma de pesos = 1
        {'type': 'ineq', 'fun': lambda x: x},  # Pesos >= 0
    )

    # Límites para los pesos (0 a 1 para cada activo)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Suposición inicial: Distribución uniforme
    initial_guess = num_assets * [1. / num_assets]

    # Optimización
    result = minimize(
        portfolio_volatility,
        initial_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x
    else:
        return np.zeros(num_assets)

def sharpe_ratio_maximization(returns, risk_free_rate):
    if returns.empty:
        return np.array([])
    
    num_assets = len(returns.columns)
    if num_assets == 0:
        return np.array([])

    # Calcular retornos esperados y matriz de covarianza
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Función objetivo: Maximizar el ratio de Sharpe (minimizamos el negativo)
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    # Restricciones
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x},
    )

    # Límites
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Suposición inicial
    initial_guess = num_assets * [1. / num_assets]

    # Optimización
    result = minimize(
        neg_sharpe_ratio,
        initial_guess,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x
    else:
        return np.zeros(num_assets)