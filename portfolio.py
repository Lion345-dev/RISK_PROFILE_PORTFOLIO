import numpy as np
from scipy.optimize import minimize

def calculate_risk_profile(responses):
    """Calcula el puntaje de riesgo basado en las respuestas del cuestionario"""
    score = sum(responses)
    if score <= 8:
        return "Conservador", score
    elif score <= 12:
        return "Moderado", score
    else:
        return "Agresivo", score

def optimize_portfolio(returns, risk_free_rate, target_return=None):
    """Optimiza el portafolio usando Markowitz"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def portfolio_return(weights):
        return np.dot(weights, mean_returns)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return:
        constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def sharpe_ratio_maximization(returns, risk_free_rate):
    """Maximiza el ratio de Sharpe"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    
    def neg_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (portfolio_return - risk_free_rate) / portfolio_volatility
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x