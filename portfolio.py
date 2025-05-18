import numpy as np
from scipy.optimize import minimize
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def optimize_portfolio(returns, risk_free_rate):
    if returns.empty:
        logger.error("DataFrame de retornos vacío")
        return np.array([])
    
    num_assets = len(returns.columns)
    if num_assets == 0:
        logger.error("No hay activos en el DataFrame de retornos")
        return np.array([])

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x},
    )

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    try:
        result = minimize(
            portfolio_volatility,
            initial_guess,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            logger.info("Optimización de portafolio exitosa")
            return result.x
        else:
            logger.error("Optimización de portafolio fallida")
            return np.zeros(num_assets)
    except Exception as e:
        logger.error(f"Error en optimización de portafolio: {str(e)}")
        return np.zeros(num_assets)

def sharpe_ratio_maximization(returns, risk_free_rate):
    if returns.empty:
        logger.error("DataFrame de retornos vacío")
        return np.array([])
    
    num_assets = len(returns.columns)
    if num_assets == 0:
        logger.error("No hay activos en el DataFrame de retornos")
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
            logger.error("Maximización de Sharpe fallida")
            return np.zeros(num_assets)
    except Exception as e:
        logger.error(f"Error en maximización de Sharpe: {str(e)}")
        return np.zeros(num_assets)