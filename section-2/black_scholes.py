from typing import Dict

import numpy as np
from scipy.stats import norm


def black_scholes_with_greeks(
    S_t: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> Dict[str, float]:
    option = option_type.lower()

    if T <= 0.0 or sigma <= 0.0:
        price = max(0.0, S_t - K) if option == "call" else max(0.0, K - S_t)
        return {
            "Price": price,
            "Delta": 0.0,
            "Theta": 0.0,
            "Gamma": 0.0,
            "Vega": 0.0,
            "Rho": 0.0,
        }

    d1 = (np.log(S_t / K) + (r - q + 0.5 * sigma**2) * (T)) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S_t * sigma * np.sqrt(T))
    vega = S_t * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    if option == "call":
        price = S_t * np.exp(-q * T) * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            (-(S_t * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)))
            - (r * K * np.exp(-r * T) * norm.cdf(d2))
            + (q * S_t * np.exp(-q * T) * norm.cdf(d1))
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S_t * np.exp(-q * T) * norm.cdf(
            -d1
        )
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            (-(S_t * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)))
            + (r * K * np.exp(-r * T) * norm.cdf(-d2))
            - (q * S_t * np.exp(-q * T) * norm.cdf(-d1))
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Option type is invalid")

    theta = theta / 365  # per day
    rho = rho / 100  # per 1% change
    vega = vega / 100  # per 1% change

    return {
        "Price": price,
        "Delta": delta,
        "Theta": theta,
        "Gamma": gamma,
        "Vega": vega,
        "Rho": rho,
    }
