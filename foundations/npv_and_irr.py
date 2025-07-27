from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar


def npv(rate: float, cash_flows: npt.NDArray[np.float64]) -> float:
    time = np.arange(len(cash_flows))
    return float(np.sum(cash_flows / (1 + rate) ** time))


def irr(cash_flows: npt.NDArray[np.float64]) -> Optional[float]:
    if len(cash_flows) < 2:
        return None

    if np.all(cash_flows >= 0) or np.all(cash_flows <= 0):
        return None

    def f(r: float) -> float:
        return npv(r, cash_flows=cash_flows)

    try:
        solution = root_scalar(f, method="brentq", bracket=[-0.999, 50])
        return solution.root if solution.converged else None
    except Exception:
        return None
