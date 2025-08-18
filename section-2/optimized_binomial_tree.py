from typing import Callable, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np


def compute_u(sigma: float, dt: float) -> float:
    return float(np.exp(sigma * np.sqrt(dt)))


def compute_d(sigma: float, dt: float) -> float:
    return float(1.0 / compute_u(sigma=sigma, dt=dt))


def compute_probability(
    u: float,
    d: float,
    r: float,
    dt: float,
    q: Optional[float] = None,
    r_f: Optional[float] = None,
    asset_type: str = "nondividend",
) -> float:
    if u < d:
        raise ValueError(f"Need u > d (got u={u}, d={d})")
    if u - d == 0:
        raise ZeroDivisionError("u - d == 0")

    asset_type = asset_type.lower()

    def nondividend() -> float:
        return float(np.exp(r * dt))

    def dividend() -> float:
        if q is None:
            raise ValueError("Need q")
        return float(np.exp((r - q) * dt))

    def currency() -> float:
        if r_f is None:
            raise ValueError("Foreign risk-free rate not specified.")
        return float(np.exp((r - r_f) * dt))

    def future() -> float:
        return 1.0

    computations = {
        "nondividend": nondividend,
        "dividend": dividend,
        "currency": currency,
        "future": future,
    }

    if asset_type not in computations:
        raise ValueError(f"Unknown asset type: '{asset_type}'")

    a: float = computations[asset_type]()

    p = (a - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. Check inputs."
        )

    return p


def _dp_kernel_py(
    S_t: float,
    K: float,
    N: int,
    u: float,
    d: float,
    p: float,
    disc: float,
    is_call: int,
    is_amer: int,
) -> jnp.ndarray:
    j = jnp.arange(N + 1, dtype=jnp.float64)

    # stable stock prices at maturity: S = S0 * d^N * (u/d)^j
    S = S_t * (d**N) * ((u / d) ** j)

    V_call = jnp.maximum(0.0, S - K)
    V_put = jnp.maximum(K - S, 0.0)
    V = jnp.where(is_call == 1, V_call, V_put)

    # like Bellman equations in RL!
    def bellman_like_equation(
        i: int, carry: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        V_layer, S_layer = carry
        k = N - i
        # continuation (for european)
        cont = disc * (p * V_layer[1 : k + 1] + (1.0 - p) * V_layer[:k])

        # move underlying prices back one layer (for american)
        S_prev = S_layer[:k] / d

        # intrinsic for current layer
        intrinsic = jnp.where(is_call == 1, S_prev - K, K - S_prev)
        intrinsic = jnp.maximum(intrinsic, 0.0)

        # american vs european
        new_V = jnp.where(is_amer == 1, jnp.maximum(cont, intrinsic), cont)

        # keep shapes invariant for jax to not bug out
        V_layer = V_layer.at[:k].set(new_V)
        S_layer = S_layer.at[:k].set(S_prev)

        return (V_layer, S_layer)

    V_final, _ = jax.lax.fori_loop(0, N, bellman_like_equation, (V, S))

    return V_final[0]


_dp_kernel: Callable[..., jnp.ndarray] = cast(
    Callable[..., jnp.ndarray], jax.jit(_dp_kernel_py)
)


# binomial tree using dp
def dp_binomial_tree(
    sigma: float,
    T: float,
    N: int,
    S_t: float,
    r: float,
    K: float,
    q: Optional[float] = None,
    r_f: Optional[float] = None,
    asset_type: str = "nondividend",
    exercise_type: str = "european",
    option_type: str = "call",
) -> float:
    if N <= 0:
        raise ValueError("Cannot have 0 or lower time periods")

    dt = T / N
    u = compute_u(sigma, dt)
    d = compute_d(sigma, dt)
    p = compute_probability(u=u, d=d, r=r, dt=dt, q=q, r_f=r_f, asset_type=asset_type)

    discount_rate = np.exp(-r * dt)

    is_call = 1 if option_type.lower() == "call" else 0
    is_amer = 1 if exercise_type.lower() == "american" else 0

    return float(
        _dp_kernel(
            float(S_t),
            float(K),
            int(N),
            float(u),
            float(d),
            float(p),
            float(discount_rate),
            int(is_call),
            int(is_amer),
        )
    )
