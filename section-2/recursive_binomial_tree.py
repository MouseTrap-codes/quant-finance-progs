from typing import Optional

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


def compute_f(fu_next: float, fd_next: float, p: float, r: float, dt: float) -> float:
    return float(((p * fu_next) + ((1 - p) * fd_next)) * np.exp(-r * dt))


# construct the binomial tree
# binary tree
class Node:
    def __init__(self) -> None:
        self.asset_price: Optional[float] = None
        self.option_price: Optional[float] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None

    def set_option_price(self, option_price: float) -> None:
        self.option_price = option_price

    def set_asset_price(self, asset_price: float) -> None:
        self.asset_price = asset_price

    def get_option_price(self) -> Optional[float]:
        return self.option_price

    def get_asset_price(self) -> Optional[float]:
        return self.asset_price


def compute_asset_price(
    parent: Node,
    node: Node,
    sigma: float,
    dt: float,
    current_level: float,
    max_level: int,
) -> None:
    parent_price = parent.get_asset_price()
    if parent_price is None:
        raise ValueError("Parent node has no asset price set")

    if node is parent.left:
        node.set_asset_price(parent_price * compute_d(sigma, dt))
    elif node is parent.right:
        node.set_asset_price(parent_price * compute_u(sigma, dt))
    else:
        raise ValueError("Not a valid child node")

    if current_level == max_level:
        return

    current_level_new = current_level + 1

    node.left = Node()
    compute_asset_price(
        node,
        node.left,
        sigma,
        dt,
        current_level=current_level_new,
        max_level=max_level,
    )

    node.right = Node()
    compute_asset_price(
        node,
        node.right,
        sigma,
        dt,
        current_level=current_level_new,
        max_level=max_level,
    )


# recursively compute option prices
def compute_option_prices(
    node: Node,
    K: float,
    option_type: str,
    p: float,
    r: float,
    dt: float,
    is_american: bool = False,
) -> None:
    if not node:
        return

    # terminal node
    option_type = option_type.lower()
    if not node.left and not node.right:
        S_t = node.get_asset_price()
        if S_t is None:
            raise ValueError("Terminal node has no asset price")

        if option_type == "call":
            node.set_option_price(max(0.0, S_t - K))
        elif option_type == "put":
            node.set_option_price(max(K - S_t, 0.0))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return

    # postorder traversal
    if node.left:
        compute_option_prices(
            node.left, K, option_type, p, r, dt, is_american=is_american
        )
    if node.right:
        compute_option_prices(
            node.right, K, option_type, p, r, dt, is_american=is_american
        )

    # process current node
    if node.left and node.right:
        fd_next = node.left.get_option_price()
        fu_next = node.right.get_option_price()

        if fd_next is None or fu_next is None:
            raise ValueError("Child option prices not set before discounting.")

        cont = compute_f(fu_next=fu_next, fd_next=fd_next, p=p, r=r, dt=dt)

        if is_american:
            S_current = node.get_asset_price()
            if S_current is None:
                raise ValueError("Node has no asset price")
            intrinsic = (
                max(0.0, S_current - K)
                if option_type == "call"
                else max(0.0, K - S_current)
            )
            node.set_option_price(max(cont, intrinsic))
        else:
            node.set_option_price(cont)


def build_binomial_tree(S_t: float, sigma: float, dt: float, max_level: int) -> Node:
    root = Node()
    root.set_asset_price(S_t)

    root.left = Node()
    root.right = Node()

    compute_asset_price(
        root, root.left, sigma, dt, current_level=1, max_level=max_level
    )
    compute_asset_price(
        root, root.right, sigma, dt, current_level=1, max_level=max_level
    )

    return root


def recursive_binomial_tree(
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

    is_american = exercise_type.lower() == "american"

    root = build_binomial_tree(S_t, sigma, dt, max_level=N)
    compute_option_prices(root, K, option_type, p, r, dt, is_american=is_american)

    option_price = root.get_option_price()
    if option_price is None:
        raise ValueError("Option price was unable to be calculated.")

    return option_price
