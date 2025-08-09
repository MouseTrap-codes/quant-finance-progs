import os
from typing import Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandera.pandas as pa
import plotly.graph_objects as go
import yfinance as yf

jax.config.update("jax_enable_x64", True)


def create_yfinance_schema(ticker: str) -> pa.DataFrameSchema:
    schema = pa.DataFrameSchema(
        {
            "Close": pa.Column(pa.Float64),
            "High": pa.Column(pa.Float64),
            "Low": pa.Column(pa.Float64),
            "Open": pa.Column(pa.Float64),
            "Volume": pa.Column(pa.Int64, nullable=True),
        },
        index=pa.Index(pa.DateTime),
        strict=True,
        coerce=True,
    )

    return schema


def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    schema = create_yfinance_schema(ticker)

    os.makedirs("./data", exist_ok=True)
    path_name = f"./data/{ticker}-{start}to{end}.parquet"

    df: pd.DataFrame
    if os.path.exists(path_name):
        print("Loading from cache")
        df = pd.read_parquet(path_name)
    else:
        df = cast(
            pd.DataFrame,
            yf.download(ticker, start=start, end=end, multi_level_index=False),
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # flatten index

        df.index = pd.to_datetime(df.index).tz_localize(None)
        if df.empty:
            raise ValueError(
                f"No data returned for {ticker} between {start} and {end}."
            )
        df = schema.validate(df)
        df.to_parquet(path_name)

    return df


def load_prices_multi(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        df = load_yfinance(ticker=t, start=start, end=end)
        dfs.append(df[["Close"]].rename(columns={"Close": t}))
    prices = pd.concat(dfs, axis=1).dropna()
    return prices


def annualized_mu_cov(
    prices: pd.DataFrame, periods_per_year: int = 252
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    returns = (prices / prices.shift(1)).dropna()
    log_returns = np.log(returns)
    mu = jnp.asarray(log_returns.mean(axis=0) * periods_per_year)
    Sigma = jnp.asarray(np.cov(log_returns, rowvar=False) * periods_per_year)
    Sigma += 1e-8 * jnp.eye(Sigma.shape[0])

    return mu, Sigma


def efficient_frontier(
    mu: jnp.ndarray, Sigma: jnp.ndarray, n_points: int = 60
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = mu.shape[0]  # num assets
    one = jnp.ones(n)
    invS = jnp.linalg.inv(Sigma)

    A = one @ invS @ one
    B = one @ invS @ mu
    C = mu @ invS @ mu
    D = A * C - B * B

    r_min, r_max = mu.min(), mu.max()
    target = jnp.linspace(r_min, r_max, n_points)

    def w_for_r(r: jnp.ndarray) -> jnp.ndarray:
        lam = (C - B * r) / D
        gam = (A * r - B) / D
        return invS @ (lam * one + gam * mu)

    W = jax.vmap(w_for_r)(target)

    volatilities = jnp.sqrt(jnp.einsum("ij,jk,ik->i", W, Sigma, W))

    return target, volatilities, W


def plot_and_save_frontier(
    tickers: list[str],
    rets: jnp.ndarray,
    vols: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    filename: str = "efficient_frontier.html",
) -> None:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.asarray(vols),
            y=np.asarray(rets),
            mode="lines",
            name="Efficient Frontier",
        )
    )

    asset_vols = np.sqrt(np.diag(np.asarray(Sigma)))
    asset_rets = np.asarray(mu)

    fig.add_trace(
        go.Scatter(
            x=asset_vols,
            y=asset_rets,
            mode="markers+text",
            text=tickers,
            textposition="top center",
            name="Assets",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        template="plotly_white",
    )

    fig.write_html(filename, include_plotlyjs="cdn")
    print(f"Saved frontier to {filename}")


def main() -> None:
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    start, end = "2019-01-01", "2025-08-01"

    prices = load_prices_multi(tickers, start, end)
    mu, Sigma = annualized_mu_cov(prices)

    rets, vols, W = efficient_frontier(mu, Sigma, n_points=80)
    plot_and_save_frontier(tickers, rets, vols, mu, Sigma)


main()
