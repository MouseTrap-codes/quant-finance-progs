import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera.pandas as pa
import yfinance as yf


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
        df = pd.read_parquet(path_name)
    else:
        df = cast(
            pd.DataFrame,
            yf.download(ticker, start=start, end=end, multi_level_index=False),
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if df.empty:
            raise ValueError(
                f"No data returned for {ticker} between {start} and {end}."
            )
        df = schema.validate(df)
        df.to_parquet(path_name)

    return df


def plot_price_and_return_logs(ticker: str, start: str, end: str) -> None:
    df = load_yfinance(ticker, start, end)

    prices = df["Close"]
    log_returns = np.log(df["Close"] / df["Close"].shift(1))

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # first plot: Adjusted Closing prices
    ax1.plot(prices, label="Adj Close", color="blue")
    ax1.set_title(f"{ticker} Adjusted Close Price")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True)
    ax1.legend()

    # second plot: Log Returns
    ax2.plot(log_returns, label="Log Returns", color="orange")
    ax2.set_title(f"{ticker} Daily Log Returns")
    ax2.set_ylabel("Log Return")
    ax2.set_xlabel("Date")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def testMarketData() -> None:
    ticker = "AAPL"
    # test from start to end of 2024
    start = "2024-01-01"
    end = "2025-01-01"  # end is exclusive
    plot_price_and_return_logs(ticker, start, end)


testMarketData()
