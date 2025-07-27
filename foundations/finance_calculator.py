import argparse
import os
from datetime import date, datetime
from typing import NoReturn, Optional, cast

import calplot
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import yfinance as yf
from scipy.optimize import root_scalar

# loading market data given ticker, start, and end


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


# calculations and plots for market data


def compute_drawdown(log_returns: pd.Series) -> pd.Series:
    cum_returns = log_returns.cumsum()
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - rolling_max
    drawdown_pct = pd.Series(1 - np.exp(drawdown))

    return drawdown_pct


def plot_drawdown(log_returns: pd.Series) -> None:
    drawdown_pct = compute_drawdown(log_returns=log_returns)
    plt.figure(figsize=(12, 5))
    plt.plot(drawdown_pct, label="Drawdown (%) from log returns")
    plt.title("Drawdown (using log returns)")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.show()


def compute_rolling_volatility(window: int, log_returns: pd.Series) -> pd.Series:
    rolling_volatility = log_returns.rolling(window=window).std()
    return rolling_volatility


def plot_rolling_volatility(window: int, log_returns: pd.Series) -> None:
    rolling_volatility = compute_rolling_volatility(
        window=window, log_returns=log_returns
    )
    rolling_volatility = rolling_volatility * np.sqrt(252)
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_volatility)
    plt.title(f"{window}-Day Rolling Volatility (Annualized)")
    plt.ylabel("Standard Deviation")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()


def plot_heatmap_of_returns(log_returns: pd.Series) -> None:
    calplot.calplot(
        log_returns,
        cmap="RdYlGn",
        fillcolor="lightgray",
        linewidth=1,
        edgecolor="white",
        suptitle="Log Returns Heatmap",
        figsize=(16, 6),
        yearlabel_kws={"fontsize": 18},
        daylabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        dayticks=list(range(7)),
    )


def execute_market_data(args: argparse.Namespace) -> None:
    df = load_yfinance(ticker=args.ticker, start=args.start, end=args.end)
    log_returns = pd.Series(np.log(df["Close"] / df["Close"].shift(1)))
    log_returns.dropna(inplace=True)

    if args.mode == "volatility":
        plot_rolling_volatility(log_returns=log_returns, window=21)
    elif args.mode == "drawdown":
        plot_drawdown(log_returns=log_returns)
    elif args.mode == "returns":
        plot_heatmap_of_returns(log_returns=log_returns)


# npv and irr calculations
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


def execute_cash_flow(args: argparse.Namespace) -> Optional[float]:
    if args.cash_flows is None:
        return None
    cf = np.asarray(args.cash_flows, dtype=np.float64)
    if args.mode == "npv":
        return npv(rate=args.rate, cash_flows=cf)
    elif args.mode == "irr":
        return irr(cash_flows=cf)
    return None


def raise_invalid_date_error(date_string: str) -> NoReturn:
    raise argparse.ArgumentTypeError(
        f"Invalid date: '{date_string}'. Use YYYY-MM-DD format"
    )


def valid_date(date_string: str) -> date:
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        raise_invalid_date_error(date_string)


def validate_start_and_end(args: argparse.Namespace) -> None:
    if args.start >= args.end:
        raise ValueError("--start must be before --end")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Finance Calculator",
        description="A CLI that wraps NPV, IRR, 21-day rolling volatility plot, "
        "drawdown plot, and log returns heatmap for any cash flow list & ticker.",
    )

    # select mode
    parser.add_argument(
        "--mode",
        choices=["npv", "irr", "volatility", "drawdown", "returns"],
        required=True,
        help="Which calculation or plot to run.",
    )

    # input cash flow and rate
    parser.add_argument(
        "--cash-flows",
        nargs="+",
        type=float,
        help="List of cash flows (e.g. -1000 300 420 680)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        help="Discount rate for NPV calculation (e.g. 0.1 for 10%).",
    )

    # market data
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g. AAPL).")
    parser.add_argument(
        "--start", type=valid_date, help="Start data for yfinance data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=valid_date, help="End data for yfinance data (YYYY-MM-DD)."
    )

    args = parser.parse_args()

    # dispatch
    if args.mode == "npv":
        if args.cash_flows is None or args.rate is None:
            raise ValueError("NPV mode requires --cash-flows and --rate")
        print(f"Calculating NPV given {args.cash_flows} and {args.rate}...")
        execute_cash_flow(args=args)
    elif args.mode == "irr":
        if args.cash_flows is None:
            raise ValueError("IRR mode requires --cash-flows")
        print(f"Calculating IRR given {args.cash_flows}...")
        execute_cash_flow(args=args)
    elif args.mode == "drawdown":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Drawdown mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            f"Plotting drawdown for "
            f"{args.ticker} from {args.start} to {args.end}..."
        )
        execute_market_data(args=args)
    elif args.mode == "volatility":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Volatility mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            "Plotting 21-day rolling volatility for ",
            f"{args.ticker} from {args.start} to {args.end}...",
        )
        execute_market_data(args=args)
    elif args.mode == "returns":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Returns mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            "Plotting heatmap of log returns for "
            f"{args.ticker} from {args.start} to {args.end}..."
        )
        execute_market_data(args=args)
