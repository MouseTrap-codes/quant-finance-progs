import argparse
from datetime import date, datetime
from typing import NoReturn


def raise_invalid_date_error(date_string: str) -> NoReturn:
    raise argparse.ArgumentTypeError(
        f"Invalid date: '{date_string}'. Use YYYY-MM-DD format"
    )


def valid_date(date_string: str) -> date:
    """Custom validation for argparse"""
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
    elif args.mode == "irr":
        if args.cash_flows is None:
            raise ValueError("IRR mode requires --cash-flows")
        print(f"Calculating IRR given {args.cash_flows}...")
    elif args.mode == "drawdown":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Drawdown mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            f"Plotting drawdown for "
            f"{args.ticker} from {args.start} to {args.end}..."
        )
    elif args.mode == "volatility":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Volatility mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            "Plotting 21-day rolling volatility for ",
            f"{args.ticker} from {args.start} to {args.end}...",
        )
    elif args.mode == "returns":
        if args.ticker is None or args.start is None or args.end is None:
            raise ValueError("Returns mode requires --ticker, --start, and --end")
        validate_start_and_end(args)
        print(
            "Plotting heatmap of log returns for "
            f"{args.ticker} from {args.start} to {args.end}..."
        )
