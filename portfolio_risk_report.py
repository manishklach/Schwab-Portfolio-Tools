import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from portfolio_core import (
    active_option_positions,
    default_csv_path,
    estimate_equity_equivalent_exposure,
    load_schwab_holdings,
)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(__import__("math").erf)(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def bs_d1(spot, strike, t, r, sigma):
    spot = np.maximum(np.asarray(spot, dtype=float), 1e-9)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def bs_delta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(np.asarray(opt_type) == "C", call_delta, put_delta)


def bs_gamma_vec(spot, strike, t, r, sigma):
    d1 = bs_d1(spot, strike, t, r, sigma)
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
    return pdf_d1 / (np.maximum(np.asarray(spot, dtype=float), 1e-9) * np.maximum(np.asarray(sigma, dtype=float), 1e-6) * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))


def bs_vega_vec(spot, strike, t, r, sigma):
    d1 = bs_d1(spot, strike, t, r, sigma)
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.asarray(spot, dtype=float) * pdf_d1 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9))


def build_option_risk_frame(df: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    options = active_option_positions(df)
    if options.empty:
        return options

    options = options.copy()
    options["Spot Proxy"] = np.where(
        options["Strike Price"].fillna(0.0) > 0,
        options["Strike Price"].astype(float),
        np.maximum(options["Price Numeric"].astype(float) * 100.0, 1.0),
    )
    options["T"] = np.maximum(options["Days To Expiry"].fillna(1).astype(float), 1.0) / 365.0
    price_ratio = options["Price Numeric"].astype(float) / np.maximum(options["Spot Proxy"].astype(float), 1.0)
    options["IV Proxy"] = np.clip(
        np.where(options["Price Numeric"].astype(float) > 0, price_ratio * 4.0, 0.55),
        0.10,
        2.0,
    )

    spot = options["Spot Proxy"].to_numpy(dtype=float)
    strike = options["Strike Price"].to_numpy(dtype=float)
    t = options["T"].to_numpy(dtype=float)
    sigma = options["IV Proxy"].to_numpy(dtype=float)
    opt_type = options["Opt Type"].to_numpy(dtype=str)

    options["Delta"] = bs_delta_vec(spot, strike, t, risk_free_rate, sigma, opt_type)
    options["Gamma"] = bs_gamma_vec(spot, strike, t, risk_free_rate, sigma)
    options["Vega"] = bs_vega_vec(spot, strike, t, risk_free_rate, sigma)
    options["Delta Dollars"] = options["Qty"] * options["Multiplier"] * options["Delta"] * options["Spot Proxy"]
    options["Gamma Dollars 1%"] = options["Qty"] * options["Multiplier"] * options["Gamma"] * (options["Spot Proxy"] * 0.01) ** 2
    options["Vega Dollars 1 Vol"] = options["Qty"] * options["Multiplier"] * options["Vega"] * 0.01
    return options


def print_section(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def main():
    parser = argparse.ArgumentParser(description="Portfolio summary plus approximate option risk and stress report.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD (default: today in local system date)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate as decimal (default: 0.04)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)
    df = load_schwab_holdings(csv_path, as_of=as_of)
    df["Equity Equivalent Exposure"] = estimate_equity_equivalent_exposure(df)
    options = build_option_risk_frame(df, args.r)

    nav = df["Market Value Numeric"].sum()
    long_mv = df.loc[df["Market Value Numeric"] > 0, "Market Value Numeric"].sum()
    short_mv = df.loc[df["Market Value Numeric"] < 0, "Market Value Numeric"].sum()
    day_change = df["Day Change Numeric"].sum()
    realized_unknown = int((df["Market Value Numeric"] == 0).sum())
    expired_count = int(df["Is Expired"].sum())

    print("Portfolio Risk Report")
    print(f"File: {csv_path}")
    print(f"As-Of Date: {as_of}")

    print_section("Topline")
    print(f"NAV / Market Value: ${nav:,.2f}")
    print(f"Gross Long Market Value: ${long_mv:,.2f}")
    print(f"Gross Short Market Value: ${abs(short_mv):,.2f}")
    print(f"Net Day Change: ${day_change:,.2f}")
    print(f"Expired Positions Excluded From Option Risk: {expired_count}")
    print(f"Zero-Market-Value Rows: {realized_unknown}")

    print_section("Asset Mix")
    asset_mix = (
        df.groupby("Asset Type Normalized", dropna=False)["Market Value Numeric"]
        .sum()
        .sort_values(ascending=False)
    )
    for asset_type, value in asset_mix.items():
        label = asset_type if str(asset_type).strip() else "Unknown"
        print(f"{label}: ${value:,.2f}")

    print_section("Largest Underlying Exposures")
    underlying_exposure = (
        df.groupby("Underlying", dropna=False)["Equity Equivalent Exposure"]
        .sum()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(15)
    )
    for underlying, value in underlying_exposure.items():
        print(f"{underlying}: ${value:,.2f}")

    print_section("Option Risk")
    if options.empty:
        print("No active option positions found.")
    else:
        print(f"Net Delta Dollars: ${options['Delta Dollars'].sum():,.2f}")
        print(f"Net Gamma Dollars (1% move): ${options['Gamma Dollars 1%'].sum():,.2f}")
        print(f"Net Vega Dollars (per 1 vol point): ${options['Vega Dollars 1 Vol'].sum():,.2f}")

        by_underlying = (
            options.groupby("Underlying")[["Delta Dollars", "Gamma Dollars 1%","Vega Dollars 1 Vol"]]
            .sum()
            .sort_values(by="Delta Dollars", key=lambda s: s.abs(), ascending=False)
            .head(15)
        )
        for underlying, row in by_underlying.iterrows():
            print(
                f"{underlying}: delta ${row['Delta Dollars']:,.2f} | "
                f"gamma1% ${row['Gamma Dollars 1%']:,.2f} | "
                f"vega ${row['Vega Dollars 1 Vol']:,.2f}"
            )

    print_section("Stress Scenarios")
    if options.empty:
        print("No active option positions found.")
    else:
        stress_rows = [
            ("Underlying -5%, IV +10pts", -0.05, 0.10),
            ("Underlying -10%, IV +20pts", -0.10, 0.20),
            ("Underlying +5%, IV -10pts", 0.05, -0.10),
            ("Underlying +10%, IV -20pts", 0.10, -0.20),
        ]
        for label, spot_move, vol_move in stress_rows:
            pnl = (
                options["Delta Dollars"] * spot_move
                + options["Gamma Dollars 1%"] * ((spot_move / 0.01) ** 2)
                + options["Vega Dollars 1 Vol"] * (vol_move * 100.0)
            ).sum()
            print(f"{label}: ${pnl:,.2f}")


if __name__ == "__main__":
    main()
