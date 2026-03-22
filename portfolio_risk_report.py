"""Generate a consolidated portfolio summary with approximate option Greeks and stress scenarios."""

import argparse
from collections import defaultdict
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


def summarize_call_debit_spreads(options: pd.DataFrame) -> pd.DataFrame:
    spread_rows = []
    calls = options[options["Opt Type"] == "C"].copy()

    for (underlying, expiry), grp in calls.groupby(["Underlying", "Expiration"]):
        longs = []
        shorts = []
        for _, row in grp.sort_values("Strike Price").iterrows():
            qty = float(row["Qty"])
            leg = {
                "strike": float(row["Strike Price"]),
                "qty": abs(qty),
                "price": float(row["Price Numeric"]),
                "delta": float(row["Delta Dollars"]),
            }
            if qty > 0:
                longs.append(leg)
            elif qty < 0:
                shorts.append(leg)

        for short_leg in sorted(shorts, key=lambda leg: leg["strike"]):
            remaining = short_leg["qty"]
            candidates = [leg for leg in longs if leg["qty"] > 0 and leg["strike"] < short_leg["strike"]]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining <= 0:
                    break
                matched = min(remaining, long_leg["qty"])
                width = short_leg["strike"] - long_leg["strike"]
                spread_rows.append(
                    {
                        "Underlying": underlying,
                        "Expiration": expiry,
                        "Contracts": matched,
                        "Long Strike": long_leg["strike"],
                        "Short Strike": short_leg["strike"],
                        "Width": width,
                        "Max Spread Value": matched * 100.0 * width,
                        "Net Delta Dollars": matched
                        * (
                            long_leg["delta"] / max(long_leg["qty"], 1.0)
                            + short_leg["delta"] / max(short_leg["qty"], 1.0)
                        ),
                    }
                )
                remaining -= matched
                long_leg["qty"] -= matched

    if not spread_rows:
        return pd.DataFrame(
            columns=[
                "Underlying",
                "Contracts",
                "Max Spread Value",
                "Net Delta Dollars",
            ]
        )

    return (
        pd.DataFrame(spread_rows)
        .groupby("Underlying", as_index=False)[["Contracts", "Max Spread Value", "Net Delta Dollars"]]
        .sum()
        .sort_values("Max Spread Value", ascending=False)
    )


def summarize_uncovered_short_puts(options: pd.DataFrame) -> pd.DataFrame:
    puts = options[options["Opt Type"] == "P"].copy()
    uncovered_rows = []

    for (underlying, expiry), grp in puts.groupby(["Underlying", "Expiration"]):
        longs = []
        shorts = []
        for _, row in grp.sort_values("Strike Price", ascending=False).iterrows():
            qty = float(row["Qty"])
            if qty > 0:
                longs.append({"strike": float(row["Strike Price"]), "qty": qty})
            elif qty < 0:
                shorts.append(row)

        for short_row in shorts:
            remaining = abs(float(short_row["Qty"]))
            short_strike = float(short_row["Strike Price"])
            candidates = [leg for leg in longs if leg["qty"] > 0 and leg["strike"] < short_strike]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining <= 0:
                    break
                matched = min(remaining, long_leg["qty"])
                remaining -= matched
                long_leg["qty"] -= matched

            if remaining > 0:
                uncovered_rows.append(
                    {
                        "Underlying": underlying,
                        "Contracts": remaining,
                        "Assignment Notional": remaining * 100.0 * short_strike,
                    }
                )

    if not uncovered_rows:
        return pd.DataFrame(columns=["Underlying", "Contracts", "Assignment Notional"])

    return (
        pd.DataFrame(uncovered_rows)
        .groupby("Underlying", as_index=False)[["Contracts", "Assignment Notional"]]
        .sum()
        .sort_values("Assignment Notional", ascending=False)
    )


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
    call_spreads = summarize_call_debit_spreads(options) if not options.empty else pd.DataFrame()
    uncovered_puts = summarize_uncovered_short_puts(options) if not options.empty else pd.DataFrame()

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

    print_section("Net Delta Exposure By Underlying")
    if options.empty:
        print("No active option positions found.")
    else:
        delta_exposure = (
            options.groupby("Underlying")["Delta Dollars"]
            .sum()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(15)
        )
        for underlying, value in delta_exposure.items():
            print(f"{underlying}: ${value:,.2f}")

    print_section("Bullish Call Debit Spreads")
    if call_spreads.empty:
        print("No call debit spreads recognized.")
    else:
        for _, row in call_spreads.head(15).iterrows():
            print(
                f"{row['Underlying']}: {row['Contracts']:g} contracts | "
                f"max spread value ${row['Max Spread Value']:,.2f} | "
                f"net delta ${row['Net Delta Dollars']:,.2f}"
            )

    print_section("Uncovered Short Puts")
    if uncovered_puts.empty:
        print("No uncovered short puts recognized.")
    else:
        for _, row in uncovered_puts.head(15).iterrows():
            print(
                f"{row['Underlying']}: {row['Contracts']:g} contracts | "
                f"assignment notional ${row['Assignment Notional']:,.2f}"
            )

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
