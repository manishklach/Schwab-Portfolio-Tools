import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOL_RE = re.compile(r'^\s*(\S+)\s+(\d{2}/\d{2}/\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$')


@dataclass
class FitParams:
    spot: float
    sigma: float
    used_points: int


def norm_cdf(x):
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x_arr / np.sqrt(2.0)))


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() in {'N/A', '-', ''}:
        return 0.0
    val_str = str(val).replace('$', '').replace('%', '').replace(',', '').strip()
    if '(' in val_str and ')' in val_str:
        val_str = '-' + val_str.replace('(', '').replace(')', '')
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def parse_symbol(symbol):
    match = SYMBOL_RE.match(str(symbol))
    if not match:
        return None
    ticker, expiry, strike, opt_type = match.groups()
    return ticker, expiry, float(strike), opt_type


def bs_price_vec(spot, strike, t, r, sigma, opt_type_arr):
    spot = np.maximum(spot, 1e-9)
    strike = np.maximum(strike, 1e-9)
    sigma = np.maximum(sigma, 1e-6)
    t = np.maximum(t, 1e-9)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    call_price = spot * norm_cdf(d1) - strike * np.exp(-r * t) * norm_cdf(d2)
    put_price = strike * np.exp(-r * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return np.where(opt_type_arr == 'C', call_price, put_price)


def bs_vega_vec(spot, strike, t, r, sigma):
    spot = np.maximum(spot, 1e-9)
    strike = np.maximum(strike, 1e-9)
    sigma = np.maximum(sigma, 1e-6)
    t = np.maximum(t, 1e-9)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 ** 2)
    return spot * pdf_d1 * sqrt_t


def fit_spot_sigma(group_df, r):
    if group_df.empty:
        return None

    strikes = group_df['Strike Price'].to_numpy(dtype=float)
    prices = group_df['Option Price'].to_numpy(dtype=float)
    opt_types = group_df['Opt Type'].to_numpy(dtype=str)
    t = float(group_df['T'].iloc[0])

    valid = (strikes > 0) & (prices > 0) & np.isfinite(strikes) & np.isfinite(prices)
    strikes = strikes[valid]
    prices = prices[valid]
    opt_types = opt_types[valid]

    if len(prices) < 2 or t <= 0:
        return None

    min_k = float(np.min(strikes))
    max_k = float(np.max(strikes))
    if min_k <= 0 or max_k <= 0:
        return None

    s_grid = np.linspace(min_k * 0.7, max_k * 1.3, 90)
    sigma_grid = np.linspace(0.08, 2.0, 90)

    best_err = float('inf')
    best_s = None
    best_sigma = None

    for s in s_grid:
        for sigma in sigma_grid:
            model_prices = bs_price_vec(s, strikes, t, r, sigma, opt_types)
            denom = np.maximum(prices, 0.25)
            err = np.mean(((model_prices - prices) / denom) ** 2)
            if err < best_err:
                best_err = err
                best_s = s
                best_sigma = sigma

    if best_s is None or best_sigma is None:
        return None

    return FitParams(spot=float(best_s), sigma=float(best_sigma), used_points=int(len(prices)))


def match_call_spreads_for_scenario(calls_df, option_delta_per_contract_col):
    total_spread_pnl = 0.0

    for (_, _), grp in calls_df.groupby(['Ticker', 'Expiration']):
        longs = []
        shorts = []
        for _, row in grp.iterrows():
            qty = float(row['Qty'])
            if qty > 0:
                longs.append({
                    'strike': float(row['Strike Price']),
                    'rem': qty,
                    'pnl_per_contract': float(row[option_delta_per_contract_col]),
                })
            elif qty < 0:
                shorts.append({
                    'strike': float(row['Strike Price']),
                    'rem': abs(qty),
                    'pnl_per_contract': -float(row[option_delta_per_contract_col]),
                })

        longs.sort(key=lambda leg: leg['strike'], reverse=True)
        shorts.sort(key=lambda leg: leg['strike'], reverse=True)

        for s_leg in shorts:
            while s_leg['rem'] > 0:
                candidates = [l for l in longs if l['rem'] > 0 and l['strike'] < s_leg['strike']]
                if not candidates:
                    break
                long_leg = max(candidates, key=lambda l: l['strike'])
                matched = min(s_leg['rem'], long_leg['rem'])
                total_spread_pnl += matched * (long_leg['pnl_per_contract'] + s_leg['pnl_per_contract'])
                s_leg['rem'] -= matched
                long_leg['rem'] -= matched

    return total_spread_pnl


def main():
    parser = argparse.ArgumentParser(
        description='Estimate option portfolio P/L from an IV crush tied to VIX drop scenarios.'
    )
    parser.add_argument('--file', default=None, help='Path to holdings CSV (default: my_holdings.csv next to script)')
    parser.add_argument('--r', type=float, default=0.04, help='Risk-free rate as decimal (default: 0.04)')
    parser.add_argument(
        '--as-of',
        default=None,
        help='Valuation date YYYY-MM-DD (default: today in local system date)',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    csv_path = Path(args.file) if args.file else script_dir / 'my_holdings.csv'

    df = pd.read_csv(csv_path, skiprows=2)
    df = df[~df['Symbol'].isin(['Account Total', 'Cash & Cash Investments'])].copy()

    asset_type_col = 'Security Type' if 'Security Type' in df.columns else 'Asset Type'
    if asset_type_col not in df.columns:
        raise KeyError("Could not find asset type column. Expected 'Security Type' or 'Asset Type'.")

    df = df[df[asset_type_col] == 'Option'].copy()
    parsed = df['Symbol'].apply(parse_symbol)
    df = df[parsed.notna()].copy()

    parsed_df = pd.DataFrame(parsed[parsed.notna()].tolist(), columns=['Ticker', 'Expiration', 'Strike Price', 'Opt Type'], index=df.index)
    df = pd.concat([df, parsed_df], axis=1)

    df['Qty'] = df['Qty (Quantity)'].apply(clean_numeric)
    df['Option Price'] = df['Price'].apply(clean_numeric)
    df = df[(df['Qty'] != 0) & (df['Option Price'] > 0)].copy()

    as_of = datetime.strptime(args.as_of, '%Y-%m-%d').date() if args.as_of else datetime.now().date()
    df['Expiry Date'] = pd.to_datetime(df['Expiration'], format='%m/%d/%Y', errors='coerce').dt.date
    df = df[df['Expiry Date'].notna()].copy()
    df['Days'] = (df['Expiry Date'] - as_of).apply(lambda d: d.days if pd.notna(d) else -1)
    df['T'] = df['Days'].clip(lower=1) / 365.0

    fit_map = {}
    for (ticker, expiry), grp in df.groupby(['Ticker', 'Expiration']):
        fit = fit_spot_sigma(grp, args.r)
        if fit is None:
            fit = FitParams(
                spot=float(grp['Strike Price'].median()),
                sigma=0.55,
                used_points=int(len(grp)),
            )
        fit_map[(ticker, expiry)] = fit

    df['Est Spot'] = df.apply(lambda r: fit_map[(r['Ticker'], r['Expiration'])].spot, axis=1)
    df['Est IV'] = df.apply(lambda r: fit_map[(r['Ticker'], r['Expiration'])].sigma, axis=1)

    df['Vega'] = bs_vega_vec(
        df['Est Spot'].to_numpy(dtype=float),
        df['Strike Price'].to_numpy(dtype=float),
        df['T'].to_numpy(dtype=float),
        args.r,
        df['Est IV'].to_numpy(dtype=float),
    )

    scenarios = [0.10, 0.20]

    print('\nIV Crush Stress (Approximate Vega Model)')
    print(f'File: {csv_path}')
    print(f'As-Of Date: {as_of}')
    print(f'Groups Fitted: {len(fit_map)} option chains')

    for drop in scenarios:
        shock_label = f'VIX -{int(drop * 100)}%'
        delta_sigma = -drop * df['Est IV']

        df[f'1c_delta_{int(drop * 100)}'] = df['Vega'] * delta_sigma * 100.0

        puts_df = df[df['Opt Type'] == 'P'].copy()
        calls_df = df[df['Opt Type'] == 'C'].copy()

        puts_total = (puts_df['Qty'] * puts_df[f'1c_delta_{int(drop * 100)}']).sum()

        call_spreads_total = match_call_spreads_for_scenario(calls_df, f'1c_delta_{int(drop * 100)}')
        total_calls = (calls_df['Qty'] * calls_df[f'1c_delta_{int(drop * 100)}']).sum()
        unmatched_calls = total_calls - call_spreads_total

        grand_total = puts_total + call_spreads_total

        print(f'\nScenario: {shock_label}')
        print(f'  Call Spreads P/L: ${call_spreads_total:,.2f}')
        print(f'  Puts P/L:         ${puts_total:,.2f}')
        print(f'  Total (Spreads + Puts): ${grand_total:,.2f}')
        print(f'  Unmatched Calls (not in spread pairing): ${unmatched_calls:,.2f}')


if __name__ == '__main__':
    main()