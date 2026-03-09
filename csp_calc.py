import argparse
from pathlib import Path
import re

import pandas as pd


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() in ['N/A', '-', '']:
        return 0.0
    val_str = str(val).replace('$', '').replace('%', '').replace(',', '')
    if '(' in val_str:
        val_str = '-' + val_str.replace('(', '').replace(')', '')
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def extract_details(symbol):
    match = re.match(r'^\s*(\S+)\s+(\d{2}/\d{2}/\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$', str(symbol))
    if match:
        return pd.Series([match.group(1), match.group(2), float(match.group(3)), match.group(4)])
    return pd.Series(['', '', 0.0, ''])


def find_uncovered_short_puts(df_puts):
    uncovered_rows = []

    for (_, _), group in df_puts.groupby(['Ticker', 'Expiration']):
        longs = []
        shorts = []

        for _, row in group.sort_values(by='Strike Price', ascending=False).iterrows():
            qty = float(row['Qty'])
            if qty > 0:
                longs.append({'strike': float(row['Strike Price']), 'remaining': qty})
            elif qty < 0:
                shorts.append(row.copy())

        for short_row in shorts:
            remaining_short = abs(float(short_row['Qty']))
            short_strike = float(short_row['Strike Price'])

            # Only a lower-strike long put can cap a short put as a vertical spread.
            candidates = [leg for leg in longs if leg['remaining'] > 0 and leg['strike'] < short_strike]
            candidates.sort(key=lambda leg: leg['strike'], reverse=True)

            for long_leg in candidates:
                if remaining_short <= 0:
                    break
                matched = min(remaining_short, long_leg['remaining'])
                remaining_short -= matched
                long_leg['remaining'] -= matched

            if remaining_short > 0:
                uncovered = short_row.copy()
                original_contracts = abs(float(short_row['Qty']))
                ratio = remaining_short / original_contracts if original_contracts else 0.0
                uncovered['Contracts Sold'] = remaining_short
                uncovered['Current Mkt Value'] = clean_numeric(short_row['Mkt Val (Market Value)']) * ratio
                uncovered['Cash Secured ($)'] = remaining_short * 100 * short_strike
                uncovered_rows.append(uncovered)

    if not uncovered_rows:
        columns = list(df_puts.columns) + ['Contracts Sold', 'Current Mkt Value', 'Cash Secured ($)']
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(uncovered_rows)


def main():
    parser = argparse.ArgumentParser(description='Find naked short puts and estimate cash-secured requirement')
    parser.add_argument('--file', default=None, help='Path to holdings CSV (default: my_holdings.csv next to script)')
    parser.add_argument('--output', default=None, help='Output Excel path (default: naked_short_puts.xlsx next to script)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    csv_path = Path(args.file) if args.file else script_dir / 'my_holdings.csv'
    output_path = Path(args.output) if args.output else script_dir / 'naked_short_puts.xlsx'

    df = pd.read_csv(csv_path, skiprows=2)
    df = df[~df['Symbol'].isin(['Account Total', 'Cash & Cash Investments'])].copy()

    asset_type_col = 'Security Type' if 'Security Type' in df.columns else 'Asset Type'
    if asset_type_col not in df.columns:
        raise KeyError("Could not find asset type column. Expected 'Security Type' or 'Asset Type'.")

    df['Qty'] = df['Qty (Quantity)'].apply(clean_numeric)

    df_options = df[df[asset_type_col] == 'Option'].copy()
    df_options[['Ticker', 'Expiration', 'Strike Price', 'Opt Type']] = df_options['Symbol'].apply(extract_details)

    df_puts = df_options[df_options['Opt Type'] == 'P'].copy()
    df_naked_short_puts = find_uncovered_short_puts(df_puts)

    output_cols = ['Ticker', 'Symbol', 'Contracts Sold', 'Strike Price', 'Current Mkt Value', 'Cash Secured ($)']
    df_out = df_naked_short_puts[output_cols].copy()

    total_row = pd.DataFrame([
        {
            'Ticker': 'GRAND TOTAL',
            'Symbol': '',
            'Contracts Sold': df_out['Contracts Sold'].sum(),
            'Strike Price': 0.0,
            'Current Mkt Value': df_out['Current Mkt Value'].sum(),
            'Cash Secured ($)': df_out['Cash Secured ($)'].sum(),
        }
    ])
    if df_out.empty:
        df_out = total_row.copy()
    else:
        df_out = pd.concat([df_out, total_row], ignore_index=True)

    df_out.to_excel(output_path, index=False)

    grand_cash = df_naked_short_puts['Cash Secured ($)'].sum()
    grand_mkt = df_naked_short_puts['Current Mkt Value'].sum()

    print(f'\nSuccess! Data exported to Excel file: {output_path}')
    print('\n--- GRAND TOTALS ---')
    print(f'Total Cash Secured: ${grand_cash:,.2f}')
    print(f'Total Current Liability (Mkt Value): ${grand_mkt:,.2f}\n')


if __name__ == '__main__':
    main()