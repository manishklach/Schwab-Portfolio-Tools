# Schwab Portfolio Utilities

Small Python utilities for analyzing a Schwab downloaded portfolio CSV locally.

These scripts are intended for personal use on your machine. They read a Schwab portfolio export, produce terminal summaries or local spreadsheet outputs, and do not require uploading or sharing your account data.

## Privacy and data handling

This project is designed to work with local files only.

- Do not commit or share `my_holdings.csv`.
- Do not commit generated `.csv` or `.xlsx` outputs.
- Keep the project folder local if it contains real account data.
- If you ever share the code, share only the Python files and `README.md`, not your holdings files.

A `.gitignore` file is included to reduce accidental sharing of raw Schwab exports and generated reports.

## Supported input

These scripts expect a Schwab portfolio CSV export with Schwab-style columns such as:

- `Symbol`
- `Qty (Quantity)`
- `Price`
- `Day Chng $ (Day Change $)`
- `Mkt Val (Market Value)`
- `Security Type` or `Asset Type`

Option symbols are expected in the Schwab format:

```text
AAPL 06/20/2026 200 C
TSLA 04/17/2026 250 P
```

## Files

- `portfolio_day_change.py`: Aggregates `Day Chng $` by ticker from pasted Schwab portfolio text.
- `day_change_call_spreads.py`: Identifies call debit spreads and removes their day-change contribution from the portfolio total.
- `csp_calc.py`: Finds uncovered short puts and estimates cash-secured capital required.
- `iv_crush_impact.py`: Estimates option portfolio sensitivity to implied-volatility drops.

## Requirements

Install Python 3.10+.

Install packages:

```powershell
python -m pip install pandas numpy openpyxl
```

`openpyxl` is needed for Excel output from `csp_calc.py`.

## Setup

1. Export your portfolio from Schwab as CSV.
2. Copy the CSV into this folder.
3. Rename it to `my_holdings.csv`.

Default file expected by the scripts:

```text
my_holdings.csv
```

You can also keep a different filename and pass it with `--file`.

## Quick start

### 1. Aggregate day change from pasted Schwab portfolio text

This script is different from the others: it reads pasted text from standard input rather than reading the CSV file directly.

Example with a text file containing pasted Schwab portfolio table data:

```powershell
Get-Content .\input.txt | python .\portfolio_day_change.py
```

Group option contracts by underlying ticker:

```powershell
Get-Content .\input.txt | python .\portfolio_day_change.py --underlying
```

What it does:

- detects the `Day Chng $` column
- aggregates values by symbol
- excludes `Cash`, `Cash & Cash Investments`, and `Account Total`
- prints a sorted total by ticker plus a `TOTAL` line

### 2. Remove call debit spreads from portfolio day change

```powershell
python .\day_change_call_spreads.py
```

With an explicit file path:

```powershell
python .\day_change_call_spreads.py --file .\my_holdings.csv
```

Verbose spread list:

```powershell
python .\day_change_call_spreads.py --verbose
```

What it does:

- reads Schwab option rows from the CSV
- identifies call debit spreads where the long call strike is below the short call strike
- matches quantities across spreads
- reports:
  - original portfolio day change
  - spread day change removed
  - filtered portfolio day change

### 3. Estimate cash needed for uncovered short puts

```powershell
python .\csp_calc.py
```

Write to a custom output file:

```powershell
python .\csp_calc.py --output .\my_short_put_report.xlsx
```

Use a custom input CSV:

```powershell
python .\csp_calc.py --file .\my_holdings.csv --output .\my_short_put_report.xlsx
```

What it does:

- reads Schwab option rows from the CSV
- isolates put positions
- treats only lower-strike long puts as protection against a short put
- handles partial coverage correctly by quantity
- exports an Excel file with uncovered short puts and a grand total row

### 4. Estimate IV crush impact

```powershell
python .\iv_crush_impact.py
```

Set valuation date and risk-free rate:

```powershell
python .\iv_crush_impact.py --as-of 2026-03-09 --r 0.04
```

Use a custom input CSV:

```powershell
python .\iv_crush_impact.py --file .\my_holdings.csv
```

What it does:

- reads Schwab option rows from the CSV
- estimates spot and implied volatility for each option chain
- computes approximate vega exposure
- evaluates two scenarios:
  - VIX down 10%
  - VIX down 20%
- separates puts from matched call debit spreads

## Notes on the recent fixes

The current scripts include a few important corrections:

- `csp_calc.py` now handles partial hedges correctly and does not mark all short puts as covered just because a single long put exists at the same expiration.
- `iv_crush_impact.py` now matches only valid call debit spreads with long-lower and short-higher strikes.
- `day_change_call_spreads.py` and `csp_calc.py` now use normal command-line entrypoints and stricter argument parsing.

## Safe sharing checklist

If you want to share this project without exposing account data:

1. Remove `my_holdings.csv`.
2. Remove generated `.csv` and `.xlsx` reports.
3. Share only:
   - `README.md`
   - `portfolio_day_change.py`
   - `day_change_call_spreads.py`
   - `csp_calc.py`
   - `iv_crush_impact.py`

## Troubleshooting

### Missing columns

If Schwab changes its export format and you see errors about missing columns, open the CSV header row and verify the expected Schwab column names still exist.

### Excel output errors

If Excel export fails, install `openpyxl`:

```powershell
python -m pip install openpyxl
```

### Option rows not recognized

These scripts expect Schwab option symbols in the standard format shown above. If Schwab changes that symbol format, the regex parsing will need an update.