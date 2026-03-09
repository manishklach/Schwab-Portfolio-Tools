import argparse
from collections import deque
from pathlib import Path
import re

import pandas as pd


SYMBOL_RE = re.compile(r'^\s*(\S+)\s+(\d{2}/\d{2}/\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$')


class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        fwd = [v, cap, None]
        rev = [u, 0.0, None]
        fwd[2] = len(self.adj[v])
        rev[2] = len(self.adj[u])
        self.adj[u].append(fwd)
        self.adj[v].append(rev)

    def _bfs(self, s, t, level):
        for i in range(self.n):
            level[i] = -1
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, _ in self.adj[u]:
                if cap > 1e-12 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[t] >= 0

    def _dfs(self, u, t, f, level, it):
        if u == t:
            return f
        while it[u] < len(self.adj[u]):
            i = it[u]
            v, cap, rev_idx = self.adj[u][i]
            if cap > 1e-12 and level[u] + 1 == level[v]:
                pushed = self._dfs(v, t, min(f, cap), level, it)
                if pushed > 1e-12:
                    self.adj[u][i][1] -= pushed
                    self.adj[v][rev_idx][1] += pushed
                    return pushed
            it[u] += 1
        return 0.0

    def max_flow(self, s, t):
        flow = 0.0
        level = [-1] * self.n
        while self._bfs(s, t, level):
            it = [0] * self.n
            while True:
                pushed = self._dfs(s, t, float('inf'), level, it)
                if pushed <= 1e-12:
                    break
                flow += pushed
        return flow


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
    match = SYMBOL_RE.match(str(symbol))
    if match:
        return pd.Series([match.group(1), match.group(2), float(match.group(3)), match.group(4)])
    return pd.Series(['', '', 0.0, ''])


def optimize_call_spreads(group):
    longs_df = group[group['Qty'] > 0].copy().reset_index(drop=True)
    shorts_df = group[group['Qty'] < 0].copy().reset_index(drop=True)

    if longs_df.empty or shorts_df.empty:
        return []

    L = len(longs_df)
    S = len(shorts_df)
    source = 0
    long_base = 1
    short_base = long_base + L
    sink = short_base + S

    dinic = Dinic(sink + 1)

    long_caps = longs_df['Qty'].astype(float).tolist()
    short_caps = shorts_df['Qty'].abs().astype(float).tolist()

    for li, cap in enumerate(long_caps):
        dinic.add_edge(source, long_base + li, cap)

    edge_map = []
    for li in range(L):
        long_strike = float(longs_df.loc[li, 'Strike Price'])
        for si in range(S):
            short_strike = float(shorts_df.loc[si, 'Strike Price'])
            width = short_strike - long_strike
            if long_strike < short_strike and width <= 10.0:
                u = long_base + li
                v = short_base + si
                edge_idx = len(dinic.adj[u])
                dinic.add_edge(u, v, 1e12)
                edge_map.append((li, si, width, u, edge_idx))

    for si, cap in enumerate(short_caps):
        dinic.add_edge(short_base + si, sink, cap)

    total_matched = dinic.max_flow(source, sink)
    if total_matched <= 1e-12:
        return []

    spreads = []
    for li, si, width, u, edge_idx in edge_map:
        v, _, rev_idx = dinic.adj[u][edge_idx]
        flow_used = dinic.adj[v][rev_idx][1]
        if flow_used > 1e-12:
            long_row = longs_df.loc[li]
            short_row = shorts_df.loc[si]
            long_qty = float(long_row['Qty'])
            short_qty = abs(float(short_row['Qty']))
            spreads.append({
                'Ticker': str(long_row['Ticker']),
                'Expiration': str(long_row['Expiration']),
                'Long Strike': float(long_row['Strike Price']),
                'Short Strike': float(short_row['Strike Price']),
                'Matched Qty': float(flow_used),
                'Width': float(width),
                'Long Day Chng / Ctr': float(long_row['Day Chng $'] / long_qty) if long_qty != 0 else 0.0,
                'Short Day Chng / Ctr': float(short_row['Day Chng $'] / short_qty) if short_qty != 0 else 0.0,
            })

    return spreads


def main():
    parser = argparse.ArgumentParser(description='Filter debit call spread day-change impact (optimized matching)')
    parser.add_argument('--verbose', action='store_true', help='Print eliminated spreads')
    parser.add_argument('--file', default=None, help='Path to holdings CSV (default: my_holdings.csv next to script)')
    args = parser.parse_args()

    csv_path = Path(args.file) if args.file else Path(__file__).with_name('my_holdings.csv')
    df = pd.read_csv(csv_path, skiprows=2)
    df = df[~df['Symbol'].isin(['Account Total', 'Cash & Cash Investments'])].copy()

    asset_type_col = 'Security Type' if 'Security Type' in df.columns else 'Asset Type'
    if asset_type_col not in df.columns:
        raise KeyError("Could not find asset type column. Expected 'Security Type' or 'Asset Type'.")

    df['Qty'] = df['Qty (Quantity)'].apply(clean_numeric)
    df['Day Chng $'] = df['Day Chng $ (Day Change $)'].apply(clean_numeric)

    df_options = df[df[asset_type_col] == 'Option'].copy()
    df_options[['Ticker', 'Expiration', 'Strike Price', 'Opt Type']] = df_options['Symbol'].apply(extract_details)
    df_calls = df_options[df_options['Opt Type'] == 'C'].copy()

    spreads_found = []
    for (_, _), group in df_calls.groupby(['Ticker', 'Expiration']):
        spreads_found.extend(optimize_call_spreads(group))

    df_spreads = pd.DataFrame(spreads_found)
    if df_spreads.empty:
        target_spreads = pd.DataFrame(columns=['Ticker', 'Expiration', 'Long Strike', 'Short Strike', 'Matched Qty', 'Width', 'Long Day Chng / Ctr', 'Short Day Chng / Ctr'])
    else:
        target_spreads = df_spreads.copy()

    spread_day_chng_to_remove = 0.0
    for _, row in target_spreads.iterrows():
        spread_day_chng_to_remove += (row['Matched Qty'] * row['Long Day Chng / Ctr'])
        spread_day_chng_to_remove += (row['Matched Qty'] * row['Short Day Chng / Ctr'])

    original_day_change = df['Day Chng $'].sum()
    filtered_day_change = original_day_change - spread_day_chng_to_remove

    print('--- SPREAD FILTERING ---')
    print(f'Identified {len(target_spreads)} optimized Call Debit Spread pairings ($10 or less width).')

    if args.verbose:
        print('\n--- ELIMINATED SPREADS (VERBOSE) ---')
        target_spreads = target_spreads.sort_values(by=['Ticker', 'Expiration', 'Long Strike', 'Short Strike'])
        for _, row in target_spreads.iterrows():
            print(
                f"[{row['Ticker']}] Exp: {row['Expiration']} | QTY: {row['Matched Qty']:g} | "
                f"Long: {row['Long Strike']:g}C / Short: {row['Short Strike']:g}C (Width: ${row['Width']:g})"
            )
        print()

    print('\n--- DAY CHANGE RESULTS ---')
    print(f'Original Portfolio Day Change: ${original_day_change:,.2f}')
    print(f'Removed Spread Day Change:   ${spread_day_chng_to_remove:,.2f}')
    print(f'Filtered Portfolio Day Change: ${filtered_day_change:,.2f}\n')


if __name__ == '__main__':
    main()