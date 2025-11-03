#!/usr/bin/env python3
"""CLI wrapper for generate_fiec009_data.

Provides --mode (clean|dirty|paired), --error-rate (percent), --seed,
--start/--end, and --out-dir. Uses the generator defined in
`009form/generate_fiec009_data.py`.
"""
import argparse
from pathlib import Path
import runpy
import numpy as np
import pandas as pd


def apply_corruptions_to_df(df: pd.DataFrame, seed: int, error_rates: dict):
    np.random.seed(seed)
    df = df.copy()
    n = len(df)
    # missing fields
    m = int(error_rates.get('missing_rate', 0.0) * n)
    fields = ['Account_Number', 'Balance_Amount', 'Currency', 'Rate_of_Interest']
    if m > 0:
        for i in np.random.choice(n, size=m, replace=False):
            f = np.random.choice(fields)
            df.iat[i, df.columns.get_loc(f)] = np.nan
    # invalid account numbers
    ia = int(error_rates.get('invalid_account_rate', 0.0) * n)
    if ia > 0:
        valid_idx = df[df['Account_Number'].notna()].index.to_numpy()
        if len(valid_idx) > 0:
            size = min(ia, len(valid_idx))
            for i in np.random.choice(valid_idx, size=size, replace=False):
                a = str(df.at[i, 'Account_Number'])
                if np.random.rand() < 0.5:
                    df.at[i, 'Account_Number'] = 'X' + a[1:]
                else:
                    df.at[i, 'Account_Number'] = a[:5]
    # wrong currency
    wc = int(error_rates.get('wrong_currency_rate', 0.0) * n)
    if wc > 0:
        for i in np.random.choice(n, size=wc, replace=False):
            df.iat[i, df.columns.get_loc('Currency')] = 'XXX'
    # maturity errors
    me = int(error_rates.get('maturity_error_rate', 0.0) * n)
    if me > 0:
        for i in np.random.choice(n, size=me, replace=False):
            if np.random.rand() < 0.5:
                df.iat[i, df.columns.get_loc('Maturity_Date')] = (pd.to_datetime(df.iat[i, df.columns.get_loc('Date')]) - pd.Timedelta(days=int(np.random.randint(1,30)))).strftime('%Y-%m-%d')
            else:
                df.iat[i, df.columns.get_loc('Maturity_Date')] = ''
    # interest scale errors
    ie = int(error_rates.get('interest_scale_error_rate', 0.0) * n)
    if ie > 0:
        for i in np.random.choice(n, size=ie, replace=False):
            col = df.columns.get_loc('Rate_of_Interest')
            val = df.iat[i, col]
            try:
                if pd.notna(val):
                    df.iat[i, col] = float(val) * 100.0
            except Exception:
                pass
    return df


def main():
    p = argparse.ArgumentParser(description='FIEC009 synthetic data CLI')
    p.add_argument('--mode', choices=['clean','dirty','paired'], default='dirty')
    p.add_argument('--start', default='2024-01-01')
    p.add_argument('--end', default='2024-12-31')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='.')
    p.add_argument('--error-rate', type=float, default=2.0, help='overall percent error rate')
    args = p.parse_args()

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()
    seed = args.seed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load generator
    mod = runpy.run_path(str(Path(__file__).parent / 'generate_fiec009_data.py'))
    gen = mod['generate_fiec009_data']

    total = args.error_rate / 100.0
    prop = {
        'misclass_rate': 0.15,
        'missing_rate': 0.5,
        'invalid_account_rate': 0.1,
        'wrong_currency_rate': 0.1,
        'maturity_error_rate': 0.075,
        'interest_scale_error_rate': 0.075,
    }
    error_rates = {k: total * v for k,v in prop.items()}

    if args.mode == 'clean':
        gen(start, end, seed=seed, out_dir=str(out_dir), error_rates={k:0.0 for k in prop})
        print('Clean files written to', out_dir)
    elif args.mode == 'dirty':
        gen(start, end, seed=seed, out_dir=str(out_dir), error_rates=error_rates)
        print('Dirty files written to', out_dir)
    else:
        # paired: generate clean then deterministic corruptions
        clean_dir = out_dir / 'ground_truth'
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_files = gen(start, end, seed=seed, out_dir=str(clean_dir), error_rates={k:0.0 for k in prop})
        # load clean CSVs
        dep = pd.read_csv(clean_files[0])
        tre = pd.read_csv(clean_files[1])
        combined = pd.concat([dep, tre], ignore_index=True)
        np.random.seed(seed)
        mis_k = int(error_rates.get('misclass_rate', 0.0) * len(combined))
        mis_idx = np.random.choice(len(combined), size=mis_k, replace=False) if mis_k>0 else []
        noisy = apply_corruptions_to_df(combined, seed+1, error_rates)
        # flip prod_set for mis_idx
        for i in mis_idx:
            noisy.at[i, 'PROD_SET'] = 'DEPOSIT' if noisy.at[i, 'PROD_SET'] != 'DEPOSIT' else 'Treasury Liability'
        dep_noisy = noisy[noisy['PROD_SET']=='DEPOSIT']
        tre_noisy = noisy[noisy['PROD_SET']=='Treasury Liability']
        noisy_dir = out_dir / 'noisy'
        noisy_dir.mkdir(parents=True, exist_ok=True)
        dep_noisy.to_csv(noisy_dir / 'fiec009_DEPOSIT_data.csv', index=False)
        tre_noisy.to_csv(noisy_dir / 'fiec009_TREASURY_LIABILITY_data.csv', index=False)
        print('Wrote paired clean ->', clean_dir, 'and noisy ->', noisy_dir)

if __name__ == '__main__':
    main()
