import pandas as pd
import numpy as np
from datetime import timedelta, date
from pathlib import Path


def generate_fiec009_data(start_date, end_date, *, seed: int = None, out_dir: str = '.', error_rates: dict = None):
    """
    Generate synthetic FIEC 009 data between start_date and end_date (inclusive).

    Parameters
    - start_date, end_date: datetime.date
    - seed: optional int for reproducibility
    - out_dir: output directory for CSVs
    - error_rates: dict of error rates (fractions) for injecting mistakes

    Returns tuple(deposit_filename, treasury_filename)
    """
    if seed is not None:
        np.random.seed(seed)

    if error_rates is None:
        error_rates = {
            'misclass_rate': 0.01,
            'missing_rate': 0.02,
            'invalid_account_rate': 0.005,
            'wrong_currency_rate': 0.005,
            'maturity_error_rate': 0.005,
            'interest_scale_error_rate': 0.002,
        }

    print("Starting data generation with error rates:", error_rates)

    # --- 1. Build base rows ---
    delta = end_date - start_date
    dates = []
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        num_transactions = np.random.randint(50, 151)
        dates.extend([current_date] * num_transactions)

    df = pd.DataFrame({'Date': dates})
    record_count = len(df)
    print(f"Total records to generate: {record_count}")

    # PROD_SET allocation
    df['PROD_SET'] = np.random.choice(['DEPOSIT', 'Treasury Liability'], size=record_count, p=[0.60, 0.40])

    # Customer and account
    df['Customer_ID'] = np.random.randint(100000, 999999, size=record_count)

    def gen_account():
        if np.random.rand() < 0.5:
            return f"{np.random.randint(10**9, 10**10 - 1)}"
        else:
            return f"{np.random.randint(10**11, 10**12 - 1)}"

    df['Account_Number'] = [gen_account() for _ in range(record_count)]

    # Balance (log-normal) and clip to reasonable range
    balances = np.exp(np.random.normal(loc=9, scale=1.5, size=record_count))
    df['Balance_Amount'] = np.clip(balances, 1.0, 5e7).round(2)

    # Currency and rates
    df['Currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'CAD', 'JPY'], size=record_count, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    df.loc[df['PROD_SET'] == 'DEPOSIT', 'Rate_of_Interest'] = np.random.uniform(0.01, 0.05, size=(df['PROD_SET'] == 'DEPOSIT').sum()).round(4)
    # Treasury rates are percentages too (0.5% - 4.5% -> 0.005 - 0.045)
    df.loc[df['PROD_SET'] == 'Treasury Liability', 'Rate_of_Interest'] = np.random.uniform(0.005, 0.045, size=(df['PROD_SET'] == 'Treasury Liability').sum()).round(4)

    # Maturity (30 days to 5 years)
    maturity_days = np.random.randint(30, 365 * 5, size=record_count)
    df['Maturity_Date'] = pd.to_datetime(df['Date']) + pd.to_timedelta(maturity_days, unit='D')

    df['Bank_Branch_Code'] = np.random.choice([f"BRN{i:03}" for i in range(1, 11)], size=record_count)
    df['Form_Fill_Status'] = np.random.choice(['Complete', 'Error'], size=record_count, p=[0.95, 0.05])

    # --- 2. Inject realistic errors ---
    n = record_count

    # 2.1 Misclassification: flip PROD_SET for a small fraction
    k = int(error_rates.get('misclass_rate', 0.01) * n)
    if k > 0:
        idx = np.random.choice(n, size=k, replace=False)
        df.loc[idx, 'PROD_SET'] = df.loc[idx, 'PROD_SET'].apply(lambda x: 'DEPOSIT' if x != 'DEPOSIT' else 'Treasury Liability')

    # 2.2 Missing random fields
    m = int(error_rates.get('missing_rate', 0.02) * n)
    fields = ['Account_Number', 'Balance_Amount', 'Currency', 'Rate_of_Interest']
    for _ in range(m):
        i = np.random.randint(0, n)
        f = np.random.choice(fields)
        df.at[i, f] = np.nan

    # 2.3 Invalid account numbers (insert letters or wrong length)
    ia = int(error_rates.get('invalid_account_rate', 0.005) * n)
    if ia > 0:
        valid_idx = df[df['Account_Number'].notna()].index.to_numpy()
        if len(valid_idx) > 0:
            size = min(ia, len(valid_idx))
            for i in np.random.choice(valid_idx, size=size, replace=False):
                a = str(df.at[i, 'Account_Number'])
                # corrupt by inserting letters or trimming
                if np.random.rand() < 0.5:
                    df.at[i, 'Account_Number'] = 'X' + a[1:]
                else:
                    df.at[i, 'Account_Number'] = a[:5]

    # 2.4 Wrong currency codes
    wc = int(error_rates.get('wrong_currency_rate', 0.005) * n)
    if wc > 0:
        for i in np.random.choice(n, size=wc, replace=False):
            df.at[i, 'Currency'] = 'XXX'

    # 2.5 Maturity errors: maturity before date or null
    me = int(error_rates.get('maturity_error_rate', 0.005) * n)
    if me > 0:
        for i in np.random.choice(n, size=me, replace=False):
            if np.random.rand() < 0.5:
                df.at[i, 'Maturity_Date'] = pd.to_datetime(df.at[i, 'Date']) - pd.Timedelta(days=np.random.randint(1, 30))
            else:
                df.at[i, 'Maturity_Date'] = pd.NaT

    # 2.6 Interest scale errors (e.g., values stored as percent instead of decimal)
    ie = int(error_rates.get('interest_scale_error_rate', 0.002) * n)
    if ie > 0:
        for i in np.random.choice(n, size=ie, replace=False):
            if pd.notna(df.at[i, 'Rate_of_Interest']):
                df.at[i, 'Rate_of_Interest'] = df.at[i, 'Rate_of_Interest'] * 100.0

    # 2.7 Add a few rows with malformed Date strings for format errors
    fe = max(1, int(0.001 * n))
    for i in np.random.choice(n, size=fe, replace=False):
        df.at[i, 'Date'] = '2024/31/12'  # invalid format

    # --- 3. Split and export ---
    deposit_df = df[df['PROD_SET'] == 'DEPOSIT'].copy()
    treasury_df = df[df['PROD_SET'] == 'Treasury Liability'].copy()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    deposit_filename = out_path / 'fiec009_DEPOSIT_data.csv'
    treasury_filename = out_path / 'fiec009_TREASURY_LIABILITY_data.csv'

    deposit_df.to_csv(deposit_filename, index=False)
    treasury_df.to_csv(treasury_filename, index=False)

    print("\n✅ Data generation complete!")
    print(f"Generated {len(deposit_df):,} records for 'DEPOSIT' and saved to: {deposit_filename}")
    print(f"Generated {len(treasury_df):,} records for 'Treasury Liability' and saved to: {treasury_filename}")
    return str(deposit_filename), str(treasury_filename)


# If run as script, produce one year with a fixed seed for reproducibility
if __name__ == '__main__':
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    generate_fiec009_data(start_date, end_date, seed=42, out_dir='.')