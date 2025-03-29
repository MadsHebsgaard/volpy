

import pandas as pd
import numpy as np

import load_clean_lib
from global_settings import *

def import_sum_raw(om_folder):
    om_folder = load_clean_lib.volpy_output_dir(om_folder)
    time_type = days_type()
    sum_df = pd.read_csv(f"{om_folder}/{time_type}summary_dly.csv")
    sum_df["date"] = pd.to_datetime(sum_df["date"])
    sum_df[f"return_next"] = sum_df[f"return"].shift(-1)

    od_raw = pd.read_csv(f"{om_folder}/{time_type}od_raw.csv")
    od_raw['optionid'] = od_raw['optionid'].astype(int)
    od_raw["date"] = pd.to_datetime(od_raw["date"])
    return sum_df, od_raw


def import_sum(om_folder):
    om_folder = load_clean_lib.volpy_output_dir(om_folder)
    time_type = days_type()
    sum_df = pd.read_csv(f"{om_folder}/{time_type}summary_dly.csv")
    sum_df["date"] = pd.to_datetime(sum_df["date"])
    sum_df[f"return_next"] = sum_df[f"return"].shift(-1)
    return sum_df


def create_od_hl(od_raw, sum_df, price_type, IV_type):
    days_var = days_type() + "days"

    od_raw['price'] = od_raw[price_type]

    od_hl = od_raw.merge(
        sum_df[['ticker', 'date', 'high days', 'low days']],
        on=['ticker', 'date'],
        how='left'
    )

    # Create two new columns 'high' and 'low'
    od_hl['low'] = od_hl[days_var] == od_hl['low days']
    od_hl['high'] = od_hl[days_var] == od_hl['high days']
    od_hl = od_hl[(od_hl['low']) | (od_hl['high'])]
    od_hl = od_hl.sort_values(by=['ticker', 'date'], ascending=True)

    od_hl["moneyness"] = np.log(od_hl["K"] / od_hl["F"])
    od_hl["abs(moneyness)"] = np.abs(od_hl["moneyness"])
    od_hl["IV"] = od_hl[f"IV_{IV_type}"]

    return od_hl


# Compute F_high and F_low
def add_F_to_sum_df(od_hl, sum_df):
    f_low_values = od_hl[od_hl["low"]].groupby(['ticker', 'date'])['F'].first().reset_index()
    f_high_values = od_hl[od_hl["high"]].groupby(['ticker', 'date'])['F'].first().reset_index()

    # Rename the columns before merging
    f_low_values.rename(columns={'F': 'F_low'}, inplace=True)
    f_high_values.rename(columns={'F': 'F_high'}, inplace=True)

    # Merge into sum_df
    sum_df = sum_df.merge(f_low_values, on=['ticker', 'date'], how='left')
    sum_df = sum_df.merge(f_high_values, on=['ticker', 'date'], how='left')

    return sum_df


from scipy.stats import norm


def process_options_ATM(df, low_high, cp_value, prefix):
    TTM_var = days_type() + "TTM"

    # Filter and sort options
    filtered = df[(df[low_high]) & (df['cp_flag'] == cp_value)].copy()
    filtered = filtered.sort_values(['ticker', 'date', 'abs(moneyness)'],
                                    ascending=[True, True, True])
    filtered["T"] = filtered[TTM_var]

    # Calculate d1 and delta
    filtered['d1'] = (np.log(filtered["F"] / filtered["K"]) + 0.5 * filtered["IV"] ** 2 * filtered["T"]) / (
                filtered["IV"] * np.sqrt(filtered["T"]))
    if cp_value == 'C':
        filtered['delta'] = norm.cdf(filtered['d1'])
    else:
        filtered['delta'] = norm.cdf(filtered['d1']) - 1

    # Create ranks and filter top 2
    filtered['rank'] = filtered.groupby(['ticker', 'date']).cumcount() + 1
    ranked = filtered[filtered['rank'] <= 2].copy()

    # Convert optionid to integer
    ranked['optionid'] = ranked['optionid'].astype(int)

    # Create pivot table
    pivot_df = ranked.pivot_table(
        index=['ticker', 'date'],
        columns='rank',
        values=['optionid', 'price', 'K', 'delta'],
        aggfunc='first'
    )

    # Flatten multi-index columns
    pivot_df.columns = [f'{prefix}_{col}_{rank}' for col, rank in pivot_df.columns]
    return pivot_df.reset_index()


def add_next_prices(sum_df, next_lookup):
    optionid_cols = [col for col in sum_df.columns if '_optionid' in col]

    for col in optionid_cols:
        prefix = col.replace('_optionid', '')
        # Merge next prices using optionid and current date
        sum_df = sum_df.merge(
            next_lookup.rename(columns={'optionid': col}),
            left_on=['ticker', 'date', col],
            right_on=['ticker', 'date', col],
            how='left'
        )
        # Clean up column names
        sum_df.rename(columns={'price_next': f'{prefix}_price_next'}, inplace=True)

    return sum_df


def T_day_interpolation(T1, T2, r1, r2, T = 30):
    return (r1 * np.abs(T2 - T) + r2 * np.abs(T1 - T)) / (np.abs(T2 - T) + np.abs(T1 - T))