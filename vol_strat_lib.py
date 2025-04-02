
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


def process_options_ATM(df, low_high, cp_value, prefix, alpha):
    TTM_var = days_type() + "TTM"

    # Filter and sort options
    filtered = df[(df[low_high]) & (df['cp_flag'] == cp_value)].copy()

    is_call = 2 * (df['cp_flag'] == "C") - 1
    filtered["abs(moneyness+-alpha)"] = np.abs(filtered["moneyness"] - is_call * alpha)
    filtered["fit_error"] = filtered["abs(moneyness+-alpha)"]

    filtered = filtered.sort_values(['ticker', 'date', 'fit_error'],
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
    ranked = filtered[filtered['rank'] <= 1].copy()

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
    if alpha == 0:
        pivot_df.columns = [f'{prefix}_ATM_{col}' for col, rank in pivot_df.columns]
    else:
        pivot_df.columns = [f'{prefix}_OTM_{round(alpha*100)}%_{col}' for col, rank in pivot_df.columns]
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


def T_day_interpolation(T1, T2, r1, r2):
    if days_type() == "t_":
        T = 21
    elif days_type() == "c_":
        T = 30
    else:
        print("days_type() is neither t_ nor c_")
        T = np.nan
    return (r1 * np.abs(T2 - T) + r2 * np.abs(T1 - T)) / (np.abs(T2 - T) + np.abs(T1 - T))


def add_current_option_info(sum_df, od_hl, OTMs):
    '''
    Find and add ATM options to sum_df
    '''
    # Define processing configurations
    configs = [
        ('low', 'C', 'low_call'),
        ('low', 'P', 'low_put'),
        ('high', 'C', 'high_call'),
        ('high', 'P', 'high_put'),
    ]

    # Process all combinations
    moneyness_list = [0] + OTMs
    pivoted_dfs = [process_options_ATM(od_hl, low_high, cp_flag, prefix, alpha)
                   for low_high, cp_flag, prefix in configs for alpha in moneyness_list]

    # Merge all results
    for p_df in pivoted_dfs:
        sum_df = sum_df.merge(p_df, on=['ticker', 'date'], how='left')
    return sum_df


def add_next_prices_to_sum(sum_df, od_raw):
    '''
    Add next price of options to sum_df
    '''
    # Create next price lookup
    od_sorted = od_raw.sort_values(['optionid', 'date']).copy()
    od_sorted['price_next'] = od_sorted.groupby('optionid')['price'].shift(-1)
    next_price_lookup = od_sorted[['ticker', 'date', 'optionid', 'price_next']]

    # Add next prices to all option columns
    sum_df = add_next_prices(sum_df, next_price_lookup)
    return sum_df


def add_ATM_options_to_sum_df(sum_df, od_hl, od_raw, OTMs):
    '''
    Add options and their future value to sum_df
    '''
    sum_df = add_current_option_info(sum_df, od_hl, OTMs)
    sum_df = add_next_prices_to_sum(sum_df, od_raw)
    return sum_df