

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


def add_current_option_info(sum_df, od_hl, moneyness):
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
    pivoted_dfs = [process_options_ATM(od_hl, low_high, cp_flag, prefix, alpha)
                   for low_high, cp_flag, prefix in configs for alpha in [0, moneyness]]

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


def add_ATM_options_to_sum_df(sum_df, od_hl, od_raw, moneyness):
    '''
    Add options and their future value to sum_df
    '''
    sum_df = add_current_option_info(sum_df, od_hl, moneyness)
    sum_df = add_next_prices_to_sum(sum_df, od_raw)
    return sum_df


def add_put_and_call_sgy(df, OTM_moneyness = None):
    if OTM_moneyness == None:
        OTM_str = "ATM"
        OTM_simple = OTM_str
    else:
        OTM_str = f"OTM_{round(OTM_moneyness * 100)}%"
        OTM_simple = "OTM"

    for put_call in ["put", "call"]:
        for low_high in ["low", "high"]:
            df[f"free_{low_high}_{put_call}_{OTM_str}"] = (
                    df[f"{low_high}_{put_call}_{OTM_str}_price_next"] - (1 + df[f"RF"]) * df[f"{low_high}_{put_call}_{OTM_str}_price"]
            ).shift(1)

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    for put_call in ["put", "call"]:
        # df[f"30_{put_call}_{OTM_str}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{put_call}_{OTM_str}"],
        #                                               r2=df[f"free_high_{put_call}_{OTM_str}"])
        df[f"30_{put_call}_{OTM_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{put_call}_{OTM_str}"],
                                                      r2=df[f"free_high_{put_call}_{OTM_str}"])
    return df


def add_delta_put_and_call_sgy(df, OTM_moneyness = None):
    if OTM_moneyness == None:
        OTM_str = "ATM"
        OTM_simple = OTM_str
    else:
        OTM_str = f"OTM_{round(OTM_moneyness * 100)}%"
        OTM_simple = "OTM"

    for put_call in ["put", "call"]:
        for low_high in ["low", "high"]:
            df[f"free_{low_high}_{put_call}_D_{OTM_str}"] = (
                    df[f"{low_high}_{put_call}_{OTM_str}_price_next"] - (1 + df[f"RF"]) * df[f"{low_high}_{put_call}_{OTM_str}_price"] -
                    df[f"{low_high}_{put_call}_{OTM_str}_delta"] * df[f"close"] * (df[f"return"].shift(-1) - df[f"RF"])
            # Delta hedge (self financed)
            ).shift(1)

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    for put_call in ["put", "call"]:
        # df[f"30_{put_call}_D_{OTM_str}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{put_call}_D_{OTM_str}"],
        #                                                 r2=df[f"free_high_{put_call}_D_{OTM_str}"])
        df[f"30_{put_call}_D_{OTM_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{put_call}_D_{OTM_str}"],
                                                        r2=df[f"free_high_{put_call}_D_{OTM_str}"])
    return df


# def add_straddle_sgy_old(df):
#     for low_high in ["low", "high"]:
#         df[f"free_{low_high}_straddle"] = df[f"free_{low_high}_put_ATM"] + df[f"free_{low_high}_call_ATM"]
#
#     # 30 day version
#     T1 = df["low days"].shift(1)
#     T2 = df["high days"].shift(1)
#     df["30_straddle"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_straddle"], r2=df[f"free_high_straddle"])
#     return df

def add_straddle_sgy(df):
    add_strangle_sgy(df) #Passed without 'OTM_moneyness'
    return df

def add_strangle_sgy(df, OTM_moneyness = None):
    if OTM_moneyness == None:
        OTM_str = "ATM"
        sgy_name = "straddle"
        sgy_name_simple = sgy_name
    else:
        OTM_str = f"OTM_{round(OTM_moneyness * 100)}%"
        sgy_name = f"strangle_{round(OTM_moneyness * 100)}%"
        sgy_name_simple = "strangle"

    for low_high in ["low", "high"]:
        df[f"free_{low_high}_{sgy_name}"] = df[f"free_{low_high}_put_{OTM_str}"] + df[f"free_{low_high}_call_{OTM_str}"]

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    # df[f"30_{sgy_name}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    df[f"30_{sgy_name_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    return df


def add_butterfly_spread_sgy(df, OTM_moneyness):
    ATM_str = f"ATM"
    OTM_str = f"OTM_{round(OTM_moneyness * 100)}%"
    sgy_name = f"butterfly_spread_{round(OTM_moneyness * 100)}%"
    sgy_name_simple = "butterfly_spread"

    for low_high in ["low", "high"]:
        ATM_payoff = df[f"free_{low_high}_put_{ATM_str}"] + df[f"free_{low_high}_call_{ATM_str}"]
        OTM_payoff = df[f"free_{low_high}_put_{OTM_str}"] + df[f"free_{low_high}_call_{OTM_str}"]
        df[f"free_{low_high}_{sgy_name}"] = ATM_payoff - OTM_payoff

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    # df[f"30_{sgy_name}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    df[f"30_{sgy_name_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    return df

def add_delta_butterfly_spread_sgy(df, OTM_moneyness):
    ATM_str = f"D_ATM"
    OTM_str = f"D_OTM_{round(OTM_moneyness * 100)}%"
    sgy_name = f"D_butterfly_spread_{round(OTM_moneyness * 100)}%"
    sgy_name_simple = ("D_butterfly_spread")

    for low_high in ["low", "high"]:
        ATM_payoff = df[f"free_{low_high}_put_{ATM_str}"] + df[f"free_{low_high}_call_{ATM_str}"] # buy straddle
        OTM_payoff = df[f"free_{low_high}_put_{OTM_str}"] + df[f"free_{low_high}_call_{OTM_str}"] # sell strangle
        df[f"free_{low_high}_{sgy_name}"] = ATM_payoff - OTM_payoff

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    # df[f"30_{sgy_name}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    df[f"30_{sgy_name_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    return df


# def add_delta_straddle_sgy_old(df):
#     for low_high in ["low", "high"]:
#         df[f"free_D_{low_high}_straddle"] = df[f"free_D_{low_high}_put"] + df[f"free_D_{low_high}_call"]
#
#     # 30 day version
#     T1 = df["low days"].shift(1)
#     T2 = df["high days"].shift(1)
#     df["30_D_straddle"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_D_low_straddle"],
#                                                  r2=df[f"free_D_high_straddle"])
#     return df

def add_delta_strangle_sgy(df, OTM_moneyness = None):
    if OTM_moneyness == None:
        OTM_str = "D_ATM"
        sgy_name = "D_straddle"
        sgy_name_simple = sgy_name
    else:
        OTM_str = f"D_OTM_{round(OTM_moneyness * 100)}%"
        sgy_name = f"D_strangle_{round(OTM_moneyness * 100)}%"
        sgy_name_simple = "D_strangle"

    for low_high in ["low", "high"]:
        df[f"free_{low_high}_{sgy_name}"] = df[f"free_{low_high}_put_{OTM_str}"] + df[f"free_{low_high}_call_{OTM_str}"]

    # 30 day version
    T1 = df["low days"].shift(1)
    T2 = df["high days"].shift(1)
    # df[f"30_{sgy_name}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    df[f"30_{sgy_name_simple}"] = T_day_interpolation(T1=T1, T2=T2, r1=df[f"free_low_{sgy_name}"], r2=df[f"free_high_{sgy_name}"])
    return df

def add_delta_straddle_sgy(df):
    add_delta_strangle_sgy(df) #Passed without 'OTM_moneyness'
    return df

def add_self_financed_stock_sgy(df):
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        df.loc[ticker_mask, 'ticker_change_free'] = (
                df.loc[ticker_mask, 'close'] -
                df.loc[ticker_mask, 'close'].shift(1) * (1 + df.loc[ticker_mask, 'RF'])
        )
    return df