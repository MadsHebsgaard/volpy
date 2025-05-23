
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

    sum_df.loc[sum_df["SW_0_30"] <= 1e-6, "SW_0_30"] = 1e-6
    return sum_df, od_raw


def import_orpy_sum(om_folder):
    om_folder = load_clean_lib.volpy_output_dir(om_folder)
    time_type = days_type()
    sum_df = pd.read_csv(f"{om_folder}/{time_type}summary_dly.csv")
    sum_df["date"] = pd.to_datetime(sum_df["date"])
    sum_df[f"return_next"] = sum_df[f"return"].shift(-1)


    orpy_df = pd.read_csv(f"{om_folder}/{time_type}df_orpy.csv")
    orpy_df["date"] = pd.to_datetime(orpy_df["date"])

    # Fix for double assets (for some reason the wrong ones have close == 0)
    sum_df = sum_df[(sum_df["open"] > 0) & (sum_df["close"] > 0)].copy()
    orpy_df = orpy_df[(orpy_df["open"] > 0) & (orpy_df["close"] > 0)].copy()

    sum_df.loc[sum_df["SW_0_30"] <= 1e-6, "SW_0_30"] = 1e-6
    orpy_df.loc[orpy_df["SW_0_30"] <= 1e-6, "SW_0_30"] = 1e-6

    return orpy_df, sum_df


def create_od_hl(od_raw, sum_df, price_type, IV_type):
    days_var = days_type() + "days"

    od_raw['price'] = od_raw[price_type] # todo: Can be removed? what is 'od_raw['price']' used for?

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
    # od_hl["IV"] = od_hl[f"IV_{IV_type}"]

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


def T_day_interpolation_old(T1, T2, r1, r2):
    if days_type() == "t_":
        T = 21
    elif days_type() == "c_":
        T = 30
    else:
        print("days_type() is neither t_ nor c_")
        T = np.nan

    return np.where(T1 == T2, (r1+r2)/2,
                    (r1 * np.abs(T2 - T) + r2 * np.abs(T1 - T)) / (np.abs(T2 - T) + np.abs(T1 - T)))

# (r1 * np.abs(T2 - T) + r2 * np.abs(T1 - T)) / (np.abs(T2 - T) + np.abs(T1 - T))

def T_day_interpolation_CW(T1, T2, r1, r2, t = 0):
    if days_type() == "t_":
        T = 21
    elif days_type() == "c_":
        T = 30
    else:
        print("days_type() is neither t_ nor c_")
        T = np.nan
    theta = np.where(
        T1 == T2,
         0.5,
         (T1 - t)*(T2 - T)/((T2-T1)*(T-t))
    )
    SW_t_T = (r1 * theta + r2 * (1-theta))
    return SW_t_T

def T_day_interpolation_linear(T1, T2, r1, r2, t = 0):
    if days_type() == "t_":
        T = 21
    elif days_type() == "c_":
        T = 30
    else:
        print("days_type() is neither t_ nor c_")
        T = np.nan

    theta = np.where(
        T1 == T2,
        0.5,
        (T2-(T-t)) / (T2 - T1)
    )
    return r1 * theta + r2 * (1-theta)

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


def create_option_sgys(sum_df, od_raw, price_type="mid", IV_type="om", OTMs=[0.05, 0.15], save_files=True, om_folder = None):
    import option_returns as orpy

    # load data
    od_hl = create_od_hl(od_raw=od_raw, sum_df=sum_df, price_type=price_type, IV_type=IV_type)
    sum_df = add_F_to_sum_df(od_hl=od_hl, sum_df=sum_df)
    sum_df = add_ATM_options_to_sum_df(sum_df=sum_df, od_hl=od_hl, od_raw=od_raw, OTMs=OTMs)

    # add strategies
    HL30_list = ["30"]  # ["low", "high", "30"], ["30"] # here 30 represents 21

    df_orpy = sum_df.copy()
    df_orpy = orpy.prepare_for_sgys(df_orpy, OTMs)

    Strategies = [orpy.add_put_and_call_sgy, orpy.add_straddle_strangle_sgy, orpy.add_butterfly_spread_sgy,
                  orpy.add_condor_strangle_sgy, orpy.add_stacked_straddle_sgy, orpy.add_full_stacked_straddle_sgy,
                  orpy.add_full_stacked_strangle_sgy]
    for add_sgy in Strategies:
        df_orpy = add_sgy(df_orpy, OTMs, HL30_list)

    # save
    if save_files:
        if om_folder is not None:
            output_dir = load_clean_lib.volpy_output_dir(om_folder)
            time_type = days_type()
            df_orpy.to_csv(f"{output_dir}/{time_type}df_orpy.csv")
            sum_df.to_csv(f"{output_dir}/{time_type}sum_df_big.csv")

    return sum_df, df_orpy


import re
from tqdm import tqdm


def add_IVs_to_sum_df(sum_df, od_hl, IV_type="om"):
    iv_colname = f"IV_{IV_type}"

    # Brugte optionid-kolonner (du har bekræftet navne tidligere)
    used_optionid_cols = [
        "low_call_ATM_optionid",
        "high_call_ATM_optionid",
        "low_put_ATM_optionid",
        "high_put_ATM_optionid"
    ]

    # Lav en lookup dictionary én gang
    iv_lookup_dict = od_hl.set_index("optionid")[iv_colname].to_dict()

    # Brug map til hver kolonne (ingen merge = hurtigt og memory-sikkert)
    for col in used_optionid_cols:
        new_iv_col = col.replace("optionid", "IV")
        sum_df[new_iv_col] = sum_df[col].map(iv_lookup_dict)

    return sum_df





def create_option_sgys_from_chunks(sum_df, om_folder, price_type="mid", IV_type="om", OTMs=[0.05, 0.15], save_files_each_chunk=True, save_files=True):
    import option_returns as orpy

    output_dir = load_clean_lib.volpy_output_dir(om_folder)
    time_type = days_type()

    # Find alle od_raw chunk-filer (fx od_raw_1996.csv, ...)
    od_files = sorted(output_dir.glob(f"{time_type}od_raw_*.csv"))

    all_sgy_df = []
    all_sum_df_chunks = []  # <-- til opdateret sum_df

    # Progress bar med tqdm
    for od_file in tqdm(od_files, desc="Processing option chunks", unit="file"):
        od_raw = pd.read_csv(od_file, parse_dates=["date", "exdate"])

        # STEP 1: Create od_hl
        od_hl = create_od_hl(od_raw=od_raw, sum_df=sum_df, price_type=price_type, IV_type=IV_type)

        # Filtrér kun de datoer som er i den pågældende od_raw chunk
        dates_in_chunk = od_raw["date"].unique()
        sum_df_chunk = sum_df[sum_df["date"].isin(dates_in_chunk)].copy()
        sum_df_chunk = add_F_to_sum_df(od_hl=od_hl, sum_df=sum_df_chunk)        
        sum_df_chunk = add_ATM_options_to_sum_df(sum_df=sum_df_chunk, od_hl=od_hl, od_raw=od_raw, OTMs=OTMs)

        # STEP 3: Forbered og tilføj strategier
        df_orpy = orpy.prepare_for_sgys(sum_df_chunk.copy(), OTMs)
        HL30_list = ["30"]
        for add_sgy in [
            orpy.add_put_and_call_sgy,
            orpy.add_straddle_strangle_sgy # ,
            # orpy.add_butterfly_spread_sgy,
            # orpy.add_condor_strangle_sgy,
            # orpy.add_stacked_straddle_sgy,
            # orpy.add_full_stacked_straddle_sgy,
            # orpy.add_full_stacked_strangle_sgy
        ]:
            df_orpy = add_sgy(df_orpy, OTMs, HL30_list)

        
        # this is the correlation strategy
        sum_df_chunk = add_IVs_to_sum_df(sum_df_chunk, od_hl, IV_type=IV_type)


        # Gem og tilføj
        if save_files_each_chunk:
            stem = od_file.stem
            stamp = re.search(r'(\d{4}(?:-\d{2})?)$', stem).group(1)
            df_orpy.to_csv(f"{output_dir}/{time_type}df_orpy_{stamp}.csv", index=False)

        all_sgy_df.append(df_orpy)
        all_sum_df_chunks.append(sum_df_chunk)

    # Merge updated sum_df chunks (for alle datoer)
    sum_df_all = pd.concat(all_sum_df_chunks).drop_duplicates(subset=["ticker", "date"]).sort_values("date").reset_index(drop=True)
    df_orpy_all = pd.concat(all_sgy_df).sort_values("date").reset_index(drop=True)

    if save_files: df_orpy_all.to_csv(f"{output_dir}/{time_type}df_orpy.csv", index=False)

    return sum_df_all, df_orpy_all


