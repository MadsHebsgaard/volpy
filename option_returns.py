
import pandas as pd
import numpy as np

import load_clean_lib
from global_settings import *
from vol_strat_lib import T_day_interpolation

def Calculate_CashFlow(df, current_price, next_price):
    return (next_price - (1+df["RF"]) * current_price).shift(1)

def Calculate_return(df, current_price, next_price):
    ret = (next_price / (current_price * (1 + df["RF"])) - 1).shift(1)
    # ret = ((next_price - (1+df["RF"]) * current_price) / current_price).shift(1)
    # ret = Calculate_CashFlow(df, current_price, next_price) / current_price.shift(-1)
    return ret


def prepare_for_sgys(df, curr_OTMs):
    df = add_stock_sgy(df)
    df = add_stock_adj_close(df)
    df = D_PC_option_prices(df, curr_OTMs)
    return df


def add_stock_adj_close(df):
    def compute_adj_close(group):
        group = group.sort_values("date")
        # Get the first non-NaN closing price.
        first_valid_close_idx = group["close"].first_valid_index()
        if first_valid_close_idx is None:
            return pd.Series(np.nan, index=group.index)
        start = group.loc[first_valid_close_idx, "close"]

        # Copy returns and set the first non-NaN return to 0.
        r = group["return"].copy()
        first_valid_return_idx = r.first_valid_index()
        if first_valid_return_idx is not None:
            r.loc[first_valid_return_idx] = 0

        # Compute the cumulative product on the modified returns.
        factor = (1 + r).cumprod()

        return start * factor

    # Apply to each ticker
    # df["adj_close"] = df.groupby("ticker").apply(lambda g: compute_adj_close(g)).reset_index(level=0, drop=True)

    # this currently comes back as a DataFrame if your returned Series has a name
    tmp = df.groupby("ticker") \
        .apply(compute_adj_close) \
        .reset_index(level=0, drop=True)

    # squeeze it down (turns a single‑col DataFrame into a Series)
    df["adj_close"] = tmp.squeeze()

    return df



def D_PC_option_prices(df, OTMs):
    ATM_str = "ATM"
    OTM_str = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    moneynesses = [ATM_str] + OTM_str

    T1 = df["low days"]
    T2 = df["high days"]

    for moneyness in moneynesses:
        for put_call in ["put", "call"]:
            for low_high in ["low", "high"]:
                df[f"D_{low_high}_{put_call}_{moneyness}_price"] = (
                    df[f"{low_high}_{put_call}_{moneyness}_price"]
                )

                df[f"D_{low_high}_{put_call}_{moneyness}_price_next"] = (
                    df[f"{low_high}_{put_call}_{moneyness}_price_next"] -
                    df[f"{low_high}_{put_call}_{moneyness}_delta"] * df[f"adj_close"] * df["r_stock"].shift(-1)
                )

            for time in ["", "_next"]:
                for D_str in ["D_", ""]:
                    low_price_time = df[f"{D_str}low_{put_call}_{moneyness}_price{time}"]
                    high_price_time = df[f"{D_str}high_{put_call}_{moneyness}_price{time}"]
                    df[f"{D_str}30_{put_call}_{moneyness}_price{time}"] = T_day_interpolation(T1=T1, T2=T2, r1=low_price_time, r2=high_price_time)
    return df

def add_stock_sgy(df):
    df["r_stock"] = df["return"] - df["RF"]
    return df

    # # identical to this, however works when stocksplits
    # for ticker in df['ticker'].unique():
    #     ticker_mask = df['ticker'] == ticker
    #     next_price = df.loc[ticker_mask, 'adj_close'].shift(-1)
    #     current_price = df.loc[ticker_mask, 'adj_close']
    #
    #     df.loc[ticker_mask, f'CF_stock'] = Calculate_CashFlow(df, current_price, next_price)
    #     df.loc[ticker_mask, f'r_stock'] = Calculate_return(df, current_price, next_price)
    #     return df


def add_put_and_call_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    ATM_str = "ATM"
    OTM_str = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    moneynesses = [ATM_str] + OTM_str
    new_columns = {} # Dictionary to collect new columns

    for put_call in ["put", "call"]:
        for D_str in ["D_", ""]:
            for moneyness in moneynesses:
                for low_high in HL30_list: #["low", "high", "30"]
                    next_price = df[f"{D_str}{low_high}_{put_call}_{moneyness}_price_next"]
                    current_price = df[f"{D_str}{low_high}_{put_call}_{moneyness}_price"]

                    CF = Calculate_CashFlow(df, current_price, next_price)
                    ret = Calculate_return(df, current_price, next_price)

                    new_columns[f"CF_{D_str}{low_high}_{put_call}_{moneyness}"] = CF
                    new_columns[f"r_{D_str}{low_high}_{put_call}_{moneyness}"] = ret
    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    return df

def add_straddle_strangle_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    ATM_str = "ATM"
    OTM_str = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    new_columns = {} # Dictionary to collect new columns

    moneynesses = [ATM_str] + OTM_str
    sgy_names = ["straddle"] + [f"strangle_{round(OTM * 100)}%" for OTM in OTMs]

    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, moneyness, next_str):
        price = df[f"{D_str}{low_high}_put_{moneyness}_price{next_str}"] + df[f"{D_str}{low_high}_call_{moneyness}_price{next_str}"]
        return price

    for D_str in ["D_", ""]:
        for moneyness, sgy_name in zip(moneynesses, sgy_names):
            for low_high in HL30_list: #["low", "high", "30"]:
                next_price = calculate_price(low_high, D_str, moneyness, next_str="_next")
                current_price = calculate_price(low_high, D_str, moneyness, next_str="")

                CF = Calculate_CashFlow(df, current_price, next_price)
                ret = Calculate_return(df, current_price, next_price)

                new_columns[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
                new_columns[f"r_{D_str}{low_high}_{sgy_name}"] = ret
    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    return df

def add_butterfly_spread_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    ATM_str = "ATM"
    OTM_strs = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    sgy_names = [f"butterfly_spread_{round(OTM * 100)}%" for OTM in OTMs]
    new_columns = {} # Dictionary to collect new columns

    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, OTM_str, next_str):
        ATM_price = df[f"{D_str}{low_high}_put_{ATM_str}_price{next_str}"] + df[f"{D_str}{low_high}_call_{ATM_str}_price{next_str}"]
        OTM_price = df[f"{D_str}{low_high}_put_{OTM_str}_price{next_str}"] + df[f"{D_str}{low_high}_call_{OTM_str}_price{next_str}"]
        return ATM_price - OTM_price

    for D_str in ["D_", ""]:
        for OTM_str, sgy_name in zip(OTM_strs, sgy_names):
            for low_high in HL30_list:  # ["low", "high", "30"]
                next_price = calculate_price(low_high, D_str, OTM_str, next_str = "_next")
                current_price = calculate_price(low_high, D_str, OTM_str, next_str = "")

                CF = Calculate_CashFlow(df, current_price, next_price)
                ret = Calculate_return(df, current_price, next_price)

                new_columns[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
                new_columns[f"r_{D_str}{low_high}_{sgy_name}"] = ret

    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    return df


def add_condor_strangle_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    sgy_name_base = f"condor_strangle"
    new_columns = {} # Dictionary to collect new columns

    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, less_OTM, more_OTM, next_str):
        price_less_OTM = df[f"{D_str}{low_high}_put_{less_OTM}_price{next_str}"] + df[f"{D_str}{low_high}_call_{less_OTM}_price{next_str}"]
        price_more_OTM = df[f"{D_str}{low_high}_put_{more_OTM}_price{next_str}"] + df[f"{D_str}{low_high}_call_{more_OTM}_price{next_str}"]
        return price_less_OTM - price_more_OTM

    for D_str in ["D_", ""]:
        for less_OTM in OTMs:
            for more_OTM in OTMs:
                if less_OTM >= more_OTM:
                    continue
                less_OTM_str = f"OTM_{round(less_OTM * 100)}%"
                more_OTM_str = f"OTM_{round(more_OTM * 100)}%"
                sgy_name = sgy_name_base + f"_{round(less_OTM * 100)}%_{round(more_OTM * 100)}%"

                for low_high in HL30_list: #["low", "high", "30"]
                    next_price = calculate_price(low_high, D_str, less_OTM_str, more_OTM_str, next_str = "_next")
                    current_price = calculate_price(low_high, D_str, less_OTM_str, more_OTM_str, next_str = "")

                    CF = Calculate_CashFlow(df, current_price, next_price)
                    ret = Calculate_return(df, current_price, next_price)

                    new_columns[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
                    new_columns[f"r_{D_str}{low_high}_{sgy_name}"] = ret



    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    return df



# Other strategies for fun (very easy to make)
def add_stacked_straddle_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    sgy_name_base = f"stacked_straddle"

    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, less_OTM, more_OTM, next_str):
        price_less_OTM = df[f"{D_str}{low_high}_put_{less_OTM}_price{next_str}"] + df[f"{D_str}{low_high}_call_{less_OTM}_price{next_str}"]
        price_more_OTM = df[f"{D_str}{low_high}_put_{more_OTM}_price{next_str}"] + df[f"{D_str}{low_high}_call_{more_OTM}_price{next_str}"]
        return price_less_OTM + price_more_OTM

    for D_str in ["D_", ""]:
        for less_OTM in OTMs:
            for more_OTM in OTMs:
                if less_OTM >= more_OTM:
                    continue
                less_OTM_str = f"OTM_{round(less_OTM * 100)}%"
                more_OTM_str = f"OTM_{round(more_OTM * 100)}%"
                sgy_name = sgy_name_base + f"_{round(less_OTM * 100)}%_{round(more_OTM * 100)}%"

                for low_high in HL30_list: #["low", "high", "30"]
                    next_price = calculate_price(low_high, D_str, less_OTM_str, more_OTM_str, next_str = "_next")
                    current_price = calculate_price(low_high, D_str, less_OTM_str, more_OTM_str, next_str = "")

                    CF = Calculate_CashFlow(df, current_price, next_price)
                    ret = Calculate_return(df, current_price, next_price)

                    df[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
                    df[f"r_{D_str}{low_high}_{sgy_name}"] = ret
    return df


def add_full_stacked_straddle_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    sgy_name = f"full_stacked_straddle"
    ATM_str = "ATM"
    OTM_str = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    OA_TM_str = [ATM_str] + OTM_str


    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, OTM_strs, next_str):
        price = sum([df[f"{D_str}{low_high}_put_{OTM_str}_price{next_str}"] + df[f"{D_str}{low_high}_call_{OTM_str}_price{next_str}"] for OTM_str in OTM_strs])
        return price

    for D_str in ["D_", ""]:
        for low_high in HL30_list: #["low", "high", "30"]
            next_price = calculate_price(low_high, D_str, OA_TM_str, next_str = "_next")
            current_price = calculate_price(low_high, D_str, OA_TM_str, next_str = "")

            CF = Calculate_CashFlow(df, current_price, next_price)
            ret = Calculate_return(df, current_price, next_price)

            df[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
            df[f"r_{D_str}{low_high}_{sgy_name}"] = ret
    return df


def add_full_stacked_strangle_sgy(df, OTMs, HL30_list = ["low", "high", "30"]):
    sgy_name = f"full_stacked_strangle"
    OTM_str = [f"OTM_{round(OTM * 100)}%" for OTM in OTMs]
    OA_TM_str = OTM_str


    # Function to calculate the portfolio price for a given scenario
    def calculate_price(low_high, D_str, OTM_strs, next_str):
        price = sum([df[f"{D_str}{low_high}_put_{OTM_str}_price{next_str}"] + df[f"{D_str}{low_high}_call_{OTM_str}_price{next_str}"] for OTM_str in OTM_strs])
        return price

    for D_str in ["D_", ""]:
        for low_high in HL30_list: #["low", "high", "30"]
            next_price = calculate_price(low_high, D_str, OA_TM_str, next_str = "_next")
            current_price = calculate_price(low_high, D_str, OA_TM_str, next_str = "")

            CF = Calculate_CashFlow(df, current_price, next_price)
            ret = Calculate_return(df, current_price, next_price)

            df[f"CF_{D_str}{low_high}_{sgy_name}"] = CF
            df[f"r_{D_str}{low_high}_{sgy_name}"] = ret
    return df