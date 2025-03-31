import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import sys
import time
from itertools import islice

# from global_settings import days_type
import importlib
import global_settings
importlib.reload(global_settings)
days_type = global_settings.days_type
clean_t_days = global_settings.clean_t_days


def load_option_data(file_path):
    import os
    columns_to_load = [
        "ticker",
        "optionid",
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "impl_volatility",
        # "delta",
        # "index_flag",
        # "forward_price",
        # "exercise_style",
        # "secid"
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)


    df.rename(columns={"strike_price": "K"}, inplace=True)
    df.rename(columns={"best_bid": "bid"}, inplace=True)
    df.rename(columns={"best_offer": "ask"}, inplace=True)
    df.rename(columns={"impl_volatility": "IV_om"}, inplace=True)

    # Format columns
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d', errors='coerce')
    df['cp_flag'] = df['cp_flag'].astype('string')
    df['ticker'] = df['ticker'].astype('string')
    df['optionid'] = df['optionid'].astype(int)

    # df['exercise_style'] = df['exercise_style'].astype('string')
    df['K'] /= 1000

    # load valid dates
    unique_calendar_days = df["date"].drop_duplicates().sort_values()  # [df["ticker"] == "SPX"]
    valid_dates_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), "dates.csv")
    valid_dates = pd.read_csv(valid_dates_path, usecols=["DATE"], parse_dates=["DATE"])
    valid_dates.rename(columns={"DATE": "date"}, inplace=True)
    valid_dates = valid_dates[valid_dates["date"] >= unique_calendar_days.iloc[0]]
    valid_dates = valid_dates["date"]

    # Create a mapping of date to trading day number
    trading_day_map = {date: n for n, date in enumerate(valid_dates)}
    dates = valid_dates.values.astype("datetime64[ns]")
    exdates = df["exdate"].values.astype("datetime64[ns]")
    indices = np.searchsorted(dates, exdates, side="right") - 1
    indices = np.where(indices < 0, np.nan, indices)

    total_trading_days = unique_calendar_days.nunique()
    total_calendar_days = (unique_calendar_days.max() - unique_calendar_days.min()).days + 1

    average_calendar_days_per_year = 365
    cal_years_total = total_calendar_days / average_calendar_days_per_year
    average_trading_days_per_year = total_trading_days / cal_years_total
    # print("average_trading_days_per_year", average_trading_days_per_year)

    # Add the n_trading_day column to the DataFrame
    df["n_trading_day"] = df["date"].map(trading_day_map)
    df["n_trading_day_exdate"] = indices

    # df["n_trading_day_exdate"] = df["exdate"].apply(get_trading_day)
    df["t_days"] = df["n_trading_day_exdate"] - df["n_trading_day"]
    df["t_TTM"] = df["t_days"] / average_trading_days_per_year #todo: do this on yearly basis to fix if average trading days change throughout years

    # Calendar (_cal)
    df["c_days"] = (df["exdate"] - df["date"]).dt.days
    df["c_TTM"] = df["c_days"] / average_calendar_days_per_year

    df = df.sort_values(by="date")

    # Trading (_trd)
    # df["TTM_trd"] = (df["n_trading_day_exdate"] - df["n_trading_day"]) / round(average_trading_days_per_year)

    return df


# def load_implied_volatility(file_path):
#     columns_to_load = [
#         "secid",
#         "ticker",
#         "date",
#         "days",
#         "delta",
#         "impl_volatility",
#         "cp_flag",
#     ]
#
#     # Load only the specified columns
#     df = pd.read_csv(file_path, usecols=columns_to_load)
#
#     # Format columns
#     df['ticker'] = df['ticker'].astype('string')
#     df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
#     df['cp_flag'] = df['cp_flag'].astype('string')
#     return df
#
#
# def load_realized_volatility(file_path):
#     columns_to_load = [
#         "secid",
#         "ticker",
#         "date",
#         "days",
#         "volatility"
#     ]
#
#     # Load only the specified columns
#     df = pd.read_csv(file_path, usecols=columns_to_load)
#
#     # Format columns
#     df['ticker'] = df['ticker'].astype('string')
#     df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
#     return df


def load_forward_price(file_path):
    columns_to_load = [
        "secid",
        "ticker",
        "date",
        "expiration",
        "forwardprice" #ForwardPrice
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)

    # df.rename(columns={"ForwardPrice": "forwardprice"}, inplace=True)

    # Format columns
    df['ticker'] = df['ticker'].astype('string')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y-%m-%d', errors='coerce')
    df["c_days"] = (df['expiration'] - df['date']).dt.days

    # Create a mapping of date to trading day number
    unique_calendar_days = df["date"].drop_duplicates().sort_values()  # [df["ticker"] == "SPX"]
    trading_day_map = {date: n for n, date in enumerate(unique_calendar_days)}

    df["n_trading_day"] = df["date"].map(trading_day_map) # todo: make these variables instead, not to clog up dataframe
    df["n_trading_day_exdate"] = df["expiration"].map(trading_day_map)
    df["t_days"] = df["n_trading_day_exdate"] - df["n_trading_day"]

    return df



def load_returns_and_price(file_path):
    # Read the CSV file into a DataFrame
    returns_and_prices = pd.read_csv(file_path)

    # Convert 'date' column to datetime format
    returns_and_prices['date'] = pd.to_datetime(returns_and_prices['date'])

    # remove na
    returns_and_prices = returns_and_prices.dropna(subset=["return"])

    # Ensure the DataFrame is sorted
    returns_and_prices = returns_and_prices.sort_values(by=["ticker", "date"])

    # # Filtrer data for at inkludere kun rækker med datoer mellem first_day og last_day
    # first_day = pd.to_datetime("1996-01-04")
    # last_day = pd.to_datetime("2003-02-28") + pd.Timedelta(days=30)
    # returns_and_prices = returns_and_prices[
    #     (returns_and_prices['date'] >= first_day) & (returns_and_prices['date'] <= last_day)
    # ]
    return returns_and_prices


def load_ZC_yield_curve(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['rate'] /= 100
    df.rename(columns={"days": "c_days"}, inplace=True)
    df['c_days'] = df['c_days'].astype(int)
    return df



def add_FW_to_od(od, FW):
    # Pre-group FW by date and ticker, and store sorted arrays for interpolation

    days_var = days_type() + "days"
    days_var = "c_days" #todo: maybe should be able to use trading days (don't think so though)

    FW_grouped = {}
    for key, group in FW.groupby(["date", "ticker"]):
        group_sorted = group.sort_values("c_days")
        FW_grouped[key] = (group_sorted[days_var].values, group_sorted["forwardprice"].values)

    def interpolate_group(group):
        # Each group in od corresponds to one (date, ticker) pair
        key = (group["date"].iloc[0], group["ticker"].iloc[0])
        if key not in FW_grouped:
            raise ValueError(f"No FW data for key: {key}")
        x, y = FW_grouped[key]

        # Check that all target days are within the FW range (to mimic your error condition)
        if group[days_var].min() < x[0] or group[days_var].max() > x[-1]:
            raise ValueError(f"Not enough data points to interpolate for some days in group {key}")

        # Use numpy's vectorized interpolation for the entire group
        group.loc[:, "F"] = np.interp(group[days_var], x, y)
        return group

    # Apply the interpolation function to each group in od
    od = od.groupby(["date", "ticker"], group_keys=False).apply(interpolate_group)
    return od


def calc_realized_var(returns_and_prices, first_day, last_day):
    # Bestem T ud fra days_type()

    if days_type() == "c_":
        T = 30  # Kalenderdage
    elif days_type() == "t_":
        T = 21  # Trading days: brug altid de næste 21 observationer

    # Filter data for perioden fra first_day til last_day + 30 dage (skal være bredt nok til at dække T observationer)
    last_day_ret = last_day + pd.Timedelta(days=30)
    returns_and_prices = returns_and_prices[
        (returns_and_prices['date'] >= first_day) & (returns_and_prices['date'] <= last_day_ret)
    ].copy()

    # Beregn squared daily returns
    returns_and_prices["squared_return"] = returns_and_prices["return"] ** 2

    # Beregn log-returns (hvis nødvendigt)
    returns_and_prices['log_return'] = np.log(1 + returns_and_prices['return'])
    returns_and_prices['squared_log_return'] = returns_and_prices['log_return'] ** 2

    # Definer en funktion til at summe over de næste T observationer (for trading days) eller T kalenderdage
    def sum_next_T_days(group):
        group = group.sort_values(by='date')
        rv_list = []
        active_days_list = []
        
        # if days_type() == "t_":
        #     # Trading days: brug præcis de næste T observationer – hvis ikke nok, returnér eksisterende sum
        #     for idx, row in group.iterrows():
        #         # window = group.iloc[idx:(idx + min(T, len(group) - idx))]
        #         window = group.iloc[idx:(idx + T)]
        #         rv_sum = window['squared_return'].sum()
        #         active_days = window.shape[0]
        #         rv_list.append(rv_sum)
        #         active_days_list.append(active_days)


        if days_type() == "t_":
            # Trading days: brug de næste T trading-dage baseret på dato – med mask ligesom i 'c_'
            dates = group["date"].tolist()
            
            for _, row in group.iterrows():
                current_date = row["date"]
                # future_dates = [d for d in dates if d >= current_date][:T]
                future_dates = list(islice((d for d in dates if d >= current_date), T))
                mask = group["date"].isin(future_dates)
                window = group.loc[mask]
                rv_sum = window["squared_return"].sum()
                active_days = window.shape[0]
                rv_list.append(rv_sum)
                active_days_list.append(active_days)


        elif days_type() == "c_":
            # Hvis det er kalenderdage, brug tidsbaseret slicing med pd.Timedelta(days=T)
            for _, row in group.iterrows():
                current_date = row['date']
                end_date = current_date + pd.Timedelta(days=T)
                mask = (group['date'] >= current_date) & (group['date'] < end_date)
                window = group.loc[mask]
                rv_sum = window['squared_return'].sum()
                active_days = window.shape[0]
                rv_list.append(rv_sum)
                active_days_list.append(active_days)
        
        result = pd.DataFrame({
            'RV_unscaled': rv_list,
            'N_tradingdays': active_days_list
        }, index=group.index)
        return result

    result_df = (
        returns_and_prices.groupby('ticker')
        .apply(sum_next_T_days)
        .reset_index(level=0, drop=True)
    )

    # Annualiser RV_unscaled: divider med antallet af dage (observationer) i vinduet og gang med 252
    returns_and_prices.loc[:, 'RV'] = result_df['RV_unscaled'] * (252 / result_df['N_tradingdays'])
    returns_and_prices.loc[:, 'N_tradingdays'] = result_df['N_tradingdays']

    # Filtrer igen for perioden first_day til last_day
    returns_and_prices = returns_and_prices[
        (returns_and_prices['date'] >= first_day) & (returns_and_prices['date'] <= last_day)
    ].copy()

    # Fjern midlertidige kolonner
    returns_and_prices.drop(['log_return', 'squared_log_return'], axis=1, inplace=True)

    return returns_and_prices




from functools import partial
import concurrent.futures


def compute_rates_for_date(group, ZCY_curves):

    # days_var = days_type() + "days"
    days_var = "c_days"

    current_date = group["date"].iloc[0]
    # Compute the yield curve for the current date once
    ZCY_date = interpolate_yield_curve_between_dates(ZCY_curves, current_date)
    ZCY_date = ZCY_date.sort_values(days_var)
    x = ZCY_date[days_var].values
    y = ZCY_date["rate"].values

    # Get the target 'days' values for this group
    target_days = group[days_var].values
    interpolated_rates = np.empty_like(target_days, dtype=float)

    # Masks for in-range, below-range, and above-range target days
    mask_in_range = (target_days >= x[0]) & (target_days <= x[-1])
    mask_below = target_days < x[0]
    mask_above = target_days > x[-1]

    # For in-range values, use numpy's vectorized interpolation
    interpolated_rates[mask_in_range] = np.interp(target_days[mask_in_range], x, y)

    # For target days below the minimum, extrapolate using the first two points
    if np.any(mask_below):
        x0, x1 = x[0], x[1]
        y0, y1 = y[0], y[1]
        interpolated_rates[mask_below] = y0 + (y1 - y0) * (target_days[mask_below] - x0) / (x1 - x0)

    # For target days above the maximum, extrapolate using the last two points
    if np.any(mask_above):
        x0, x1 = x[-2], x[-1]
        y0, y1 = y[-2], y[-1]
        interpolated_rates[mask_above] = y0 + (y1 - y0) * (target_days[mask_above] - x0) / (x1 - x0)

    group["r"] = interpolated_rates
    return group


def add_r_to_od_parallel(od, ZCY_curves, max_workers=None):
    # Split the DataFrame into groups based on 'date'
    groups = [group for _, group in od.groupby("date", group_keys=False)]

    # Use ProcessPoolExecutor to parallelize processing of each group
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function to include ZCY_curves as an argument
        compute_func = partial(compute_rates_for_date, ZCY_curves=ZCY_curves)
        results = list(executor.map(compute_func, groups))

    # Recombine the groups into a single DataFrame
    return pd.concat(results)


def add_r_to_od(od, ZCY_curves):

    days_var = days_type() + "days"

    def compute_rates_for_date(group):
        current_date = group["date"].iloc[0]
        # Compute the yield curve for the current date once
        ZCY_date = interpolate_yield_curve_between_dates(ZCY_curves, current_date)
        ZCY_date = ZCY_date.sort_values(days_var)
        x = ZCY_date[days_var].values
        y = ZCY_date["rate"].values

        # Get the target 'days' values for this group
        target_days = group[days_var].values
        interpolated_rates = np.empty_like(target_days, dtype=float)

        # Mask for in-range, below-range, and above-range target days
        mask_in_range = (target_days >= x[0]) & (target_days <= x[-1])
        mask_below = target_days < x[0]
        mask_above = target_days > x[-1]

        # For in-range values, use numpy's vectorized interpolation
        interpolated_rates[mask_in_range] = np.interp(target_days[mask_in_range], x, y)

        # For target days below the minimum, extrapolate using the first two points
        if np.any(mask_below):
            x0, x1 = x[0], x[1]
            y0, y1 = y[0], y[1]
            interpolated_rates[mask_below] = y0 + (y1 - y0) * (target_days[mask_below] - x0) / (x1 - x0)

        # For target days above the maximum, extrapolate using the last two points
        if np.any(mask_above):
            x0, x1 = x[-2], x[-1]
            y0, y1 = y[-2], y[-1]
            interpolated_rates[mask_above] = y0 + (y1 - y0) * (target_days[mask_above] - x0) / (x1 - x0)

        group["r"] = interpolated_rates
        return group

    # Process each date-group in od: compute yield curve once and vectorize interpolation
    od = od.groupby("date", group_keys=False).apply(compute_rates_for_date)
    return od

def vectorized_iv(F, K, T, market_price, cp_flag, tol=1e-6, max_iter=100):
    sigma = np.full_like(F, 0.2, dtype=float)
    for _ in range(max_iter):
        # Clip sigma to remain within a reasonable range
        sigma = np.clip(sigma, 1e-6, 5.0)
        sqrt_T = np.sqrt(T)

        # Compute d1 and d2 safely; note that for T==0 we set d1=0
        d1 = np.where(T > 0, (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T), 0)
        d2 = d1 - sigma * sqrt_T

        # Compute the Black option price vectorized over F, K, T and sigma
        price = np.where(
            T <= 0,
            np.where(cp_flag == "C", np.maximum(0.0, F - K), np.maximum(0.0, K - F)),
            np.where(cp_flag == "C",
                     F * norm.cdf(d1) - K * norm.cdf(d2),
                     K * norm.cdf(-d2) - F * norm.cdf(-d1))
        )

        # Compute Vega (the derivative with respect to sigma)
        vega = np.where(T <= 0, 0.0, F * norm.pdf(d1) * sqrt_T)
        diff = price - market_price

        # If converged for all, break out
        if np.all(np.abs(diff) < tol):
            break

        # Update sigma only where vega is nonzero
        mask = vega != 0
        update = np.zeros_like(sigma)
        update[mask] = diff[mask] / vega[mask]

        # Clip the update step to avoid huge jumps that can cause overflow
        update = np.clip(update, -1, 1)
        sigma[mask] = sigma[mask] - update[mask]

    # Assign NaN to any negative volatility values
    sigma = np.where(sigma < 0, np.nan, sigma)
    return sigma


def add_bid_mid_ask_IV(od, IV_type):

    TTM_var = days_type() + "TTM"

    # Compute IV_bid vectorized
    if IV_type == "om":
        return od["IV_om"]
    else:
        market_price = od[IV_type].values

    # Prepare the inputs from your DataFrame
    T = od[TTM_var].values
    F = od["F"].values
    K = od["K"].values
    cp_flag = od["cp_flag"].values  # Assumes an array of "C" and "P"

    IV = vectorized_iv(F, K, T, market_price, cp_flag)
    return IV

# def process_group_activity_summary(group):

#     days_var = days_type() + "days"

#     if days_type() == "t_" and clean_t_days():
#         x = 1
#     else:
#         x = 0

#     # Extract the current date and ticker from the group
#     current_date = group["date"].iloc[0]
#     current_ticker = group["ticker"].iloc[0]
#     summary = {}

#     # Initialize the new columns to False
#     group["low"] = False
#     group["high"] = False

#     # Get sorted unique days for this (date, ticker) pair
#     unique_days = np.sort(group[days_var].unique())
#     summary["#days"] = len(unique_days)

#     # Inactive if there are fewer than 2 unique days
#     if len(unique_days) < 2:
#         summary["Active"] = False
#         summary["Inactive reason"] = "unique(_days) < 2"
#         return group, summary

#     # Inactive if the minimum day is > 90
#     if unique_days[0] > 90 - x*90*0.3: #todo: change to above something like x_TTM > 0.25
#         summary["Active"] = False
#         summary["Inactive reason"] = "min days > 90"
#         return group, summary

#     # Select two lowest days above 7 days.
#     # If the smallest day is <= 8 and at least 3 unique days exist, skip it.
#     low_2_days = list(unique_days[:2])
#     if low_2_days[0] <= 8 - x*2: #todo: change to above something like <= 6 || <= 8 depending on 'days_type'
#         if len(unique_days) < 3:
#             summary["Active"] = False
#             summary["Inactive reason"] = "min days <= 8 & len < 3"
#             return group, summary
#         low_2_days = list(unique_days[1:3])

#     summary["low days"] = low_2_days[0]
#     summary["high days"] = low_2_days[1]

#     # Check unique strike counts for each selected day; require at least 3
#     active = True
#     for day, label in zip(low_2_days, ["low", "high"]):
#         num_strikes = group.loc[group[days_var] == day, "K"].nunique()
#         summary[f"{label} #K"] = num_strikes
#         if num_strikes < 3:
#             active = False
#             summary["Active"] = False
#             summary["Inactive reason"] = "unique(K) < 3"
#     summary[f"#K"] = (summary[f"low #K"] + summary[f"high #K"]) / 2
#     if not active:
#         return group, summary

#     summary["Active"] = True
#     summary["Inactive reason"] = ""

#     # Set the 'low' and 'high' flags for the corresponding rows
#     group.loc[group[days_var] == low_2_days[0], "low"] = True
#     group.loc[group[days_var] == low_2_days[1], "high"] = True

#     return group, summary



def process_group_activity_summary(group):
    days_var = days_type() + "days"  # fx "t_days" eller "c_days"

    if days_type() == "t_" and clean_t_days():
        x = 1
    else:
        x = 0

    current_date = group["date"].iloc[0]
    current_ticker = group["ticker"].iloc[0]
    summary = {}

    # Initialiser nye flagkolonner
    group["low"] = False
    group["high"] = False

    # Få sorteret unikke dage for denne (date, ticker) gruppe
    unique_days = np.sort(group[days_var].unique())
    summary["#days"] = len(unique_days)

    # Inaktiv hvis der er færre end 2 unikke dage
    if len(unique_days) < 2:
        summary["Active"] = False
        summary["Inactive reason"] = "unique(_days) < 2"
        return group, summary

    # Inaktiv hvis den mindste dag er for stor, med justering baseret på x
    if unique_days[0] > 90 - x * 90 * 0.3:
        summary["Active"] = False
        summary["Inactive reason"] = "min days > 90"
        return group, summary

    # Vælg de to dage, der skal bruges – opdel unikke dage i dem under og over 30 dage
    days_below_30 = unique_days[(unique_days <= 30) & (unique_days > 8)]
    days_above_30 = unique_days[unique_days > 30]
    
    if len(days_below_30) > 0 and len(days_above_30) > 0:
        low_2_days = [max(days_below_30), min(days_above_30)]
    elif len(days_below_30) >= 2:
        low_2_days = list(days_below_30[-2:])
    elif len(days_above_30) >= 2:
        low_2_days = list(days_above_30[:2])
    
    if len(unique_days) < 3 and unique_days[0] <= 8:
        summary["Active"] = False
        summary["Inactive reason"] = "min days <= 8 & len < 3"
        return group, summary 

    summary["low days"] = low_2_days[0]
    summary["high days"] = low_2_days[1]

    # Check unikke strike-antals for de valgte dage; kræv mindst 3
    active = True
    for day, label in zip(low_2_days, ["low", "high"]):
        num_strikes = group.loc[group[days_var] == day, "K"].nunique()
        summary[f"{label} #K"] = num_strikes
        if num_strikes < 3:
            active = False
            summary["Active"] = False
            summary["Inactive reason"] = "unique(K) < 3"
    summary[f"#K"] = (summary[f"low #K"] + summary[f"high #K"]) / 2
    if not active:
        return group, summary

    summary["Active"] = True
    summary["Inactive reason"] = ""

    # Sæt 'low' og 'high' flags for de respektive rækker
    group.loc[group[days_var] == low_2_days[0], "low"] = True
    group.loc[group[days_var] == low_2_days[1], "high"] = True

    return group, summary


import concurrent.futures
import load_clean_lib


def process_group_activity_summary_wrapper(args):
    # args is a tuple: ((current_date, current_ticker), group)
    key, group = args
    current_date, current_ticker = key
    proc_group, summary = process_group_activity_summary(group)
    summary["date"] = current_date
    summary["ticker"] = current_ticker
    return proc_group, summary

def od_filter_and_summary_creater(od):

    # Create a list of (key, group) tuples from the DataFrame
    pairs = list(od.groupby(["date", "ticker"]))

    # Use ProcessPoolExecutor to process groups in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_group_activity_summary_wrapper, pairs))

    # Separate the processed groups and summary information
    processed_groups = [res[0] for res in results]
    summary_list = [res[1] for res in results]

    # Recombine all groups back into the od DataFrame
    od = pd.concat(processed_groups)

    # Update summary_dly_df (indexed by (ticker, date)) using the gathered summary information
    summary_df = pd.DataFrame(summary_list).set_index(["ticker", "date"])

    # Make dataset for results for each day for each asset
    summary_dly_df = load_clean_lib.summary_dly_df_creator(od)
    summary_dly_df.update(summary_df)

    return od, summary_dly_df


def interpolate_yield_curve_between_dates(ZCY_curves, date):

    days_var = "c_days"

    next_date_val = ZCY_curves[ZCY_curves['date'] >= date].head(1)['date'].iloc[0]
    prev_date_val = ZCY_curves[ZCY_curves['date'] <= date].tail(1)['date'].iloc[0]

    next_Curve = ZCY_curves[ZCY_curves['date'] == next_date_val]
    prev_Curve = ZCY_curves[ZCY_curves['date'] == prev_date_val]

    valid_days = set(next_Curve[days_var]).union(set(prev_Curve[days_var]))
    ZCY_date = pd.DataFrame({days_var: sorted(valid_days)})
    ZCY_date['date'] = date

    next_date_diff = (next_date_val - date) / np.timedelta64(1, 'D')
    prev_date_diff = (date - prev_date_val) / np.timedelta64(1, 'D')
    total_diff = (next_date_val - prev_date_val) / np.timedelta64(1, 'D')

    if total_diff == 0:
        # Avoid division by zero: if the two dates are identical, use next_Curve rates directly.
        ZCY_date['rate'] = ZCY_date[days_var].apply(lambda d: interpolated_ZCY(next_Curve, d))
    else:
        def interpolate_rate(days):
            next_rate = interpolated_ZCY(next_Curve, days)
            prev_rate = interpolated_ZCY(prev_Curve, days)
            return (next_rate * prev_date_diff + prev_rate * next_date_diff) / total_diff

        ZCY_date['rate'] = ZCY_date[days_var].apply(interpolate_rate)
    return ZCY_date



def interpolated_ZCY(ZCY_date, target_days, date = None, filter_date = True):

    days_var = "c_days"

    # Find the two nearest points
    lower = ZCY_date[ZCY_date[days_var] <= target_days].tail(1)
    upper = ZCY_date[ZCY_date[days_var] >= target_days].head(1)

    # Handle cases where lower or upper is empty
    if lower.empty:
        # If lower is empty, set lower to the lowest point and upper to the second lowest
        lower = ZCY_date.iloc[[0]]  # Lowest point
        upper = ZCY_date.iloc[[1]]  # Second lowest point
    elif upper.empty:
        # If upper is empty, set upper to the highest point and lower to the second highest
        upper = ZCY_date.iloc[[-1]]  # Highest point
        lower = ZCY_date.iloc[[-2]]  # Second highest point

    # Extract the days and rates for interpolation
    x0, y0 = lower[days_var].values[0], lower['rate'].values[0]
    x1, y1 = upper[days_var].values[0], upper['rate'].values[0]

    # Perform linear interpolation
    if x0 == x1:
        return y0
    else:
        return y0 + (y1 - y0) * (target_days - x0) / (x1 - x0)


# # likely irrelevant now.
# def interpolated_FW(FW_date, target_days, date = None, ticker=None, filter_date = True):
#     # # Filter the DataFrame for the given date
#     # FW_date = FW_curves
#     # if filter_date == False:
#     #     FW_date = FW_curves[(FW_curves['date'] == date)]
#     # if ticker != None:
#     #     FW_date = FW_date[(FW_date['ticker'] == ticker)]
#     # if FW_date.empty:
#     #     raise ValueError(f"No data available for the date {date}")
#     #
#     # # Sort the DataFrame by days to find the nearest points
#     # # FW_date = FW_date.sort_values(by='days')
#
#     # Find the two nearest points
#     lower = FW_date[FW_date['days'] <= target_days].tail(1)
#     upper = FW_date[FW_date['days'] >= target_days].head(1)
#
#     if lower.empty or upper.empty:
#         raise ValueError(f"Not enough data points to interpolate for {target_days} days")
#
#     # Extract the days and forward prices for interpolation
#     x0, y0 = lower['days'].values[0], lower['forwardprice'].values[0]
#     x1, y1 = upper['days'].values[0], upper['forwardprice'].values[0]
#
#     # Perform linear interpolation
#     if x0 == x1:
#         return y0
#     else:
#         return y0 + (y1 - y0) * (target_days - x0) / (x1 - x0)


from tqdm import tqdm


def high_low_swap_rates(summary_dly_df, od_rdy, n_points=200):
    # 1) Calculate var_swap_rate on od_rdy
    # df_swaps = process_od_rdy(od_rdy, replicate_SW, n_points=n_points)
    # Use the parallel version to calculate var_swap_rate
    df_swaps = process_od_rdy_parallel(od_rdy, replicate_SW, n_points=n_points)

    if summary_dly_df.index.names != ["ticker", "date"]:
        summary_dly_df = summary_dly_df.set_index(["ticker", "date"])
        
    # 2) Extract low/high rows and merge
    df_low = (
        df_swaps[df_swaps["low"]]
        .drop_duplicates(["ticker", "date"])
        .loc[:, ["ticker", "date", "var_swap_rate"]]
        .rename(columns={"var_swap_rate": "low SW"})
    )

    df_high = (
        df_swaps[df_swaps["high"]]
        .drop_duplicates(["ticker", "date"])
        .loc[:, ["ticker", "date", "var_swap_rate"]]
        .rename(columns={"var_swap_rate": "high SW"})
    )

    df_merged = pd.merge(df_low, df_high, on=["ticker", "date"], how="outer")
    df_merged.set_index(["ticker", "date"], inplace=True)

    # 3) Update summary_dly_df
    # summary_dly_df.update(df_merged[["low SW", "high SW"]])
    summary_dly_df.loc[df_merged.index, ["low SW", "high SW"]] = df_merged[["low SW", "high SW"]]

    return summary_dly_df

def interpolate_swaps_and_returns(summary_dly_df):

    if days_type() == "c_":
        T = 30
    elif days_type() == "t_":
        T = 21

    # T = 30
    t = 0
    T1 = summary_dly_df["low days"]
    T2 = summary_dly_df["high days"]
    SW1 = summary_dly_df["low SW"]
    SW2 = summary_dly_df["high SW"]
    RF = summary_dly_df["RF"]

    # summary_dly_df["SW_0_30"] = (1 / (T - t)) * (SW1 * (T1 - t) * (T2 - T) + SW2 * (T2 - t) * (T - T1)) / (T2 - T1)
    theta = (T - T1) / (T2 - T1)
    summary_dly_df["SW_0_30"] = SW1 * (1-theta) + SW2 * theta

    summary_dly_df["SW_m30_0"] = summary_dly_df.groupby("ticker")["SW_0_30"].shift(30)
    summary_dly_df["RV_m30_0"] = summary_dly_df.groupby("ticker")["RV"].shift(30)
    summary_dly_df["SW_month"] = summary_dly_df["RV_m30_0"] - summary_dly_df['SW_m30_0']

    summary_dly_df['SW_month_ln_ret'] = np.log(np.maximum(summary_dly_df["RV_m30_0"], 0.001) / np.maximum(summary_dly_df['SW_m30_0'], 0.001))
    summary_dly_df["SW_month_ln_ret_RF"] = np.log(np.maximum(summary_dly_df["RV_m30_0"], 0.001) / np.maximum((1 + RF) * summary_dly_df['SW_m30_0'], 0.001))


    summary_dly_df["SW_m1_29"] = summary_dly_df.groupby("ticker")["SW_0_30"].shift(1)
    # summary_dly_df["SW_0_29"] = (1 / (T - t)) * (SW1 * (T1 - t) * (T2 - T) + SW2 * (T2 - t) * (T - T1)) / (T2 - T1)

    theta = ((T-1) - T1) / (T2 - T1)
    summary_dly_df["SW_0_29"] = SW1 * (1-theta) + SW2 * theta
    # summary_dly_df["SW_1_30"] = summary_dly_df.groupby("ticker")["SW_0_29"].shift(-1)

    buy_price = summary_dly_df["SW_m1_29"]
    sell_price = (1/T) * 252 * summary_dly_df["squared_return"] + (T-1)/T * summary_dly_df["SW_0_29"] #todo: change 252 for the actual true average trading days a year
    summary_dly_df["SW_day"] = sell_price - buy_price

    # summary_dly_df['SW_day_RF'] = sell_price - (1 + RF) * buy_price
    # summary_dly_df['SW_day_ln_ret'] = np.log(np.maximum(sell_price, 0.001) / np.maximum(buy_price, 0.001))
    # summary_dly_df["SW_day_ln_ret_RF"] = np.log(np.maximum(sell_price, 0.001) / np.maximum((1 + RF) * buy_price, 0.001))
    summary_dly_df["SW_sell"] = sell_price
    summary_dly_df["SW_buy"] = buy_price
    # summary_dly_df["SW_return_day_scaled"] = summary_dly_df["SW_return_day"] / summary_dly_df["SW_m1_29"]

    return summary_dly_df


def add_realized_vol_to_summary(summary_dly_df, real_vol):
    # Reset index so that 'date' and 'ticker' become columns
    summary_dly_df_reset = summary_dly_df.reset_index()

    # Merge by specifying left_on and right_on keys
    merged_df = summary_dly_df_reset.merge(
        real_vol[['date', 'ticker', 'RV', 'N_tradingdays', 'open', 'close', 'squared_return', 'return']],
        left_on=['date', 'ticker'],
        right_on=['date', 'ticker'],
        how='left'
    )

    # Optionally, set the index back to ['date', 'ticker']
    summary_dly_df = merged_df.set_index(['date', 'ticker'])
    summary_dly_df = summary_dly_df.reset_index()
    return summary_dly_df

def process_od_rdy_worker(args):
    group, calc_func, kwargs = args
    return calc_func(group, **kwargs)


def process_od_rdy_parallel(od_rdy, calc_func, **kwargs):

    days_var = days_type() + "days"

    # Group by unique combination of ticker, date, and days
    groups = [group for _, group in od_rdy.groupby(["ticker", "date", days_var], group_keys=False)]
    worker_args = [(group, calc_func, kwargs) for group in groups]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_od_rdy_worker, worker_args),
            total=len(worker_args),
            desc="Processing Groups"
        ))
    return pd.concat(results)


def process_od_rdy(od_rdy, calc_func, **kwargs):

    days_var = days_type() + "days"

    # Group by the unique combination of ticker, date, and days
    grouped = list(od_rdy.groupby(["ticker", "date", days_var], group_keys=False))

    # Initialize tqdm for progress tracking
    results = []
    for key, group in tqdm(grouped, total=len(grouped), desc="Processing Groups"):
        results.append(calc_func(group, **kwargs))

    # Combine results into a single DataFrame
    return pd.concat(results)



def replicate_SW(group, n_points = 100):
    """
    For each group (e.g. a single (date, ticker, maturity)), build an implied volatility
    curve on moneyness k = ln(K/F), interpolate to a dense grid of ±8 stdevs,
    price out-of-the-money options, and numerically integrate to approximate
    the variance swap rate per Carr & Wu (2009).

    Returns the group with a new column 'var_swap_rate'.
    """

    TTM_var = days_type() + "TTM"

    group = group.copy()

    # Extract relevant parameters (assuming they're constant within this group)
    F = group["F"].iloc[0]
    r = group["r"].iloc[0]
    T = group[TTM_var].iloc[0]  # T in years (e.g. days / 365)

    # 1) Average implied volatility as a rough stdev estimate
    #    (This is the "standard deviation" used to define ±8 stdev range.)
    avg_iv = group["IV"].mean()
    stdev = avg_iv * np.sqrt(T)

    # 2) Convert strikes to moneyness k = ln(K/F)
    group["k"] = np.log(group["K"] / F)

    # Sort by k
    group = group.sort_values("k")

    # Arrays of available moneyness and implied vol
    K_data = group["K"].values
    iv_data = group["IV"].values

    # 3) Define a fine moneyness grid ±8 stdevs from 0 (i.e. from -8*stdev to +8*stdev)
    k_min = -8 * stdev
    k_max = 8 * stdev
    k_grid = np.linspace(k_min, k_max, n_points)  # 2000 points ect.

    # Convert moneyness back to strikes
    K_grid = F * np.exp(k_grid)


    # 4) Interpolate implied vol across k_grid
    #    Extrapolate by taking the edge vol for K < K_data.min() or K > K_data.max().
    iv_grid = np.interp(K_grid, K_data, iv_data,
                        left=iv_data[0],
                        right=iv_data[-1])

    # 5) Compute out-of-the-money option prices:
    #    Cal if K>F and put else: only out-of-the-money options are used.
    is_call = (K_grid > F)  # Boolean array for call/put decision
    # d1 = (np.log(F / K_grid) + (r + 0.5 * iv_grid ** 2) * T) / (iv_grid * np.sqrt(T)) # Not the forward method (r should be removed)
    d1 = (np.log(F / K_grid) + 0.5 * iv_grid ** 2 * T) / (iv_grid * np.sqrt(T))
    d2 = d1 - iv_grid * np.sqrt(T)
    call_prices = np.exp(-r * T) * (F * norm.cdf(d1) - K_grid * norm.cdf(d2))
    put_prices = call_prices - np.exp(-r * T) * (F - K_grid)
    option_prices = np.where(is_call, call_prices, put_prices)

    # 6) Approximate the variance swap rate by numerical integration over K:
    #    The continuous-time formula is roughly:
    #
    #    VarSwap ≈ 2 * e^{rT} / T * ∫_{0}^{∞} [OTMOptionPrice(K)] / K^2 dK
    #
    #    For numerical integration, we use trapezoidal rule: np.trapz(y, x).
    #
    #    Important detail: BSM_call_put returns a present value. The formula
    #    typically wants the undiscounted payoff. We'll re-scale accordingly.

    # The OTM option price from BSM_call_put is discounted, so multiply by e^{rT}.
    undiscounted_option_prices = option_prices * np.exp(r * T)

    # Perform the integral ∫(OTMPrice(K)/K^2) dK from K_min to K_max
    integrand = undiscounted_option_prices / (K_grid ** 2)
    integral_value = np.trapz(integrand, K_grid)

    # Multiply by the prefactor (2 / T)
    var_swap_rate = (2.0 / T) * integral_value

    # Store result in the group
    group["var_swap_rate"] = var_swap_rate

    return group


import matplotlib.pyplot as plt

def plot_diff_akk(df, tickers, from_date=None, to_date=None, logreturn=False):
    """
    Plots the accumulated difference (diff_akk) as a function of date for one or more tickers.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        tickers (str or list): A single ticker symbol (e.g., 'SPX') or a list of ticker symbols (e.g., ['SPX', 'AAPL']).
        from_date (str, optional): Start date for the plot (format: 'YYYY-MM-DD'). If not provided, the earliest date for the tickers is used.
        to_date (str, optional): End date for the plot (format: 'YYYY-MM-DD'). If not provided, the latest date for the tickers is used.
    """

    #filter and calculating "returns"
    df = df[df['Active']==True].copy()
    df.loc[:, 'diff'] = (df['SW'] - df['RV'])
    df.loc[:,'diff_akk'] = df.groupby('ticker')['diff'].cumsum()

    if logreturn == True:
        df.loc[:, 'diff'] = np.log(df['SW']  / df['RV'])/30 + 1
        df.loc[:,'diff_akk'] = df.groupby('ticker')['diff'].cumprod() 

    
    if isinstance(tickers, str):
        tickers = [tickers]

    # Create a new plot
    plt.figure(figsize=(12, 6))  # Set the size of the plot

    # Define a list of colors for different tickers
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Blue, Red, Green, Cyan, Magenta, Yellow, Black

    # Loop through each ticker and plot its data
    for i, ticker in enumerate(tickers):
        # Filter data for the selected ticker
        ticker_data = df[df['ticker'] == ticker].copy()

        # If from_date is not provided, use the earliest date for the ticker
        if from_date is None:
            ticker_from_date = ticker_data['date'].min()
        else:
            ticker_from_date = pd.to_datetime(from_date)  # Convert to datetime if provided

        # If to_date is not provided, use the latest date for the ticker
        if to_date is None:
            ticker_to_date = ticker_data['date'].max()
        else:
            ticker_to_date = pd.to_datetime(to_date)  # Convert to datetime if provided

        # Filter data for the selected date range
        filtered_data = ticker_data[(ticker_data['date'] >= ticker_from_date) & (ticker_data['date'] <= ticker_to_date)]

        # Plot 'diff_akk' as a function of 'date' (thin line, no markers)
        plt.plot(filtered_data['date'], filtered_data['diff_akk'], linestyle='-', linewidth=1, color=colors[i % len(colors)], label=ticker)

    # Add titles and labels
    plt.title('Accumulated Difference (SW - RV) Over Time', fontsize=16)
    plt.xlabel('date', fontsize=14)
    plt.ylabel('Accumulated Difference (diff_akk)', fontsize=14)

    # Rotate x-axis date labels for better readability
    plt.xticks(rotation=45)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    if logreturn == True: plt.yscale('log')

    # Add a legend to distinguish between tickers
    plt.legend()

    # Display the plot
    plt.tight_layout()  # Ensure everything fits in the plot
    plt.show()

