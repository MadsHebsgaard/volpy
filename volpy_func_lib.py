import pandas as pd
import numpy as np
# from dask.dataframe import read_csv
from scipy.stats import norm
from datetime import datetime
import sys
import time
from itertools import islice
import load_clean_lib
import re

# from global_settings import days_type
import importlib
import global_settings
importlib.reload(global_settings)
days_type = global_settings.days_type
clean_t_days = global_settings.clean_t_days

importlib.reload(load_clean_lib)


def load_option_data(file_path, valid_dates):

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
        "ss_flag",
        "exchange_d",
        "issue_type",
        "issuer",
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
    df['mid'] = (df['bid'] + df['ask'])/2

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



def clean_option_data_chunks(od_i_file, file_path):
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
    df = pd.read_csv(od_i_file, usecols=columns_to_load)

    df.rename(columns={"strike_price": "K"}, inplace=True)
    df.rename(columns={"best_bid": "bid"}, inplace=True)
    df.rename(columns={"best_offer": "ask"}, inplace=True)
    df.rename(columns={"impl_volatility": "IV_om"}, inplace=True)
    df['mid'] = (df['bid'] + df['ask'])/2

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
    valid_dates_path = os.path.join(file_path, "dates.csv")
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


    last_available_date = returns_and_prices["date"].max()

    if days_type() == "c_":
        T = 30  # Kalenderdage
    elif days_type() == "t_":
        T = 21  # Trading days: brug altid de næste 21 observationer

    # Filter data for perioden fra first_day til last_day + 30 dage (skal være bredt nok til at dække T observationer)
    last_day_ret = last_day + pd.Timedelta(days=60)
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
                
                n_obs = window['squared_return'].notna().sum()
                includes_last_available = last_available_date in window['date'].values

                if n_obs >= 5 or includes_last_available:
                    rv_sum = window['squared_return'].sum()
                    active_days = T if includes_last_available else n_obs
                    rv_list.append(rv_sum)
                    active_days_list.append(active_days)
                else:
                    rv_list.append(np.nan)
                    active_days_list.append(np.nan)


        elif days_type() == "c_":
            # Hvis det er kalenderdage, brug tidsbaseret slicing med pd.Timedelta(days=T)
            for _, row in group.iterrows():
                current_date = row['date']
                end_date = current_date + pd.Timedelta(days=T)
                mask = (group['date'] >= current_date) & (group['date'] < end_date)
                window = group.loc[mask]
                rv_sum = window['squared_return'].sum()

                n_obs = window['squared_return'].notna().sum()
                includes_last_available = last_available_date in window['date'].values

                if n_obs >= 5 or includes_last_available:
                    rv_sum = window['squared_return'].sum()
                    active_days = T if includes_last_available else n_obs
                    rv_list.append(rv_sum)
                    active_days_list.append(active_days)
                else:
                    rv_list.append(np.nan)
                    active_days_list.append(np.nan)
        

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

    try:

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
    except Exception:
        import traceback, sys
        print(f"\nERROR in compute_rates_for_date for date {current_date!r}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # re-raise so the ProcessPoolExecutor still knows it failed
        raise


def add_r_to_od_parallel(od, ZCY_curves, max_workers=None):
    # Split the DataFrame into groups based on 'date'
    groups = [group for _, group in od.groupby("date", group_keys=False)]

    # from volpy_parallel import add_r_to_od_parallel
    # df = add_r_to_od_parallel(od, ZCY_curves, max_workers=4)
    # df.head()

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

def vectorized_iv(F, K, T, market_price, cp_flag, tol=1e-6, max_iter=300):
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


from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

def vectorized_iv_safer(F, K, T, market_price, cp_flag,
                  tol=1e-6, max_iter=50,
                  sigma_min=1e-6, sigma_max=5.0):
    # 1) initial guess & convergence mask
    sigma     = np.full_like(F, 0.2, dtype=float)
    converged = np.zeros_like(F, dtype=bool)

    # 2) pre‐compute intrinsic value & mark floor‐cases
    intrinsic = np.where(cp_flag=='C', np.maximum(0, F-K), np.maximum(0, K-F))
    floor_mask = market_price <= intrinsic + tol
    sigma[floor_mask]    = sigma_min
    converged[floor_mask] = True

    for _ in range(max_iter):
        # 3) clip bounds
        sigma = np.clip(sigma, sigma_min, sigma_max)

        # 4) compute price & vega
        sqrtT = np.sqrt(T)
        d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT

        price = np.where(
            cp_flag=='C',
            F * norm.cdf(d1) - K * norm.cdf(d2),
            K * norm.cdf(-d2) - F * norm.cdf(-d1)
        )
        vega = F * norm.pdf(d1) * sqrtT

        # 5) Newton update only on unconverged & vega>0
        diff = price - market_price
        update_mask = (~converged) & (vega>0)
        step = diff[update_mask] / vega[update_mask]
        # avoid huge jumps
        step = np.clip(step, -1.0, 1.0)
        sigma[update_mask] -= step

        # 6) update convergence
        newly = np.abs(diff) < tol
        converged |= newly
        if converged.all():
            break

    # 7) fallback bisection on anything still unconverged
    bad = ~converged
    for i in np.where(bad)[0]:
        def f(s):
            if T[i]==0:
                return intrinsic[i] - market_price[i]
            sq = np.sqrt(T[i])
            d1i = (np.log(F[i]/K[i]) + 0.5*s**2*T[i])/(s*sq)
            d2i = d1i - s*sq
            pi = (F[i]*norm.cdf(d1i) - K[i]*norm.cdf(d2i)
                  if cp_flag[i]=='C'
                  else K[i]*norm.cdf(-d2i) - F[i]*norm.cdf(-d1i))
            return pi - market_price[i]

        try:
            sigma[i] = brentq(f, sigma_min, sigma_max, xtol=tol)
        except ValueError:
            sigma[i] = np.nan

    return sigma



def add_bid_mid_ask_IV(od, IV_type, safe_slow_IV = False):

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

    if safe_slow_IV:
        IV = vectorized_iv_safer(F, K, T, market_price, cp_flag)
    else:
        IV = vectorized_iv(F, K, T, market_price, cp_flag)
    return IV


def process_group_activity_summary(group):
    group = group.copy()

    days_var = days_type() + "days"  # fx "t_days" eller "c_days"
    if days_var == "t_days":
        target_days = 21
    else:
        target_days = 30

    if days_type() == "t_" and clean_t_days():
        t_bool = 1
    else:
        t_bool = 0

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

    # Vælg de to dage, der skal bruges – opdel unikke dage i dem under og over 30 dage
    days_below_target = unique_days[(unique_days <= target_days) & (unique_days > 8 - t_bool*3)]
    days_above_target = unique_days[unique_days >= target_days]

    low_day = None
    high_day = None
    if len(days_below_target) > 0 and len(days_above_target) > 0:
        # 1) Find low_day: tættest på cutoff under target_days
        for day in sorted(days_below_target, reverse=True):
            if group.loc[group[days_var] == day, "K"].dropna().nunique() >= 3:
                low_day = day
                break

        # 2) Find high_day: tættest på cutoff over target_days
        for day in sorted(days_above_target):
            if group.loc[group[days_var] == day, "K"].dropna().nunique() >= 3:
                high_day = day
                break

    elif len(days_below_target) >= 2:
        valid_days = []
        # Iterate days below in reverse (closest to target first)
        for day in sorted(days_below_target, reverse=True):
            if group.loc[group[days_var] == day, "K"].dropna().nunique() >= 3:
                valid_days.append(day)
                if len(valid_days) == 2:
                    break
        if len(valid_days) >= 2:
            low_day, high_day = valid_days[0], valid_days[1]

    elif len(days_above_target) >= 2:
        valid_days = []
        # Iterate days above in ascending order (closest first)
        for day in sorted(days_above_target):
            if group.loc[group[days_var] == day, "K"].dropna().nunique() >= 3:
                valid_days.append(day)
                if len(valid_days) == 2:
                    break
        if len(valid_days) >= 2:
            low_day, high_day = valid_days[0], valid_days[1]

    # if len(unique_days) < 3 and unique_days[0] <= 8:
    #     summary["Active"] = False
    #     summary["Inactive reason"] = "min days <= 8 & len < 3"
    #     return group, summary

    # first check if low and high is found
    if low_day is None or high_day is None:
        summary["Active"] = False
        summary["Inactive reason"] = "Not 2 close days"
        return group, summary

    # Inaktiv hvis den mindste dag er for stor, med justering baseret på x
    if low_day > 90 - t_bool * 90 * 0.3:
        summary["Active"] = False
        summary["Inactive reason"] = f"min days > {3*target_days}"
        return group, summary

    # # Sørg for at 'K' er numerisk
    # group["K"] = pd.to_numeric(group["K"], errors="coerce")
    # # Tæl unikke strikes per exdate
    # k_per_ex = group.groupby(days_var)["K"].nunique()
    # # Gem gennemsnittet
    # summary["#K_before_filter"] = k_per_ex.mean()
    # Sørg for at 'K' er numerisk
    group["K"] = pd.to_numeric(group["K"], errors="coerce")
    # Tæl unikke strikes per exdate
    k_per_ex = group.groupby(days_var)["K"].nunique()
    # Beregn #K før filter opdelt i low/high dage
    low_counts  = [k_per_ex.get(d, np.nan) for d in days_below_target]
    high_counts = [k_per_ex.get(d, np.nan) for d in days_above_target]
    summary["#K_low_mean_3"]  = np.nanmean(low_counts)  if low_counts  else np.nan
    summary["#K_high_mean_3"] = np.nanmean(high_counts) if high_counts else np.nan

    # 3) Brug dem hvis begge fundet
    if low_day is not None and high_day is not None:
        summary["low days"]  = low_day
        summary["high days"] = high_day

        low_cnt  = group.loc[group[days_var] == low_day,  "K"].dropna().nunique()
        high_cnt = group.loc[group[days_var] == high_day, "K"].dropna().nunique()
        summary["low #K"]  = low_cnt
        summary["high #K"] = high_cnt
        summary["#K"]      = (low_cnt + high_cnt) / 2

        summary["Active"]         = True
        summary["Inactive reason"] = ""
        group["low"]  = False # new cols 
        group["high"] = False # new cols 
        group.loc[group[days_var] == low_day,  "low"] = True # set new col to true for match
        group.loc[group[days_var] == high_day, "high"] = True # set new col to true for match

    else:
        summary["Active"] = False
        summary["Inactive reason"] = "unique(K) < 3"
        return group, summary

    # # Sæt 'low' og 'high' flags for de respektive rækker
    # group.loc[group[days_var] == low_day, "low"] = True
    # group.loc[group[days_var] == high_day, "high"] = True

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

from tqdm import tqdm


def high_low_swap_rates(summary_dly_df, od_rdy, n_points=200):
    # 1) Calculate var_swap_rate on od_rdy
    # df_swaps = process_od_rdy(od_rdy, replicate_SW, n_points=n_points)
    # Use the parallel version to calculate var_swap_rate
    df_swaps = process_od_rdy_parallel(od_rdy, replicate_SW, n_points=n_points)

    min_swap_rate = 1e-6
    df_swaps.loc[df_swaps["var_swap_rate"] <= min_swap_rate, "var_swap_rate"] = min_swap_rate



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

# def interpolate_swaps_and_returns(summary_dly_df):

#     if days_type() == "c_":
#         T = 30
#     elif days_type() == "t_":
#         T = 21

#     # T = 30
#     t = 0
#     T1 = summary_dly_df["low days"]
#     T2 = summary_dly_df["high days"]
#     SW1 = summary_dly_df["low SW"]
#     SW2 = summary_dly_df["high SW"]
#     RF = summary_dly_df["RF"]

#     # summary_dly_df["SW_0_30"] = (1 / (T - t)) * (SW1 * (T1 - t) * (T2 - T) + SW2 * (T2 - t) * (T - T1)) / (T2 - T1)
#     theta = (T - T1) / (T2 - T1)
#     summary_dly_df["SW_0_30"] = SW1 * (1-theta) + SW2 * theta

#     summary_dly_df["SW_m30_0"] = summary_dly_df.groupby("ticker")["SW_0_30"].shift(30)
#     summary_dly_df["RV_m30_0"] = summary_dly_df.groupby("ticker")["RV"].shift(30)
#     summary_dly_df["SW_month"] = summary_dly_df["RV_m30_0"] - summary_dly_df['SW_m30_0']

#     summary_dly_df['SW_month_ln_ret'] = np.log(np.maximum(summary_dly_df["RV_m30_0"], 0.001) / np.maximum(summary_dly_df['SW_m30_0'], 0.001))
#     summary_dly_df["SW_month_ln_ret_RF"] = np.log(np.maximum(summary_dly_df["RV_m30_0"], 0.001) / np.maximum((1 + RF) * summary_dly_df['SW_m30_0'], 0.001))


#     summary_dly_df["SW_m1_29"] = summary_dly_df.groupby("ticker")["SW_0_30"].shift(1)
#     # summary_dly_df["SW_0_29"] = (1 / (T - t)) * (SW1 * (T1 - t) * (T2 - T) + SW2 * (T2 - t) * (T - T1)) / (T2 - T1)

#     theta = ((T-1) - T1) / (T2 - T1)
#     summary_dly_df["SW_0_29"] = SW1 * (1-theta) + SW2 * theta
#     # summary_dly_df["SW_1_30"] = summary_dly_df.groupby("ticker")["SW_0_29"].shift(-1)

#     summary_dly_df["SW_sell"] = (1/T) * 252 * summary_dly_df["squared_return"] + (T-1)/T * summary_dly_df["SW_0_29"] #todo: change 252 for the actual true average trading days a year
#     summary_dly_df["SW_buy"] = summary_dly_df["SW_m1_29"]

#     summary_dly_df["CF_30_SW_day"] = summary_dly_df["SW_sell"] - summary_dly_df["SW_buy"]
#     summary_dly_df["r_30_SW_day"] = summary_dly_df["CF_30_SW_day"] / summary_dly_df["SW_buy"].shift(1).rolling(window=21).mean()

#     # summary_dly_df['SW_day_RF'] = sell_price - (1 + RF) * buy_price
#     # summary_dly_df['SW_day_ln_ret'] = np.log(np.maximum(sell_price, 0.001) / np.maximum(buy_price, 0.001))
#     # summary_dly_df["SW_day_ln_ret_RF"] = np.log(np.maximum(sell_price, 0.001) / np.maximum((1 + RF) * buy_price, 0.001))
#     # summary_dly_df["SW_return_day_scaled"] = summary_dly_df["SW_return_day"] / summary_dly_df["SW_m1_29"]

#     return summary_dly_df

# load_clean_lib.create_log_with_enty("3")
def interpolate_swaps_and_returns(df):
    from vol_strat_lib import T_day_interpolation_CW, T_day_interpolation_linear
    if days_type() == "c_":
        T = 30
    elif days_type() == "t_":
        T = 21

    #interpolation
    T1 = df["low days"]
    T2 = df["high days"]
    SW1 = df["low SW"]
    SW2 = df["high SW"]


    df["SW_0_30_linear"] = T_day_interpolation_linear(T1, T2, SW1, SW2, t=0)
    df["theta_linear"] = np.where(
        T1 == T2,
        0.5,
        (T2-(T-0)) / (T2 - T1)
    )

    df["SW_0_30"] = T_day_interpolation_CW(T1, T2, SW1, SW2, t=0)
    SW = df["SW_0_30"]
    df["theta_CW"] = np.where(
        T1 == T2,
        0.5,
        (T1 - 0) * (T2 - T) / ((T2 - T1) * T) * (T/(T-0))
    )

    # Calculating high/low compatibility
    df["high_low_compatibility"] = -(T1 - T) * (T2 - T) * (SW1 - SW2) ** 2 / T ** 2
    # If positive they are good (interpolating from either side), yet best if closer to 0
    # if negative they are bad, getting worse as the compatibility decreases.
    # If 0 the swap is either low/high SW or the rates of both are identical, in which case the swap rate time curve could be convex or concave leading to some interpolation error still.

    # Calculating SW score
    min_SW = np.minimum(SW1, SW2)
    max_SW = np.maximum(SW1, SW2)

    # df["SW_score"] = np.maximum(
    #     SW / min_SW, max_SW/SW
    # )
    # # if 1 then the interpolated swap is perfectly close to either swap, accuracy decreases when score increases.
    # # if negative the swap is negative

    df.loc[:, "SW_score"] = np.maximum(SW / min_SW, max_SW / SW)

    # Filtrér med .loc og lav eksplicit kopi
    mask = (
        (df["SW_score"] < 3) &
        (df["SW_score"] > 0) &
        (df["high_low_compatibility"] > -0.25) &
        (df["high_low_compatibility"] < 2)
    )
    df["Acceptable"] = mask

    # theta_29 = np.where(
    #     T2 == T1,
    #     1,
    #     ((T - 1) - T1) / (T2 - T1)
    # )
    # df["SW_0_29"] = SW1 * (1 - theta_29) + SW2 * theta_29
    df["SW_0_29_linear"] = T_day_interpolation_linear(T1, T2, SW1, SW2, t=1)
    df["SW_0_29"] = T_day_interpolation_CW(T1, T2, SW1, SW2, t=1)

    #Shifted values
    df["SW_m1_29"] = df.groupby("ticker")["SW_0_30"].shift(1)


    #discount factor
    df["discount_factor_30d"] = 1 / (1 + df["RF"]) ** (20)

    # Casflow version baseret på markedsværdi af en daglig (-short) swap = long
    df["CF_30_SW_day"] = - (
            df["SW_m1_29"]  # fast ben er det samme
            - df["SW_0_29"] * (T - 1) / T  # ny "rest" variabelt ben
            - df["squared_return"] * 252 * 1 / T  # realiseret variabllt ben
    ) * df["discount_factor_30d"]  # diskonteres

    # Casflow version baseret på markedsværdi af en daglig (-short) swap = long
    df["CF_30_SW_day_noRF"] = - (
            df["SW_m1_29"]  # fast ben er det samme
            - df["SW_0_29"] * (T - 1) / T  # ny "rest" variabelt ben
            - df["squared_return"] * 252 * 1 / T  # realiseret variabllt ben
    )

    df = df.loc[mask].copy() # Only keep if acceptable

    df["RV-SW 30"] = df["RV"] - df["SW_0_30"]

    df["Average SW"] = df["SW_m1_29"].rolling(window=21, min_periods=5).mean()
    df["EWMA SW 87.5%"] = df["SW_m1_29"].ewm(alpha=0.095, min_periods=5, adjust=False).mean()
    df["EWMA SW 80%"] = df["SW_m1_29"].ewm(alpha=0.075, min_periods=5, adjust=False).mean()
    df["EWMA SW 66%"] = df["SW_m1_29"].ewm(alpha=0.55, min_periods=5, adjust=False).mean()

    df["r_30_SW_day"] = df["CF_30_SW_day"] / df["Average SW"]
    df["r_30_SW_day_noRF"] = df["CF_30_SW_day_noRF"] / df["Average SW"]

    for alpha in np.arange(5, 30, 5):
        df[f"EWMA SW .{alpha}"] = df["SW_m1_29"].ewm(alpha=alpha * 0.01, min_periods=5, adjust=False).mean()
        df[f"r_30_SW_day .{alpha}"] = df["CF_30_SW_day"] / df[f"EWMA SW .{alpha}"]
        df[f"r_30_SW_day_noRF .{alpha}"] = df["CF_30_SW_day_noRF"] / df[f"EWMA SW .{alpha}"]

    return df

def add_calcs_to_files(filename_list = ["sum1", "sum2", "orpy"]):
    if days_type() == "c_":
        T = 30
    elif days_type() == "t_":
        T = 21

    base_dir = load_clean_lib.Option_metrics_path_from_profile()

    for filename in filename_list:
        file_dir = base_dir / "Tickers" / "SumAndOrpy" / "t_" / filename

        for ticker in file_dir.iterdir():
            df = pd.read_csv(file_dir / f"{ticker}")

            # df["r_30_SW_day old"] = df["CF_30_SW_day"] / df["SW_0_30"].rolling(window=21, min_periods=5).mean()
            df["RV-SW 30"] = df["RV"] - df["SW_0_30"]


            # #add calculation here
            # df["EWMA SW 0.11"] = df["SW_m1_29"].ewm(alpha=0.11, min_periods=5, adjust=False).mean()
            # df["r_30_SW_day 0.11"] = df["CF_30_SW_day"] / df["EWMA SW 0.11"]
            # df["r_30_SW_day_noRF 0.11"] = df["CF_30_SW_day_noRF"] / df["EWMA SW 0.11"]

            for alpha in np.arange(5,30,5):
                df[f"EWMA SW .{alpha}"] = df["SW_m1_29"].ewm(alpha=alpha*0.01, min_periods=5, adjust=False).mean()
                df[f"r_30_SW_day .{alpha}"] = df["CF_30_SW_day"] / df[f"EWMA SW .{alpha}"]
                df[f"r_30_SW_day_noRF .{alpha}"] = df["CF_30_SW_day_noRF"] / df[f"EWMA SW .{alpha}"]

            df.to_csv(file_dir / f"{ticker}")
        print(f"finished with {filename}")


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
    curve on moneyness k = ln(K/F), interpolate to a dense grid of ±10 stdevs,
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
    k_min = -10 * stdev
    k_max = 10 * stdev
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




def replicate_SW_k(group, n_points = 100):
    """
    For each group (e.g. a single (date, ticker, maturity)), build an implied volatility
    curve on moneyness k = ln(K/F), interpolate to a dense grid of ±10 stdevs,
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
    k_min = -10 * stdev
    k_max = 10 * stdev
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




def replicate_SW_K(group, n_points = 100):
    """
    For each group (e.g. a single (date, ticker, maturity)), build an implied volatility
    curve on moneyness k = ln(K/F), interpolate to a dense grid of ±10 stdevs,
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
    k_min = -10 * stdev
    k_max = 10 * stdev

    K_min = F * np.exp(k_min)
    K_max = F * np.exp(k_max)

    K_grid = np.linspace(K_min, K_max, n_points)  # 2000 points ect.


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




def load_analyze_create_swap(om_folder="i2s1_full_v2", ticker_list=["SPX", "OEX"], first_day=None, last_day=None,
                             IV_type="om", save_files = True, safe_slow_IV = False):
    # Load data and clean
    od, returns_and_prices, od_raw = load_clean_lib.load_clean_and_prepare_od(om_folder=om_folder,
                                                                              tickers=ticker_list,
                                                                              first_day=None,
                                                                              last_day=None,
                                                                              IV_type=IV_type,
                                                                              safe_slow_IV = safe_slow_IV)
    
    
    # Calculate results such as SW, RV ect.
    summary_dly_df, od_rdy = load_clean_lib.create_summary_dly_df(od, returns_and_prices,
                                                                  first_day=None,
                                                                  last_day=None,
                                                                  n_grid=2000)
    summary_dly_df = interpolate_swaps_and_returns(summary_dly_df)
    summary_dly_df = summary_dly_df.reset_index()

    # save
    if save_files:
        output_dir = load_clean_lib.volpy_output_dir(om_folder)
        time_type = days_type()
        summary_dly_df.to_csv(f"{output_dir}/{time_type}summary_dly.csv", index=False)
        od_raw.to_csv(f"{output_dir}/{time_type}od_raw.csv", index=False)
        # od_rdy.to_csv(f"{output_dir}/{time_type}od_rdy.csv", index=False)

    return summary_dly_df, od




def load_analyze_create_swap_chunks(om_folder="i2s1_full_v2", ticker_list=["SPX", "OEX"], first_day=None, last_day=None,
                             IV_type="om", save_files = True, safe_slow_IV = False):
    
    om_dir = load_clean_lib.Option_metrics_path_from_profile()
    dir = om_dir / om_folder
    option_data_folder = dir / "option data"
    load_clean_lib.volpy_output_dir(om_folder).mkdir(parents=True, exist_ok=True)
    files = sorted(option_data_folder.glob("*.csv"))

    # Load metadata kun én gang
    FW, ZCY_curves, returns_and_prices = load_clean_lib.load_option_metadata(
        om_folder=om_folder, tickers=ticker_list
    )

    all_summaries = []   # <-- her samler vi alle chunks
    for file in files:
        # Load & clean
        od, od_raw = load_clean_lib.load_clean_and_prepare_od_chunks(
            od_i_file=file,
            om_folder=om_folder,
            FW=FW,
            ZCY_curves=ZCY_curves,
            tickers=ticker_list,
            first_day=None,
            last_day=None,
            IV_type=IV_type,
            safe_slow_IV=safe_slow_IV
        )

        # Summary
        summary_dly_df, od_rdy = load_clean_lib.create_summary_dly_df(
            od, returns_and_prices,
            first_day=None,
            last_day=None,
            n_grid=2000
        )

        summary_dly_df = interpolate_swaps_and_returns(summary_dly_df).reset_index()

        # Gem per-chunk
        if save_files:
            output_dir = load_clean_lib.volpy_output_dir(om_folder)
            time_type = days_type()
            stem = file.stem
            m = re.search(r'(\d{4}(?:-\d{2})?)$', stem)
            stamp = m.group(1) if m else stem
            od_raw.to_csv(f"{output_dir}/{time_type}od_raw_{stamp}.csv", index=False)
            od_rdy.to_csv(f"{output_dir}/{time_type}od_rdy_{stamp}.csv", index=False)

        # Append til liste
        all_summaries.append(summary_dly_df)

    # Efter løkken: sæt det hele sammen og returnér kun summary
    if all_summaries:
        full_summary = pd.concat(all_summaries).sort_values("date").reset_index(drop=True)
    else:
        full_summary = pd.DataFrame()

    full_summary.to_csv(f"{output_dir}//{time_type}summary_dly.csv", index=False)

    return full_summary



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




def create_sgy_list(sgy_common = "CF_D_30_", sgy_list = ["straddle", "strangle_15%", "call_ATM", "put_ATM"]):
    return [sgy_common + sgy for sgy in sgy_list] + ["CF_30_SW_day", "r_30_SW_day"]


def return_df(df_big, sgy_list = create_sgy_list(), ticker_list = ["SPX"], extra_columns = []):
    df = df_big[df_big["ticker"].isin(ticker_list)]

    # df = df[df["CF_D_30_put_ATM"].isna() == False]
    # df = df[df["CF_D_30_call_ATM"].isna() == False]

    col_list = ["ticker", "date", "r_stock"] + sgy_list + extra_columns
    df = df[col_list]
    return df


def scale_columns_to_r_stock_std_dev(df, sgy_list, ref_column="r_stock"):
    """
    Scale the columns in sgy_list within the dataframe df so that each has the same volatility
    (standard deviation) as the reference column, default 'r_stock'.

    Parameters:
    - df: pandas.DataFrame containing the data.
    - sgy_list: list of column names in df to be scaled.
    - ref_column: the name of the reference column to match volatility (default is 'r_stock').

    Returns:
    - df: The dataframe with scaled columns.
    """
    # Ensure the reference column exists.
    if ref_column not in df.columns:
        raise ValueError(f"Reference column '{ref_column}' not found in DataFrame.")

    # Calculate the standard deviation (volatility) of the reference column.
    ref_vol = df[ref_column].std()

    # Loop through each column in sgy_list and scale it.
    for col in sgy_list:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame, skipping.")
            continue

        # Calculate the current volatility of the column.
        col_vol = df[col].std()
        if col_vol == 0:
            print(f"Column '{col}' has zero volatility, skipping scaling.")
            continue

        # Calculate the scaling factor and apply the scaling.
        scale_factor = ref_vol / col_vol
        df[col] = df[col] * scale_factor

    return df

def scale_columns_to_r_stock_average(df, sgy_list, ref_column="r_stock"):
    """
    Scale the columns in sgy_list within the dataframe df so that each has the same average
    (mean) as the reference column, default 'r_stock'.

    Parameters:
    - df: pandas.DataFrame containing the data.
    - sgy_list: list of column names in df to be scaled.
    - ref_column: the name of the reference column to match the average (default is 'r_stock').

    Returns:
    - df: The DataFrame with scaled columns.
    """
    # Ensure the reference column exists.
    if ref_column not in df.columns:
        raise ValueError(f"Reference column '{ref_column}' not found in DataFrame.")

    # Compute the mean of the reference column.
    ref_mean = df[ref_column].mean()

    # Loop through each column in sgy_list and scale it.
    for col in sgy_list:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame, skipping.")
            continue

        # Calculate the current mean of the column.
        col_mean = df[col].mean()
        if col_mean == 0:
            print(f"Column '{col}' has a zero mean, skipping scaling to avoid division by zero.")
            continue

        # Compute scaling factor so that the new average becomes ref_mean.
        scale_factor = ref_mean / col_mean
        df[col] = df[col] * scale_factor

    return df


def plot_returns(df, sgy_common, sgy_names, factors, figsize=(30, 10)):
    plt.figure(figsize=figsize)

    for sgy_name in sgy_names:
        sgy_str = sgy_common + sgy_name
        plt.plot(df["date"], np.cumsum(df[f"{sgy_str}"]), label=rf"{sgy_name}", alpha=0.8)

    plt.plot(df["date"], np.cumsum(df["r_stock"]),
        label="Stock", alpha=0.4)

    for factor in factors:
        plt.plot(df["date"], np.cumsum(df[f"{factor}"]), label=rf"{factor}", alpha=0.8)

    plt.plot(df["date"], np.cumsum(df["CF_30_SW_day"]), label = "Swap day")

    plt.plot(df["date"], np.cumsum(df["r_30_SW_day"]), label = "Swap day return")

    plt.grid()
    plt.legend()
    plt.show()

# missing ticker
def make_df_strats(df, sgy_common = "CF_D_30_", sgy_names = ["straddle", "strangle_15%", "call_ATM", "put_ATM"],
                   factors=['Mkt', 'SPX', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'BAB', 'QMJ', 'RF'], vol_index = False, sign=True,
                   scale=True, plot = False, ticker_list = None, extra_columns = [], figsize = (30, 10)):
    if sgy_names is None:
        sgy_names = [col.replace(sgy_common, "") for col in df.columns if sgy_common in col]

    if ticker_list == None:
        ticker_list = list(df["ticker"].unique())

    sgy_list = create_sgy_list(sgy_common, sgy_names)

    df = return_df(df, sgy_list = sgy_list, ticker_list = ticker_list, extra_columns = extra_columns)

    if vol_index:   vol_symbols = load_clean_lib.vol_symbols
    else:           vol_symbols = []
    df = add_factor_df_columns(df, factors + vol_symbols)

    if sign:
        df = scale_columns_to_r_stock_average(df, sgy_list + factors, ref_column="r_stock")
    if scale:
        df = scale_columns_to_r_stock_std_dev(df, sgy_list + factors, ref_column="r_stock")

    if plot:
        factors = [col for col in factors if col != "RF"]
        plot_returns(df, sgy_common, sgy_names, factors, figsize)

    return df


def add_factor_df_columns(df, factor_df_columns=['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'BAB', 'QMJ', 'RF']):
    factor_df = pd.read_csv("data/factor_df.csv")
    factor_df['date'] = pd.to_datetime(factor_df['date'], format='%Y-%m-%d')

    factor_df = factor_df[["date"] + factor_df_columns]
    return_df = df.merge(factor_df, on='date', how='left')
    return return_df


def lm_regress(df, y_column, x_columns, print_summary=True):
    df = df[df[y_column].isna() == False]
    import statsmodels.api as sm

    # Define your target variable and factor variables
    X = df[x_columns]  # Independent variables
    y = df[y_column]  # Dependent variable

    # Add a constant term to the independent variables for the intercept
    X = sm.add_constant(X)

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    if print_summary:
        # Print the summary statistics
        print(model.summary())

    return model.summary()



from scipy.stats import skew, kurtosis


def compute_performance_measures_cashflows(df, cvar_alpha=0.05):
    df = df.drop(columns=["date", "RF"], errors="ignore")
    df = df.apply(pd.to_numeric, errors='coerce')

    results = {}

    for col in df.columns:
        data = df[col].dropna()

        mean = data.mean()
        std = data.std()
        downside_std = data[data < 0].std()
        sharpe = mean / std if std != 0 else np.nan
        sortino = mean / downside_std if downside_std != 0 else np.nan

        # PnL path and max drawdown
        pnl_path = data.cumsum()
        peak = pnl_path.cummax()
        drawdown = pnl_path - peak
        max_drawdown = drawdown.min()

        # Risk metrics
        var = np.quantile(data, cvar_alpha)
        cvar = data[data <= var].mean()

        results[col] = {
            "Mean": mean,
            "Std Dev": std,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Skew": skew(data),
            "Kurtosis": kurtosis(data),
            "Max Drawdown (CumCF)": max_drawdown,
            f"VaR {int(cvar_alpha * 100)}%": var,
            f"CVaR {int(cvar_alpha * 100)}%": cvar,
            "Total Cashflow": pnl_path.iloc[-1]
        }

    return pd.DataFrame(results).T


def compute_extensive_performance_measures_cashflows(df, cvar_alpha=0.05, periods_per_year=252):
    # Drop unnecessary columns and convert to numeric values
    df = df.drop(columns=["date", "RF"], errors="ignore")
    df = df.apply(pd.to_numeric, errors='coerce')

    results = {}

    for col in df.columns:
        data = df[col].dropna()

        # Compute daily basic statistics
        mean = data.mean()
        std = data.std()
        downside_std = data[data < 0].std()

        # Annualized metrics
        annualized_mean = mean * periods_per_year
        annualized_std = std * (periods_per_year ** 0.5)
        annualized_sharpe = annualized_mean / annualized_std if annualized_std != 0 else np.nan
        # Annualized Sortino: scale daily ratio by sqrt(periods_per_year)
        annualized_sortino = (mean / downside_std * (periods_per_year ** 0.5)) if downside_std != 0 else np.nan

        # Cumulative cashflow path and drawdown
        pnl_path = data.cumsum()
        peak = pnl_path.cummax()
        drawdown = pnl_path - peak
        max_drawdown = drawdown.min()

        # Risk metrics: compute daily VaR and CVaR then annualize via sqrt(time) scaling
        var = np.quantile(data, cvar_alpha)
        cvar = data[data <= var].mean()
        annualized_var = var * (periods_per_year ** 0.5)
        annualized_cvar = cvar * (periods_per_year ** 0.5)

        total_cashflow = pnl_path.iloc[-1]

        # Win/loss statistics (not annualized)
        pos = data[data > 0]
        neg = data[data < 0]
        win_rate = len(pos) / len(data) if len(data) > 0 else np.nan
        avg_gain = pos.mean() if not pos.empty else np.nan
        avg_loss = neg.mean() if not neg.empty else np.nan
        gain_loss_ratio = (avg_gain / abs(avg_loss)) if (avg_loss != 0 and not np.isnan(avg_loss)) else np.nan
        profit_factor = pos.sum() / abs(neg.sum()) if neg.sum() != 0 else np.nan

        # Calmar Ratio: Annualized mean return divided by the absolute max drawdown
        calmar_ratio = annualized_mean / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # Tail Ratio: Invariant to scaling, so calculated directly on daily returns
        q90 = data.quantile(0.9)
        q10 = data.quantile(0.1)
        tail_ratio = q90 / abs(q10) if q10 != 0 else np.nan

        # Recovery Duration: Measure from the maximum drawdown point (trough) to the recovery point
        # Identify the index of maximum drawdown (trough)
        max_dd_idx = np.argmin(drawdown)

        # Recovery target is the peak value at the time of max drawdown
        recovery_target = peak.iloc[max_dd_idx]

        # Initialize recovery period calculation
        recovery_periods = None

        # Loop from max drawdown index onward to see when recovery is achieved
        for j in range(max_dd_idx, len(pnl_path)):
            if pnl_path.iloc[j] >= recovery_target:
                recovery_periods = j - max_dd_idx
                break

        if recovery_periods is None:
            # Recovery never occurred in the observed data.
            # Calculate elapsed periods since max drawdown.
            elapsed_periods = len(pnl_path) - max_dd_idx
            # Calculate the gap between the target and the last pnl value.
            current_gap = recovery_target - pnl_path.iloc[-1]
            # Use the mean return (assumed positive) to estimate extra periods for recovery.
            mean_return = data.mean()
            if mean_return > 0:
                additional_periods = current_gap / mean_return
            else:
                additional_periods = np.nan  # Alternatively, you could use a default value or float('inf')
            recovery_periods = elapsed_periods + additional_periods

        # Convert recovery time to years
        recovery_duration_years = recovery_periods / periods_per_year


        # Lag-1 Autocorrelation (daily data)
        autocorr = data.autocorr(lag=1)

        results[col] = {
            "Ann. Mean": annualized_mean,
            "Ann. Std Dev": annualized_std,
            "Ann. Sharpe Ratio": annualized_sharpe,
            "Ann. Sortino Ratio": annualized_sortino,
            "Skew": skew(data),
            "Kurtosis": kurtosis(data),
            "Max Drawdown (CumCF)": max_drawdown,
            f"Ann. VaR {int(cvar_alpha * 100)}%": annualized_var,
            f"Ann. CVaR {int(cvar_alpha * 100)}%": annualized_cvar,
            "Total Cashflow": total_cashflow,
            "Win Rate": win_rate,
            "Average Gain": avg_gain,
            "Average Loss": avg_loss,
            "Gain/Loss Ratio": gain_loss_ratio,
            "Profit Factor": profit_factor,
            "Calmar Ratio": calmar_ratio,
            "Tail Ratio": tail_ratio,
            "Recovery Duration (years)": recovery_duration_years,
            "Lag-1 Autocorrelation": autocorr
        }

    return pd.DataFrame(results).T


def compute_extensive_performance_measures_cashflows_FF_factors(df, cvar_alpha=0.05, periods_per_year=252):
    from scipy.stats import skew, kurtosis
    import statsmodels.api as sm

    # Define all factor names (assumed to be present in the DataFrame)
    factor_names = ['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'BAB', 'QMJ']
    # FF3 regression will use only these three factors:
    ff3_factors = ['Mkt', 'SMB', 'HML']

    # Identify asset return columns (exclude factor columns and date/risk-free)
    asset_cols = [col for col in df.columns if col not in factor_names + ["date", "RF"]]

    results = {}

    for col in asset_cols:
        data = df[col].dropna()

        # Compute daily basic statistics
        mean = data.mean()
        std = data.std()
        downside_std = data[data < 0].std()

        # Annualized metrics
        annualized_mean = mean * periods_per_year
        annualized_std = std * (periods_per_year ** 0.5)
        annualized_sharpe = annualized_mean / annualized_std if annualized_std != 0 else np.nan
        annualized_sortino = (mean / downside_std * (periods_per_year ** 0.5)) if downside_std != 0 else np.nan

        # Cumulative cashflow path and drawdown
        pnl_path = data.cumsum()
        peak = pnl_path.cummax()
        drawdown = pnl_path - peak
        max_drawdown = drawdown.min()

        # Risk metrics: compute daily VaR and CVaR then annualize via sqrt(time) scaling
        var = np.quantile(data, cvar_alpha)
        cvar = data[data <= var].mean()
        annualized_var = var * (periods_per_year ** 0.5)
        annualized_cvar = cvar * (periods_per_year ** 0.5)

        total_cashflow = pnl_path.iloc[-1]

        # Win/loss statistics
        pos = data[data > 0]
        neg = data[data < 0]
        win_rate = len(pos) / len(data) if len(data) > 0 else np.nan
        avg_gain = pos.mean() if not pos.empty else np.nan
        avg_loss = neg.mean() if not neg.empty else np.nan
        gain_loss_ratio = (avg_gain / abs(avg_loss)) if (avg_loss != 0 and not np.isnan(avg_loss)) else np.nan
        profit_factor = pos.sum() / abs(neg.sum()) if neg.sum() != 0 else np.nan

        # Calmar Ratio
        calmar_ratio = annualized_mean / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # Tail Ratio
        q90 = data.quantile(0.9)
        q10 = data.quantile(0.1)
        tail_ratio = q90 / abs(q10) if q10 != 0 else np.nan

        # Recovery Duration Calculation
        max_dd_idx = np.argmin(drawdown)
        recovery_target = peak.iloc[max_dd_idx]
        recovery_periods = None
        for j in range(max_dd_idx, len(pnl_path)):
            if pnl_path.iloc[j] >= recovery_target:
                recovery_periods = j - max_dd_idx
                break
        if recovery_periods is None:
            elapsed_periods = len(pnl_path) - max_dd_idx
            current_gap = recovery_target - pnl_path.iloc[-1]
            mean_return = data.mean()
            additional_periods = current_gap / mean_return if mean_return > 0 else np.nan
            recovery_periods = elapsed_periods + additional_periods
        recovery_duration_years = recovery_periods / periods_per_year

        # Lag-1 Autocorrelation
        autocorr = data.autocorr(lag=1)

        # FF3 Regression (for asset excess returns vs. Mkt, SMB, and HML)
        # Ensure that the factor columns exist and align with the asset data
        if all(f in df.columns for f in ff3_factors):
            # Extract factor data aligned to the asset's data index
            factors_data = df[ff3_factors].loc[data.index].dropna()
            # Align asset returns with the factor data
            aligned_data = data.loc[factors_data.index]
            if len(aligned_data) > 0:
                X = sm.add_constant(factors_data)
                model = sm.OLS(aligned_data, X).fit()
                ff3_alpha = model.params['const'] * periods_per_year
                ff3_beta_mkt = model.params.get('Mkt', np.nan)
                ff3_beta_smb = model.params.get('SMB', np.nan)
                ff3_beta_hml = model.params.get('HML', np.nan)
            else:
                ff3_alpha = np.nan
                ff3_beta_mkt = np.nan
                ff3_beta_smb = np.nan
                ff3_beta_hml = np.nan
        else:
            ff3_alpha = np.nan
            ff3_beta_mkt = np.nan
            ff3_beta_smb = np.nan
            ff3_beta_hml = np.nan

        # Store all performance measures including FF3 regression outputs
        results[col] = {
            "Ann. Mean": annualized_mean,
            "Ann. Std Dev": annualized_std,
            "Ann. Sharpe Ratio": annualized_sharpe,
            "Ann. Sortino Ratio": annualized_sortino,
            "Skew": skew(data),
            "Kurtosis": kurtosis(data),
            "Max Drawdown (CumCF)": max_drawdown,
            f"Ann. VaR {int(cvar_alpha * 100)}%": annualized_var,
            f"Ann. CVaR {int(cvar_alpha * 100)}%": annualized_cvar,
            "Total Cashflow": total_cashflow,
            "Win Rate": win_rate,
            "Average Gain": avg_gain,
            "Average Loss": avg_loss,
            "Gain/Loss Ratio": gain_loss_ratio,
            "Profit Factor": profit_factor,
            "Calmar Ratio": calmar_ratio,
            "Tail Ratio": tail_ratio,
            "Recovery Duration (years)": recovery_duration_years,
            "Lag-1 Autocorrelation": autocorr,
            "FF3 Alpha": ff3_alpha,
            "FF3 Beta (Mkt)": ff3_beta_mkt,
            "FF3 Beta (SMB)": ff3_beta_smb,
            "FF3 Beta (HML)": ff3_beta_hml
        }

    return pd.DataFrame(results).T

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_timeseries_with_pct(df, alpha=0.1, var="days", var_name=None, savefig=False, figsize = (12, 6)):

    # ensure your DataFrame is datetime‑indexed
    # SPX_df['date'] = pd.to_datetime(SPX_df['date'])
    # SPX_df.set_index('date', inplace=True)

    df[f'high {var}'] = np.where(df[f'low {var}'] == 21, 21, df[f'high {var}']) # technically should depend on if c_ or t_ days

    # half‑year window
    window_size = 21 * 6  # ≈126 trading days

    # rolling means
    rolling_high = df[f'high {var}'] \
        .rolling(window=window_size, min_periods=window_size) \
        .mean() \
        .dropna()

    rolling_low = df[f'low {var}'] \
        .rolling(window=window_size, min_periods=window_size) \
        .mean() \
        .dropna()

    if alpha is not None:
        # rolling percentiles
        high_lower = df[f'high {var}'] \
            .rolling(window=window_size, min_periods=window_size) \
            .quantile(alpha) \
            .dropna()

        high_upper = df[f'high {var}'] \
            .rolling(window=window_size, min_periods=window_size) \
            .quantile(1 - alpha) \
            .dropna()

        low_lower = df[f'low {var}'] \
            .rolling(window=window_size, min_periods=window_size) \
            .quantile(alpha) \
            .dropna()

        low_upper = df[f'low {var}'] \
            .rolling(window=window_size, min_periods=window_size) \
            .quantile(1 - alpha) \
            .dropna()

    fig, ax = plt.subplots(figsize=figsize)

    if var == "days":
        # reference line at 21 days
        ax.axhline(y=21, color="black", linestyle="--", alpha=0.75)

    # plot rolling means and store line objects to extract colors
    line_high, = ax.plot(rolling_high.index, rolling_high, label='High TTM', linewidth=1.5)
    line_low, = ax.plot(rolling_low.index, rolling_low, label='Low TTM', linewidth=1.5)

    if alpha is not None:
        # get colors from the lines
        high_color = line_high.get_color()
        low_color = line_low.get_color()

        # fill between with only facecolor (no edgecolor)
        ax.fill_between(high_upper.index,
                        high_lower,
                        high_upper,
                        facecolor=high_color,
                        edgecolor=None,
                        linewidth=0,
                        alpha=0.2)

        # draw the edges separately with their own alpha
        ax.plot(high_lower.index, high_lower, color=high_color, linewidth=1, alpha=0.5)
        ax.plot(high_upper.index, high_upper, color=high_color, linewidth=1, alpha=0.5)

        ax.fill_between(low_upper.index,
                        low_lower,
                        low_upper,
                        facecolor=low_color,
                        edgecolor=None,
                        linewidth=0,
                        alpha=0.2)

        ax.plot(low_lower.index, low_lower, color=low_color, linewidth=1, alpha=0.5)
        ax.plot(low_upper.index, low_upper, color=low_color, linewidth=1, alpha=0.5)

    # format the x‑axis to show one tick every two years
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()  # rotate & align

    ax.legend()
    # ax.set_title('Half‑Year Rolling Means with ±percentile bands')
    ax.set_xlabel('Date')
    ax.set_ylabel(var_name)
    ax.grid(alpha=0.5)
    ax.set_ylim(0, None)

    plt.tight_layout()

    import os
    if savefig:
        os.makedirs("figures/summary", exist_ok=True)
        plt.savefig(f"figures/summary/rolling average with percentiles ({var}).pdf")
    plt.show()

# define your window length
def plot_lowest_number_of_strikes_timeseries(sum_df, savefig = False, figsize = (12, 6)):
    from matplotlib.ticker import MultipleLocator

    window_size = 21 * 12  # e.g. 252 days
    alpha = 1 / window_size

    # helper to average the bottom 5% of values in an array
    def bottom_5pct_avg(x):
        k = max(int(len(x) * alpha), 1)  # at least one value
        # partition so first k entries are the k smallest (unsorted)
        smallest_k = np.partition(x, k - 1)[:k]
        return smallest_k.mean()

    # apply rolling window with that custom function
    rolling_bot5_high = sum_df['high #K'] \
        .rolling(window=window_size, min_periods=window_size) \
        .apply(bottom_5pct_avg, raw=True)
    rolling_bot5_low = sum_df['low #K'] \
        .rolling(window=window_size, min_periods=window_size) \
        .apply(bottom_5pct_avg, raw=True)

    # drop the initial NaNs (first window_size‑1 days)
    rolling_bot5_high = rolling_bot5_high.dropna()
    rolling_bot5_low = rolling_bot5_low.dropna()

    # plot with true dates on x-axis (every other year)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling_bot5_high.index, rolling_bot5_high, label='High TTM', linewidth=1)
    ax.plot(rolling_bot5_low.index, rolling_bot5_low, label='Low  TTM', linewidth=1)

    # format x-axis: a tick every other year
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()  # rotate labels

    ax.legend()
    # ax.set_title(f'Rolling average of the lowest {alpha*100:.1f}% of #K over the past year')
    ax.set_xlabel('Date')
    ax.set_ylabel('# Strikes')
    ax.grid(alpha=0.5)
    ax.set_ylim(0, None)

    ax.yaxis.set_major_locator(MultipleLocator(10))

    plt.tight_layout()

    import os
    if savefig:
        os.makedirs("figures/summary", exist_ok=True)
        plt.savefig(f"figures/summary/lowest number of strikes timeseries.pdf")

    plt.show()


def plot_ticker_SW_vs_vix(df, ticker, figsize = (10, 6), show_fig = True, save_fig = False, filename_suffix = ""):
    vol_symbol = load_clean_lib.ticker_to_vol_symbol(ticker)
    df_tmp = df[df["ticker"] == ticker].copy()
    df_tmp = df_tmp[df_tmp[vol_symbol].isna() == False]

    plt.figure(figsize=figsize)
    plt.plot(df_tmp["date"], df_tmp["SW_0_30"], label=f"SW ({ticker})", alpha=1, lw=0.5)
    plt.plot(df_tmp["date"], df_tmp[vol_symbol] ** 2, label=rf"{vol_symbol}$^2$", alpha=1, lw=0.5)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    import os
    if save_fig:
        os.makedirs("figures/vix", exist_ok=True)
        plt.savefig(f"figures/vix/ticker SW vs vix ({ticker}){filename_suffix}.pdf")

    if show_fig:
        plt.show()
    else:
        plt.close()  # Close figure if not shown

    return


def plot_ticker_SW_minus_vix(df, ticker, figsize = (10, 6), show_fig = True, save_fig = False, filename_suffix = ""):
    vol_symbol = load_clean_lib.ticker_to_vol_symbol(ticker)
    df_tmp = df[df["ticker"] == ticker].copy()
    df_tmp = df_tmp[df_tmp[vol_symbol].isna() == False]

    plt.figure(figsize=figsize)
    plt.plot(df_tmp["date"], df_tmp[vol_symbol] ** 2 - df_tmp["SW_0_30"], label=rf"${vol_symbol}^2 - SW ({ticker})$", alpha=1, lw=0.5)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    import os
    if save_fig:
        os.makedirs("figures/vix", exist_ok=True)
        plt.savefig(f"figures/vix/ticker SW - vix ({ticker}){filename_suffix}.pdf")

    if show_fig:
        plt.show()
    else:
        plt.close()  # Close figure if not shown

    return


def plot_ticker_SW_minus_vix_scaled(df, ticker, figsize = (10, 6), show_fig = True, save_fig = False, filename_suffix = ""):
    vol_symbol = load_clean_lib.ticker_to_vol_symbol(ticker)
    df_tmp = df[df["ticker"] == ticker].copy()
    df_tmp = df_tmp[df_tmp[vol_symbol].isna() == False]

    plt.figure(figsize=figsize)
    plt.plot(df_tmp["date"], (df_tmp[vol_symbol] ** 2 - df_tmp["SW_0_30"])/(df_tmp[vol_symbol] ** 2), label=rf"${vol_symbol}^2 - SW ({ticker})$", alpha=1, lw=0.5)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    import os
    if save_fig:
        os.makedirs("figures/vix", exist_ok=True)
        plt.savefig(f"figures/vix/ticker SW - vix scaled ({ticker}){filename_suffix}.pdf")

    if show_fig:
        plt.show()
    else:
        plt.close()  # Close figure if not shown

    return




def get_unique_tickers(list_of_lists):
    unique_tickers = set()
    for lst in list_of_lists:
        unique_tickers.update(lst)
    return list(unique_tickers)


_ASSET_CLASS = {
    # Equity Markets (0)
    'SPX': 0, 'OEX': 0, 'OEF': 0, "DJX": 0, 'QQQ': 0, 'NDX': 0, 'SPY':0, 'DIA':0, 'VGK': 0, 'FXI': 0, 'EWJ': 0, 'EWZ': 0,
    'INDA': 0, 'EZA': 0, 'EWC': 0, 'EWU': 0, 'EWY': 0, 'EEM': 0,
    'EWA': 0, 'EWW': 0, 'VNQ': 0, "EFA": 0,
    # Fixed Income (1)
    'TLT': 1, 'SHY': 1, 'TIP': 1, 'LQD': 1, 'HYG': 1, 'EMB': 1,
    # Commodities (2)
    'IAU': 2, 'SLV': 2, 'UNG': 2, 'USO': 2, "GLD": 2,
    # Currency (3)
    'UUP': 3, 'FXE': 3, 'FXY': 3, 'CEW': 3,
    # None of the above
    'UVXY': 4, 'BITO': 4,
}
def ticker_to_asset_code(ticker: str) -> int:
    """
    Return:
      0 for Equity Markets,
      1 for Fixed Income,
      2 for Commodities,
      3 for Currency,
      4 for anything else.
    """
    return _ASSET_CLASS.get(ticker, 4)


# ticker_to_name_Cross_AM = {
#     # Indexes
#     "SPX": "S&P 500 Index",
#     "OEX": "S&P 100 Index",
#     "NDX": "NASDAQ 100 Index",
#     "DJX": "Dow Jones Ind. Average Index",
#
#     # Major ETFs
#     "SPY": "S&P 500 ETF Trust",
#     "OEF": "S&P 100 ETF",
#     "QQQ": "NASDAQ 100 Index ETF",
#     "DIA": "Dow Jones Ind. Average ETF",
#     "IWM": "Russell 2000 ETF",
#     "EEM": "Emerging Markets ETF",
#
#     # Cross‐market ETFs
#     "VGK": "Europe ETF",
#     "FXI": "China ETF",
#     "EWJ": "Japan ETF",
#     "EWZ": "Brazil ETF",
#     "INDA":"India ETF",
#     "EZA": "South Africa ETF",
#     "EWC": "Canada ETF",
#     "EWU": "United Kingdom ETF",
#     "EWY": "South Korea ETF",
#     "EWA": "Australia ETF",
#     "EWW": "Mexico ETF",
#     "VNQ": "Real Estate ETF",
#     "TIP": "TIPS Bond ETF",
#     "LQD": "Investment Grade Corp. Bond ETF",
#     "HYG": "High Yield Corp. Bond ETF",
#     "EMB": "Emerging Markets Bond ETF",
#     "IAU": "Gold Trust",
#     "SLV": "Silver Trust",
#     "UNG": "Natural Gas Fund",
#     "USO": "Oil Fund",
#     "UVXY": "VIX Short-Term Futures ETF",
#     "UUP": " US Dollar Trust",
#     "FXE": " Euro Trust",
#     "FXY": " Japanese Yen Trust",
#     "BITO": "Bitcoin ETF",
# }

# manual overrides: map ticker → ticker_out
ticker_to_ticker_out = {
    "DD": "DD (Nem)",
    "DD_eidp": "DD (Eid)",
    "DOW_chem": "DOW (Che)",
    "DOW_inc": "DOW (Inc)",
    "EWU": "EWU (new)",
    "EWU_old": "EWU (old)",
    "S": "S (Sea)",
    "S_sears": "S (Spr)",
    "T": "T (Inc)",
    "T_old": "T (Cor)",
}

name_overrides = {
    # Stocks
    "GOOGL": "Alphabet Inc A",
    "GOOG": "Alphabet Inc C",
    "CMCSA": "Comcast Corp A",
    "CMCSK": "Comcast Corp K",
    # "IBM":      "International Business Machines Corp",
    "SGP": "Schering-Plough Corp (Pre MRK merger)",
    "TFCFA": "Twenty-First Century Fox Inc A",
    "TFCF": "Twenty-First Century Fox Inc B",
    "UBAN": "US Bancorp Inc",
    "USB": "US Bancorp Delaware",
    "LIN": "Linde plc",
    "GX": "Global Crossing LTD",
    "JCI": "Johnson Controls International plc",
    "MDT": "Medtronic plc",
    "COV": "Covidien plc",
    "ACN": "Accenture plc",

    # Indexes
    "SPX": "S&P 500 Index",
    "OEX": "S&P 100 Index",
    "NDX": "NASDAQ 100 Index",
    "DJX": "Dow Jones Ind. Average Index",
    "INDU": "Dow Jones Ind. Average Index",
    "RUS": "Russell 2000 Index",

    # Major ETFs
    "SPY": "S&P 500 ETF",
    "OEF": "S&P 100 ETF",
    "QQQ": "NASDAQ 100 ETF",
    "DIA": "Dow Jones Ind. Average ETF",
    "IWM": "Russell 2000 ETF",

    # Cross-market ETPs
    "VGK": "Europe ETF",
    "EEM": "Emerging Markets ETF",
    "FXI": "China ETF",
    "EWJ": "Japan ETF",
    "EWZ": "Brazil ETF",
    "INDA": "India ETF",
    "EZA": "South Africa ETF",
    "EWC": "Canada ETF",
    "EWU": "United Kingdom ETF (new)",
    "EWU_old": "United Kingdom ETF (old)",
    "EWU_combined": "United Kingdom ETF",
    "EWY": "South Korea ETF",
    "EWA": "Australia ETF",
    "EWW": "Mexico ETF",
    "VNQ": "Real Estate ETF",
    "TIP": "TIPS Bond ETF",
    "LQD": "Investment Grade Corp. Bond ETF",
    "HYG": "High Yield Corp. Bond ETF",
    "EMB": "Emerging Markets Bond ETF",
    "IAU": "Gold Trust",
    "SLV": "Silver Trust",
    "UNG": "Natural Gas Fund",
    "USO": "Oil Fund",
    "UVXY": "VIX Short-Term Futures ETF",
    "UUP": "US Dollar Trust",
    "FXE": "Euro Trust",
    "FXY": "Japanese Yen Trust",
    "BITO": "Bitcoin ETF",
    "CEW": "Emerging Currency ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "EFA": "EAFE: Dev. mkt. ex. U.S. and Canada",
    "GLD":  "Gold ETF",
}

def ticker_list_to_ordered_map(ticker_list):
    base_dir      = load_clean_lib.Option_metrics_path_from_profile()
    bloomberg_dir = base_dir / "Tickers" / "index data"
    security_map  = pd.read_excel(bloomberg_dir / "Security_map.xlsx")
    # keep only tickers in the list, then reindex to preserve the list order
    security_map = (
        security_map
        .set_index("ticker")
        .loc[ticker_list]          # filters & orders by ticker_list
        .reset_index()             # bring 'ticker' back as a column
    )
    return security_map


def Index_ETF_stock(name):
    if "Index" in name:
        return 0
    if "ETF" in name:
        return 1
    return 2
def Index_ETF_stock_list(name_list):
    return [Index_ETF_stock(name) for name in name_list]



# from pathlib import Path
# def ticker_to_name_CSV(ticker):
#     base_dir = load_clean_lib.Option_metrics_path_from_profile()
#     df_path = Path(base_dir) / "Tickers" / "Input" / ticker / "returns and stock price.csv"
#
#     if ticker not in OEX_tickers: #Cross_AM_tickers + Liquid_ETF_Idx_tickers:
#         name = ticker_to_name_Cross_AM.get(ticker, ticker)
#     else:
#         # ensure the file exists
#         if not df_path.is_file():
#             print(f"No CSV found for ticker '{ticker}' at {df_path}")
#             return ''
#
#         df = pd.read_csv(df_path)
#
#         # grab the first value in the 'issuer' column and return it
#         name = df['issuer'].iloc[0]
#         name = name.title()
#
#     return name

# def ticker_list_to_name_list(ticker_list):
#     return [ticker_to_name_CSV(ticker) for ticker in ticker_list]



from pathlib import Path
def dirs_to_list():
    base_dir   = load_clean_lib.Option_metrics_path_from_profile()
    ticker_dir = Path(base_dir) / "Tickers" / "Input"
    subdirs    = [p.name for p in ticker_dir.iterdir() if p.is_dir()]
    return subdirs





# np.array(vp.dirs_to_list())


# removed tickers = ["TLT", "SHY", "CEW"] (TLT removed from VIX and Cross-AM, SHY and CEW removed from Cross-AM)
OEX_tickers_old = ["OEX", "OEF", "AA", "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AEP", "AES", "AGN", "AIG", "ALL", "AMD", "AMGN", "AMT", "AMZN", "APA", "APC", "ATI", "AVGO", "AVP", "AXP", "BA", "BAC", "BAX", "BDK", "BHI", "BIIB", "BK", "BKNG", "BMY", "BNI", "BRK", "BUD", "C", "CAT", "CELG", "CGP", "CHTR", "CI", "CL", "CLB", "CMCSA", "CMCSK", "COF", "COP", "COST", "COV", "CPB", "CRM", "CSC", "CSCO", "CVS", "CVX", "DAL", "DD", "DE", "DELL", "DHR", "DIS", "DOW", "DUK", "DVN", "EBAY", "EMC", "EMR", "ENE", "EP", "ETR", "EXC", "F", "FCX", "FDX", "G", "GBLX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "GTX", "HAL", "HCA", "HD", "HET", "HIG", "HNZ", "HON", "HPQ", "HSH", "IBM", "INTC", "IP", "JCI", "JNJ", "JPM", "KHC", "KMI", "KO", "LEH", "LIN", "LLY", "LMT", "LOW", "MA", "MAY", "MCD", "MDLZ", "MDT", "MER", "MET", "META", "MMM", "MO", "MON", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOV", "NSC", "NSM", "NT", "NVDA", "NXTL", "NYX", "OMX", "ONE", "ORCL", "OXY", "PARA", "PEP", "PFE", "PG", "PM", "PNU", "PYPL", "QCOM", "RAL", "RF", "ROK", "RSH", "RTN", "RTX", "S", "SBUX", "SCHW", "SGP", "SLB", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TOY", "TSLA", "TWX", "TXN", "UBAN", "UIS", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WB", "WBA", "WFC", "WMB", "WMT", "WY", "WYE", "XOM", "XRX"]
Cross_AM_tickers_old =  ["SPX", "VGK", "FXI", "EWJ", "EWZ", "INDA", "EZA", "EWC", "EEM", "EWU", "EWY", "EWA", "EWW", "VNQ", "TIP", "LQD", "HYG", "EMB", "IAU", "SLV", "UNG", "USO", "UVXY", "UUP", "FXE", "FXY", "BITO" ]
VIX_tickers_old = ["SPX", "GOOGL", "AMZN", "DIA", "IWM", "QQQ", "AAPL", "EWZ", "USO", "GS", "EEM", "FXE", "SLV", "IBM"]
DJX_tickers_old = ["DJX", "AA", "AAPL", "AIG", "AMGN", "AMZN", "AXP", "BA", "BAC", "C", "CAT", "CRM", "CSCO", "CVX", "DD", "DIS", "DOW", "GE", "GM", "GS", "GT", "HD", "HON", "HPQ", "IBM", "INTC", "IP", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT", "NKE", "PFE", "PG", "RTX", "S", "T", "TRV", "UK", "UNH", "V", "VZ", "WBA", "WMT", "XOM"]
Index_tickers = ["SPX", "OEX", "NDX", "DJX"]
ETF_tickers   = ["SPY", "OEF", "QQQ", "DIA"]

Liquid_stock_tickers_old = ['GOOG', 'BKNG', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'MA', 'PYPL', 'AAPL', 'AVGO', 'CHTR', 'NVDA', 'CRM', 'ABBV', 'HCA', 'V', 'GS', 'WB', 'GM', 'PM', 'MSFT', 'PARA', 'BA', 'ADBE', 'TMUS', 'OXY', 'MET', 'DE', 'DAL', 'BIIB', 'JPM', 'CAT', 'DIS', 'COST', 'COF', 'IBM', 'QCOM', 'GILD', 'ACN', 'UNH', 'C']
Liquid_ETF_Idx_tickers_old = Index_tickers + ETF_tickers
Liquid_tickers_old = Liquid_ETF_Idx_tickers_old + Liquid_stock_tickers_old
ALL_tickers_old = get_unique_tickers([OEX_tickers_old, Cross_AM_tickers_old, VIX_tickers_old, DJX_tickers_old, Liquid_ETF_Idx_tickers_old])



# Manual tickers
# all
All_tickers_original = ["NSM", "G", "DOW_chem", "DD_eidp", "LU", "MEDI", "EMC", "CCU", "UBAN", "HCA", "ONE", "S_sears", "T_old", "PHA", "AGC", "MAY", "AA", "AEP", "AES", "AIG", "AMGN", "ATI", "AVP", "AXP", "BA", "BAC", "BAX", "BBWI", "BDK", "BHGE", "BMY", "BNI", "C", "CGP", "CI", "CL", "CPB", "DXC", "CSCO", "DAL", "DIS", "EK", "ENE", "ETR", "EXC", "F", "FDX", "GD", "GE", "HAL", "HD", "HET", "HIG", "HNZ", "HON", "HPQ", "HSH", "IBM", "INTC", "IP", "JNJ", "JPM", "KO", "LEH", "MCD", "MER", "MMM", "MRK", "MS", "MSFT", "NT", "NSC", "NXTL", "OMX", "ORCL", "PARA", "PEP", "PFE", "PG", "RAL", "ROK", "RSH", "RTN", "RTX", "SLB", "SO", "TOY", "TWX", "TXN", "UIS", "VZ", "WFC", "WMB", "WMT", "WY", "XOM", "XRX", "EP", "USB", "T", "BUD", "MO", "GS", "ALL", "DELL", "CMCSA", "S", "ABT", "CAT", "TGT", "CVX", "UPS", "WB", "COF", "COP", "GOOGL", "RF", "CVS", "AAPL", "MDLZ", "BK", "NYX", "PM", "UNH", "NOV", "MA", "OXY", "QCOM", "DVN", "GILD", "LMT", "LOW", "NKE", "SGP", "WBA", "WYE", "MON", "AMZN", "COST", "MET", "FCX", "TFCFA", "BRK", "APA", "EMR", "UNP", "V", "APC", "EBAY", "LLY", "SBUX", "SPG", "ABBV", "GM", "META", "BIIB", "GOOG", "CELG", "KMI", "BKNG", "CMCSK", "PYPL", "TFCF", "BLK", "DHR", "DUK", "NEE", "KHC", "CHTR", "DD", "NFLX", "NVDA", "GTX", "ADBE", "DOW_inc", "TMO", "AMT", "CRM", "TSLA", "AVGO", "TMUS", "SCHW", "AMD", "DE", "INTU", "UK", "GT", "TRV", "SHW", "LIN", "GX", "JCI", "MDT", "COV", "ACN", "OEX", "SPX", "DJX", "BITO", "CEW", "DIA", "EEM", "EMB", "EWA", "EWC", "EWJ", "EWU_old", "EWW", "EWY", "EWZ", "EZA", "FXE", "FXI", "FXY", "HYG", "IAU", "INDA", "INDU", "IWM", "LQD", "OEF", "QQQ", "SHY", "SLV", "SPY", "TIP", "TLT", "USO", "UUP", "VGK", "VNQ", "UNG", "UVXY", "NDX", "EWU"]
All_tickers =  ["GLD", "EFA", "RUT","NSM", "G", "DOW_chem", "DD_eidp", "LU", "MEDI", "EMC", "CCU", "UBAN", "HCA", "ONE", "S_sears", "T_old", "PHA", "AGC", "MAY", "AA", "AEP", "AES", "AIG", "AMGN", "ATI", "AVP", "AXP", "BA", "BAC", "BAX", "BBWI", "BDK", "BHGE", "BMY", "BNI", "C", "CGP", "CI", "CL", "CPB", "DXC", "CSCO", "DAL", "DIS", "EK", "ENE", "ETR", "EXC", "F", "FDX", "GD", "GE", "HAL", "HD", "HET", "HIG", "HNZ", "HON", "HPQ", "HSH", "IBM", "INTC", "IP", "JNJ", "JPM", "KO", "LEH", "MCD", "MER", "MMM", "MRK", "MS", "MSFT", "NT", "NSC", "NXTL", "OMX", "ORCL", "PARA", "PEP", "PFE", "PG", "RAL", "ROK", "RSH", "RTN", "RTX", "SLB", "SO", "TOY", "TWX", "TXN", "UIS", "VZ", "WFC", "WMB", "WMT", "WY", "XOM", "XRX", "EP", "USB", "T", "BUD", "MO", "GS", "ALL", "DELL", "CMCSA", "S", "ABT", "CAT", "TGT", "CVX", "UPS", "WB", "COF", "COP", "GOOGL", "RF", "CVS", "AAPL", "MDLZ", "BK", "NYX", "PM", "UNH", "NOV", "MA", "OXY", "QCOM", "DVN", "GILD", "LMT", "LOW", "NKE", "SGP", "WBA", "WYE", "MON", "AMZN", "COST", "MET", "FCX", "TFCFA", "BRK", "APA", "EMR", "UNP", "V", "APC", "EBAY", "LLY", "SBUX", "SPG", "ABBV", "GM", "META", "BIIB", "GOOG", "CELG", "KMI", "BKNG", "CMCSK", "PYPL", "TFCF", "BLK", "DHR", "DUK", "NEE", "KHC", "CHTR", "DD", "NFLX", "NVDA", "GTX", "ADBE", "DOW_inc", "TMO", "AMT", "CRM", "TSLA", "AVGO", "TMUS", "SCHW", "AMD", "DE", "INTU", "UK", "GT", "TRV", "SHW", "LIN", "GX", "JCI", "MDT", "COV", "ACN", "OEX", "SPX", "DJX", "BITO", "CEW", "DIA", "EEM", "EMB", "EWA", "EWC", "EWJ", "EWW", "EWY", "EWZ", "EZA", "FXE", "FXI", "FXY", "HYG", "IAU", "INDA", "INDU", "IWM", "LQD", "OEF", "QQQ", "SHY", "SLV", "SPY", "TIP", "TLT", "USO", "UUP", "VGK", "VNQ", "UNG", "UVXY", "NDX", "EWU_combined"]
DJX_tickers = ["DJX", "UK", "DD_eidp", "JPM", "S_sears", "AA", "AXP", "BA", "C", "CAT", "CVX", "DIS", "EK", "GE", "GT", "HON", "HPQ", "IBM", "IP", "JNJ", "KO", "MCD", "MMM", "MO", "MRK", "PG", "RTX", "T", "WMT", "XOM", "T_old", "HD", "INTC", "MSFT", "AIG", "PFE", "VZ", "BAC", "MDLZ", "CSCO", "TRV", "UNH", "GS", "NKE", "V", "AAPL", "DD", "WBA", "DOW_inc", "AMGN", "CRM", "AMZN", "NVDA", "SHW"]
DJX_tickers_constituents = ["UK", "DD_eidp", "JPM", "S_sears", "AA", "AXP", "BA", "C", "CAT", "CVX", "DIS", "EK", "GE", "GT", "HON", "HPQ", "IBM", "IP", "JNJ", "KO", "MCD", "MMM", "MO", "MRK", "PG", "RTX", "T", "WMT", "XOM", "T_old", "HD", "INTC", "MSFT", "AIG", "PFE", "VZ", "BAC", "MDLZ", "CSCO", "TRV", "UNH", "GS", "NKE", "V", "AAPL", "DD", "WBA", "DOW_inc", "AMGN", "CRM", "AMZN", "NVDA", "SHW"]
OEX_tickers = ["OEX", "OEF", "NSM", "G", "DOW_chem", "DD_eidp", "LU", "MEDI", "EMC", "CCU", "UBAN", "HCA", "ONE", "S_sears", "T_old", "PHA", "AGC", "MAY", "AA", "AEP", "AES", "AIG", "AMGN", "ATI", "AVP", "AXP", "BA", "BAC", "BAX", "BBWI", "BDK", "BHGE", "BMY", "BNI", "C", "CGP", "CI", "CL", "CPB", "DXC", "CSCO", "DAL", "DIS", "EK", "ENE", "ETR", "EXC", "F", "FDX", "GX", "GD", "GE", "HAL", "HD", "HET", "HIG", "HNZ", "HON", "HPQ", "HSH", "IBM", "INTC", "IP", "JCI", "JNJ", "JPM", "KO", "LEH", "MCD", "MER", "MMM", "MRK", "MS", "MSFT", "NT", "NSC", "NXTL", "OMX", "ORCL", "PARA", "PEP", "PFE", "PG", "RAL", "ROK", "RSH", "RTN", "RTX", "SLB", "SO", "TOY", "TWX", "TXN", "UIS", "VZ", "WFC", "WMB", "WMT", "WY", "XOM", "XRX", "EP", "USB", "T", "BUD", "MDT", "MO", "GS", "ALL", "DELL", "CMCSA", "S", "ABT", "CAT", "TGT", "CVX", "UPS", "WB", "COF", "COP", "GOOGL", "RF", "CVS", "AAPL", "MDLZ", "BK", "COV", "NYX", "PM", "UNH", "NOV", "MA", "OXY", "QCOM", "DVN", "GILD", "LMT", "LOW", "NKE", "SGP", "WBA", "WYE", "MON", "AMZN", "COST", "MET", "FCX", "TFCFA", "BRK", "APA", "EMR", "UNP", "V", "ACN", "APC", "EBAY", "LLY", "SBUX", "SPG", "ABBV", "GM", "META", "BIIB", "GOOG", "CELG", "KMI", "BKNG", "CMCSK", "PYPL", "TFCF", "BLK", "DHR", "DUK", "NEE", "KHC", "CHTR", "DD", "NFLX", "NVDA", "GTX", "ADBE", "DOW_inc", "TMO", "AMT", "CRM", "TSLA", "AVGO", "LIN", "TMUS", "SCHW", "AMD", "DE", "INTU"]
OEX_tickers_constituents = ["NSM", "G", "DOW_chem", "DD_eidp", "LU", "MEDI", "EMC", "CCU", "UBAN", "HCA", "ONE", "S_sears", "T_old", "PHA", "AGC", "MAY", "AA", "AEP", "AES", "AIG", "AMGN", "ATI", "AVP", "AXP", "BA", "BAC", "BAX", "BBWI", "BDK", "BHGE", "BMY", "BNI", "C", "CGP", "CI", "CL", "CPB", "DXC", "CSCO", "DAL", "DIS", "EK", "ENE", "ETR", "EXC", "F", "FDX", "GX", "GD", "GE", "HAL", "HD", "HET", "HIG", "HNZ", "HON", "HPQ", "HSH", "IBM", "INTC", "IP", "JCI", "JNJ", "JPM", "KO", "LEH", "MCD", "MER", "MMM", "MRK", "MS", "MSFT", "NT", "NSC", "NXTL", "OMX", "ORCL", "PARA", "PEP", "PFE", "PG", "RAL", "ROK", "RSH", "RTN", "RTX", "SLB", "SO", "TOY", "TWX", "TXN", "UIS", "VZ", "WFC", "WMB", "WMT", "WY", "XOM", "XRX", "EP", "USB", "T", "BUD", "MDT", "MO", "GS", "ALL", "DELL", "CMCSA", "S", "ABT", "CAT", "TGT", "CVX", "UPS", "WB", "COF", "COP", "GOOGL", "RF", "CVS", "AAPL", "MDLZ", "BK", "COV", "NYX", "PM", "UNH", "NOV", "MA", "OXY", "QCOM", "DVN", "GILD", "LMT", "LOW", "NKE", "SGP", "WBA", "WYE", "MON", "AMZN", "COST", "MET", "FCX", "TFCFA", "BRK", "APA", "EMR", "UNP", "V", "ACN", "APC", "EBAY", "LLY", "SBUX", "SPG", "ABBV", "GM", "META", "BIIB", "GOOG", "CELG", "KMI", "BKNG", "CMCSK", "PYPL", "TFCF", "BLK", "DHR", "DUK", "NEE", "KHC", "CHTR", "DD", "NFLX", "NVDA", "GTX", "ADBE", "DOW_inc", "TMO", "AMT", "CRM", "TSLA", "AVGO", "LIN", "TMUS", "SCHW", "AMD", "DE", "INTU"]

# Cross-AM, VIX and Liquid ticker sets
Cross_AM_tickers = ['GLD', 'EFA', "SPX", "VGK", "EEM", "FXI", "EWJ", "EWZ", "INDA", "EZA", "EWC", "EWU_combined", "EWY", "EWA", "EWW", "VNQ", "TIP", "LQD", "HYG", "EMB", "IAU", "SLV", "UNG", "USO", "UVXY", "UUP", "FXE", "FXY", "BITO", "CEW", "SHY", "TLT"]
VIX_tickers = ['GLD', 'EFA', 'RUT', 'IBM', 'GS', 'AAPL', 'AMZN', 'GOOG', 'SPX', 'DJX', 'EEM', 'EWZ', 'TLT', 'USO', 'NDX']
Liquid_ETF_Idx_tickers = Index_tickers + ETF_tickers
Liquid_stock_tickers = ['GOOG', 'BKNG', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NFLX',
       'PYPL', 'MA', 'AAPL', 'AVGO', 'CHTR', 'NVDA', 'CRM', 'ABBV', 'V',
       'GS', 'BLK', 'GM', 'PM', 'BA', 'TMUS', 'MSFT', 'PARA', 'ADBE',
       'DE', 'BIIB', 'DIS', 'JPM', 'CAT', 'OXY', 'COST', 'MET', 'ACN',
       'COF', 'IBM', 'GILD', 'C', 'QCOM', 'UNH', 'CVS', "WFC"]
Liquid_tickers = Liquid_ETF_Idx_tickers + Liquid_stock_tickers











Market_tickers = DJX_tickers + OEX_tickers


Car_wu_tickers = ["SPX", "OEX", "DJX", "NDX", "QQQ", "MSFT", "INTC", "IBM", "AMER", "DELL",
    "CSCO", "GE", "CPQ", "YHOO", "SUNW", "MU", "MO", "AMZN", "ORCL", "LU",
    "TRV", "WCOM", "TYC", "AMAT", "QCOM", "TXN", "PFE", "MOT", "EMC", "HWP",
    "AMGN", "BRCM", "MER", "NOK", "CHL", "UNPH", "EBAY", "JNPR", "CIEN", "BRCD"
]
# Output_tickers = dirs_to_list()


def get_company_name(ticker):
    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    ticker_data_path = base_dir / "Tickers" / "ticker_info.csv"

    # Load the CSV into a DataFrame
    df = pd.read_csv(ticker_data_path, dtype=str)

    # Normalize the symbol column and the lookup ticker to uppercase
    df['Symbol'] = df['Symbol'].str.upper()
    ticker = ticker.upper()

    # Find the row where Symbol matches, then return the Name column
    match = df.loc[df['Symbol'] == ticker, 'Name']
    if not match.empty:
        name = match.iloc[0]
        name = name.replace(" New Common Stock", "")
        name = name.replace(" Common Stock", "")
        name = name.replace("Class A", "")
        name = name.replace("Class B", "")
        name = name.replace(" (The)", "")
        return name
    return None



