import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import sys
import time

def load_option_data(file_path):
    columns_to_load = [
        "ticker",
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
        "secid"
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)

    df.rename(columns={"strike_price": "K"}, inplace=True)

    # Format columns
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d', errors='coerce')
    df['cp_flag'] = df['cp_flag'].astype('string')
    df['ticker'] = df['ticker'].astype('string')
    # df['exercise_style'] = df['exercise_style'].astype('string')
    df['K'] /= 1000

    unique_trading_days = df[df["ticker"] == "SPX"]["date"].drop_duplicates().sort_values()  # [df["ticker"] == "SPX"]

    # Create a mapping of date to trading day number
    trading_day_map = {date: n for n, date in enumerate(unique_trading_days)}

    total_trading_days = unique_trading_days.nunique()
    total_calendar_days = (unique_trading_days.max() - unique_trading_days.min()).days + 1
    average_trading_days_per_year = (total_trading_days * 365) / total_calendar_days

    # Add the n_trading_day column to the DataFrame
    df["n_trading_day"] = df["date"].map(trading_day_map)
    # df["n_trading_day_exdate"] = df["exdate"].map(trading_day_map)

    # Calendar (_cal)
    df["days"] = (df["exdate"] - df["date"]).dt.days
    # df["TTM"] = df["days"] / 365

    # Trading (_trd)
    # df["TTM_trd"] = (df["n_trading_day_exdate"] - df["n_trading_day"]) / round(average_trading_days_per_year)

    return df


def load_implied_volatility(file_path):
    columns_to_load = [
        "secid",
        "ticker",
        "date",
        "days",
        "delta",
        "impl_volatility",
        "cp_flag",
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)

    # Format columns
    df['ticker'] = df['ticker'].astype('string')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['cp_flag'] = df['cp_flag'].astype('string')
    return df


def load_realized_volatility(file_path):
    columns_to_load = [
        "secid",
        "ticker",
        "date",
        "days",
        "volatility"
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)

    # Format columns
    df['ticker'] = df['ticker'].astype('string')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    return df


def load_forward_price(file_path):
    columns_to_load = [
        "secid",
        "ticker",
        "date",
        "expiration",
        "ForwardPrice"
    ]

    # Load only the specified columns
    df = pd.read_csv(file_path, usecols=columns_to_load)

    # Format columns
    df['ticker'] = df['ticker'].astype('string')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y-%m-%d', errors='coerce')

    df["days"] = (df['expiration'] - df['date']).dt.days

    return df


def load_ZC_yield_curve(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['rate'] /= 100
    return df



def add_FW_to_od(od, FW):
    # Pre-group FW by date and ticker, and store sorted arrays for interpolation
    FW_grouped = {}
    for key, group in FW.groupby(["date", "ticker"]):
        group_sorted = group.sort_values("days")
        FW_grouped[key] = (group_sorted["days"].values, group_sorted["ForwardPrice"].values)

    def interpolate_group(group):
        # Each group in od corresponds to one (date, ticker) pair
        key = (group["date"].iloc[0], group["ticker"].iloc[0])
        if key not in FW_grouped:
            raise ValueError(f"No FW data for key: {key}")
        x, y = FW_grouped[key]

        # Check that all target days are within the FW range (to mimic your error condition)
        if group["days"].min() < x[0] or group["days"].max() > x[-1]:
            raise ValueError(f"Not enough data points to interpolate for some days in group {key}")

        # Use numpy's vectorized interpolation for the entire group
        group.loc[:, "F"] = np.interp(group["days"], x, y)
        return group

    # Apply the interpolation function to each group in od
    od = od.groupby(["date", "ticker"], group_keys=False).apply(interpolate_group)
    return od

# old method for add_FW_to_od, leads to the exact same forward rates ect. but takes ~133 times longer for 8 years. Scales very poorly O(n^3) i think
    # import sys
    # import time
    #
    # # Calculate forwards FW
    # start_time = time.time()
    # for current_date in od["date"].unique():
    #     FW_date = FW[FW["date"] == current_date]
    #     od_index = od["date"] == current_date  # Store index for efficient updates
    #     tickers_date = od.loc[od_index, "ticker"].unique()
    #
    #     for current_ticker in tickers_date:
    #         FW_ticker = FW_date[FW_date["ticker"] == current_ticker]
    #         ticker_index = od_index & (od["ticker"] == current_ticker)
    #         days_ticker = od.loc[ticker_index, "days"].unique()
    #
    #         for days in days_ticker:
    #             F = vp.interpolated_FW(FW_ticker, days)
    #             od.loc[ticker_index & (od["days"] == days), "F"] = F  # Update in-place
    #
    #     # print timing
    #     elapsed_time = time.time() - start_time
    #     sys.stdout.write(f"\rFW: Running date: {current_date} | Tickers: {len(tickers_date)} | Time elapsed: {elapsed_time:.2f} seconds")
    #     sys.stdout.flush()


def add_r_to_od(od, ZCY_curves):
    def compute_rates_for_date(group):
        current_date = group["date"].iloc[0]
        # Compute the yield curve for the current date once
        ZCY_date = interpolate_yield_curve_between_dates(ZCY_curves, current_date)
        ZCY_date = ZCY_date.sort_values("days")
        x = ZCY_date["days"].values
        y = ZCY_date["rate"].values

        # Get the target 'days' values for this group
        target_days = group["days"].values
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


def process_group_activity_summary(group):
    # Extract the current date and ticker from the group
    current_date = group["date"].iloc[0]
    current_ticker = group["ticker"].iloc[0]
    summary = {}

    # Initialize the new columns to False
    group["low"] = False
    group["high"] = False

    # Get sorted unique days for this (date, ticker) pair
    unique_days = np.sort(group["days"].unique())
    summary["#days"] = len(unique_days)

    # Inactive if there are fewer than 2 unique days
    if len(unique_days) < 2:
        summary["Active"] = False
        summary["Inactive reason"] = "unique(_days) < 2"
        return group, summary

    # Inactive if the minimum day is > 90
    if unique_days[0] > 90:
        summary["Active"] = False
        summary["Inactive reason"] = "min days > 90"
        return group, summary

    # Select two lowest days above 7 days.
    # If the smallest day is <= 8 and at least 3 unique days exist, skip it.
    low_2_days = list(unique_days[:2])
    if low_2_days[0] <= 8:
        if len(unique_days) < 3:
            summary["Active"] = False
            summary["Inactive reason"] = "min days <= 8 & len < 3"
            return group, summary
        low_2_days = list(unique_days[1:3])

    summary["low days"] = low_2_days[0]
    summary["high days"] = low_2_days[1]

    # Check unique strike counts for each selected day; require at least 3
    active = True
    for day, label in zip(low_2_days, ["low", "high"]):
        num_strikes = group.loc[group["days"] == day, "K"].nunique()
        summary[f"{label} #K"] = num_strikes
        if num_strikes < 3:
            active = False
            summary["Active"] = False
            summary["Inactive reason"] = "unique(K) < 3"
    if not active:
        return group, summary

    summary["Active"] = True
    summary["Inactive reason"] = ""

    # Set the 'low' and 'high' flags for the corresponding rows
    group.loc[group["days"] == low_2_days[0], "low"] = True
    group.loc[group["days"] == low_2_days[1], "high"] = True

    # # For each of the two selected days, update IV_option_price and moneyness
    # for day in low_2_days:
    #     TTM = day / 365.0
    #     mask = group["days"] == day
    #     # Calculate IV_option_price using the BSM model row-wise
    #     group.loc[mask, "IV_option_price"] = group.loc[mask].apply(
    #         lambda row: vp.BSM(
    #             row["F"],
    #             row["K"],
    #             TTM,
    #             row["impl_volatility"],
    #             row["r"],
    #             row["cp_flag"]
    #         ),
    #         axis=1
    #     )
    #     # Assuming F is defined in the outer scope (or alternatively use row["F"])
    #     group.loc[mask, "moneyness"] = np.log10(group.loc[mask, "F"] / group.loc[mask, "K"])

    return group, summary




def interpolate_yield_curve_between_dates(ZCY_curves, date):
    next_date_val = ZCY_curves[ZCY_curves['date'] >= date].head(1)['date'].iloc[0]
    prev_date_val = ZCY_curves[ZCY_curves['date'] <= date].tail(1)['date'].iloc[0]

    next_Curve = ZCY_curves[ZCY_curves['date'] == next_date_val]
    prev_Curve = ZCY_curves[ZCY_curves['date'] == prev_date_val]

    valid_days = set(next_Curve['days']).union(set(prev_Curve['days']))
    ZCY_date = pd.DataFrame({'days': sorted(valid_days)})
    ZCY_date['date'] = date

    next_date_diff = (next_date_val - date) / np.timedelta64(1, 'D')
    prev_date_diff = (date - prev_date_val) / np.timedelta64(1, 'D')
    total_diff = (next_date_val - prev_date_val) / np.timedelta64(1, 'D')

    if total_diff == 0:
        # Avoid division by zero: if the two dates are identical, use next_Curve rates directly.
        ZCY_date['rate'] = ZCY_date['days'].apply(lambda d: interpolated_ZCY(next_Curve, d))
    else:
        def interpolate_rate(days):
            next_rate = interpolated_ZCY(next_Curve, days)
            prev_rate = interpolated_ZCY(prev_Curve, days)
            return (next_rate * prev_date_diff + prev_rate * next_date_diff) / total_diff

        ZCY_date['rate'] = ZCY_date['days'].apply(interpolate_rate)
    return ZCY_date


# def interpolate_yield_curve_between_dates(ZCY_curves, date):
#     # Find the nearest dates before and after the target date
#     next_date_val = ZCY_curves[ZCY_curves['date'] >= date].head(1)['date'].iloc[0]
#     prev_date_val = ZCY_curves[ZCY_curves['date'] <= date].tail(1)['date'].iloc[0]
#
#     next_Curve = ZCY_curves[ZCY_curves['date'] == next_date_val]
#     prev_Curve = ZCY_curves[ZCY_curves['date'] == prev_date_val]
#
#     # Create a DataFrame with unique "days" values
#     valid_days = set(next_Curve['days']).union(set(prev_Curve['days']))
#     ZCY_date = pd.DataFrame({'days': sorted(valid_days)})
#     ZCY_date['date'] = date
#
#     # Calculate the time differences in days
#     next_date_diff = (next_date_val - date) / np.timedelta64(1, 'D')
#     prev_date_diff = (date - prev_date_val) / np.timedelta64(1, 'D')
#     total_diff = (next_date_val - prev_date_val) / np.timedelta64(1, 'D')
#
#     # Define a function to interpolate rate for a given "days" value
#     def interpolate_rate(days):
#         next_rate = interpolated_ZCY(next_Curve, days)
#         prev_rate = interpolated_ZCY(prev_Curve, days)
#         return (next_rate * prev_date_diff + prev_rate * next_date_diff) / total_diff
#
#     # Apply the interpolation function to each "days" value
#     ZCY_date['rate'] = ZCY_date['days'].apply(interpolate_rate)
#     return ZCY_date

def interpolated_ZCY(ZCY_date, target_days, date = None, filter_date = True):
    # Filter the DataFrame for the given date
    # if filter_date == True:
    #     ZCY_date = ZCY_curves[ZCY_curves['date'] == date]
    # else:
    #     ZCY_date = ZCY_curves
    # if ZCY_date.empty:
    #     print(f"No data available for the date {date}, it will be interpolated")
    #     ZCY_date = interpolate_yield_curve_between_dates(ZCY_curves, date)

    # Sort the DataFrame by days to find the nearest points
    # ZCY_date = ZCY_date.sort_values(by='days')

    # Find the two nearest points
    lower = ZCY_date[ZCY_date['days'] <= target_days].tail(1)
    upper = ZCY_date[ZCY_date['days'] >= target_days].head(1)

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
    x0, y0 = lower['days'].values[0], lower['rate'].values[0]
    x1, y1 = upper['days'].values[0], upper['rate'].values[0]

    # Perform linear interpolation
    if x0 == x1:
        return y0
    else:
        return y0 + (y1 - y0) * (target_days - x0) / (x1 - x0)


# likely irrelevant now.
def interpolated_FW(FW_date, target_days, date = None, ticker=None, filter_date = True):
    # # Filter the DataFrame for the given date
    # FW_date = FW_curves
    # if filter_date == False:
    #     FW_date = FW_curves[(FW_curves['date'] == date)]
    # if ticker != None:
    #     FW_date = FW_date[(FW_date['ticker'] == ticker)]
    # if FW_date.empty:
    #     raise ValueError(f"No data available for the date {date}")
    #
    # # Sort the DataFrame by days to find the nearest points
    # # FW_date = FW_date.sort_values(by='days')

    # Find the two nearest points
    lower = FW_date[FW_date['days'] <= target_days].tail(1)
    upper = FW_date[FW_date['days'] >= target_days].head(1)

    if lower.empty or upper.empty:
        raise ValueError(f"Not enough data points to interpolate for {target_days} days")

    # Extract the days and forward prices for interpolation
    x0, y0 = lower['days'].values[0], lower['ForwardPrice'].values[0]
    x1, y1 = upper['days'].values[0], upper['ForwardPrice'].values[0]

    # Perform linear interpolation
    if x0 == x1:
        return y0
    else:
        return y0 + (y1 - y0) * (target_days - x0) / (x1 - x0)


from tqdm import tqdm


# def process_od_rdy(od_rdy, calc_func, n_points=200):
#     from functools import partial
#
#     # Create a new function with n_points set to 200
#     calc_func_cust = partial(calc_func, n_points=n_points)
#
#     # Group by the unique combination of ticker, date, and days
#     grouped = list(od_rdy.groupby(["ticker", "date", "days"], group_keys=False))
#
#     # Initialize tqdm for progress tracking
#     results = []
#     for key, group in tqdm(grouped, total=len(grouped), desc="Processing Groups"):
#         results.append(calc_func_cust(group))
#
#     # Combine results into a single DataFrame
#     return pd.concat(results)

def fill_swap_rates(summary_dly_df, od_rdy, n_points=200):
    # 1) Calculate var_swap_rate on od_rdy
    df_swaps = process_od_rdy(od_rdy, replicate_SW, n_points=n_points)

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
    summary_dly_df.update(df_merged[["low SW", "high SW"]])

    return summary_dly_df


def process_od_rdy(od_rdy, calc_func, **kwargs):
    # Group by the unique combination of ticker, date, and days
    grouped = list(od_rdy.groupby(["ticker", "date", "days"], group_keys=False))

    # Initialize tqdm for progress tracking
    results = []
    for key, group in tqdm(grouped, total=len(grouped), desc="Processing Groups"):
        results.append(calc_func(group, **kwargs))

    # Combine results into a single DataFrame
    return pd.concat(results)


from scipy.stats import norm


def BSM_call_put(F, K, T, sigma, r, is_call=True):
    """
    Black–Scholes price for a European call or put.
    is_call=True => call, is_call=False => put
    """
    if T <= 0 or sigma <= 0 or K <= 0:
        return 0.0

    d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if is_call:
        return call_price
    else:
        # put-call parity: Put = Call - e^{-rT}(F-K)
        return call_price - np.exp(-r * T) * (F - K)


def replicate_SW(group, n_points = 100):
    """
    For each group (e.g. a single (date, ticker, maturity)), build an implied volatility
    curve on moneyness k = ln(K/F), interpolate to a dense grid of ±8 stdevs,
    price out-of-the-money options, and numerically integrate to approximate
    the variance swap rate per Carr & Wu (2009).

    Returns the group with a new column 'var_swap_rate'.
    """
    group = group.copy()

    # Extract relevant parameters (assuming they're constant within this group)
    F = group["F"].iloc[0]
    r = group["r"].iloc[0]
    T = group["days"].iloc[0] / 365  # T in years (e.g. days / 365)

    # 1) Average implied volatility as a rough stdev estimate
    #    (This is the "standard deviation" used to define ±8 stdev range.)
    avg_iv = group["impl_volatility"].mean()
    stdev = avg_iv * np.sqrt(T)

    # 2) Convert strikes to moneyness k = ln(K/F)
    group["k"] = np.log(group["K"] / F)

    # Sort by k
    group = group.sort_values("k")

    # Arrays of available moneyness and implied vol
    k_data = group["k"].values
    iv_data = group["impl_volatility"].values

    # 3) Define a fine moneyness grid ±8 stdevs from 0 (i.e. from -8*stdev to +8*stdev)
    k_min = -8 * stdev
    k_max = 8 * stdev
    k_grid = np.linspace(k_min, k_max, n_points)  # 2000 points as per your spec

    # 4) Interpolate implied vol across k_grid
    #    Extrapolate by taking the edge vol for k < k_data.min() or k > k_data.max().
    iv_grid = np.interp(k_grid, k_data, iv_data,
                        left=iv_data[0],
                        right=iv_data[-1])

    # Convert moneyness back to strikes: K = F * exp(k)
    K_grid = F * np.exp(k_grid)

    # 5) Compute out-of-the-money option prices:
    #    - Call if K < F => k < 0 => actually that means k < 0 => ln(K/F) < 0 => K < F
    #      (But watch sign: ln(K/F) < 0 => K < F => OTM call)
    #    - Put if K >= F => k >= 0 => K >= F => OTM put
    #    (Carr & Wu typically use calls for K>F and puts for K<F or vice versa, but the idea is:
    #     only out-of-the-money options are used. We'll do "Call if K<F, Put if K>=F".)
    option_prices = []
    for k_val, K_val, sigma_val in zip(k_grid, K_grid, iv_grid):
        is_call = (K_val < F)  # True => call, False => put
        price = BSM_call_put(F, K_val, T, sigma_val, r, is_call=is_call)
        option_prices.append(price)
    option_prices = np.array(option_prices)

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


# def BSM_Spot(S, K, T, sigma, r, delta, type):
#     d1 = (np.log(S/K) + (r - delta + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     C = S*np.exp(-delta*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
#     if type == "C":
#         return C
#     if type == "P":
#         return C - S *np.exp(-delta*T) + K *np.exp(-r*T)
#     print("BSM error: Did not choose 'C' or 'P'")
#     return np.NAN
#
#
# def BSM_FW(F, K, T, sigma, r, type):
#     d1 = (np.log(F/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     C = np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
#     if type == "C":
#         return C
#     if type == "P":
#         return C - F + K *np.exp(-r*T)
#     print("BSM error: Did not choose 'C' or 'P'")
#     return np.NAN
#
#
# def BSM_FW_simple(FW_curves, K, T, sigma, ZCY_curves, type, date):
#     # date = datetime.strptime(date, '%Y_%m_%d')
#     F = interpolated_FW(FW_curves, date, T)
#     r = interpolated_ZCY(ZCY_curves, date, T)
#
#     d1 = (np.log(F/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     C = np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
#     if type == "C":
#         return C
#     if type == "P":
#         return C - F + K *np.exp(-r*T)
#     print("BSM error: Did not choose 'C' or 'P'")
#     return np.NAN
#
#
#
# def BSM_FW_single(F, K, T, sigma, r, cp_flag):
#     d1 = (np.log(F/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     C = np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
#     if cp_flag == "C":
#         return C
#     if cp_flag == "P":
#         return C - (F - K) *np.exp(-r*T)
#     print("BSM error: Did not choose 'C' or 'P'")
#     return np.NAN



# BSM = BSM_FW_single # selected as BSM