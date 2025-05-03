import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import sys
import time
from itertools import islice
import load_clean_lib
import re
import volpy_func_lib as vp
import os

# from global_settings import days_type
import importlib
import global_settings
importlib.reload(global_settings)
days_type = global_settings.days_type
clean_t_days = global_settings.clean_t_days

importlib.reload(load_clean_lib)


def load_od_FW_ticker(ticker, valid_dates):
    """Loader optionsdata, forward priser, yield curves and returns."""

    OM_dir        = load_clean_lib.Option_metrics_path_from_profile()
    ticker_dir      = OM_dir / "Tickers" / "Input" / ticker

    if not ticker_dir.is_dir():
        raise FileNotFoundError(f"Input folder {ticker_dir!r} does not exist.")

    # Load hvert dataset direkte
    od = vp.load_option_data(ticker_dir / "option data.csv", valid_dates)


    FW = vp.load_forward_price(ticker_dir / "forward price.csv")
    ret = vp.load_returns_and_price(ticker_dir / "returns and stock price.csv")

    return od, FW, ret


# outdated
# def add_FW_to_od_ticker(od, FW):
#     """Interpolate forward prices onto each od row, grouping only by date."""
#     days_var = "c_days"
#
#     # Build a date → (days_array, forward_array) mapping
#     FW_grouped = {
#         date: (
#             grp[days_var].values,
#             grp["forwardprice"].values
#         )
#         for date, grp in FW.groupby("date")
#     }
#
#     def _interpolate_per_date(group):
#         date = group["date"].iloc[0]
#         if date not in FW_grouped:
#             raise ValueError(f"No forward‐price data for date {date!r}")
#         x, y = FW_grouped[date]
#
#         # ensure all target days lie within the curve’s support
#         if group[days_var].min() < x[0] or group[days_var].max() > x[-1]:
#             raise ValueError(f"Cannot interpolate: {date!r} has days outside [{x[0]}, {x[-1]}]")
#
#         # vectorized interpolation, writing into a new “F” column
#         interpolated = np.interp(group[days_var], x, y)
#         # avoid SettingWithCopy by returning a fresh DataFrame
#         group = group.copy()
#         group["F"] = interpolated
#         return group
#
#     # apply per‐date interpolation and return the augmented od
#     return od.groupby("date", group_keys=False).apply(_interpolate_per_date)


import warnings
import logging
# logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
# logging.debug(f"Input FW shape: {FW.shape if not FW.empty else 'empty'}")


def add_FW_to_od_ticker(od, FW):
    """Interpolate forward prices, using neighboring dates if necessary."""
    days_var = "c_days"

    # Handle empty FW input
    if FW.empty:
        od["F"] = np.nan
        return od

    # Convert all dates to Timestamps upfront
    FW["date"] = pd.to_datetime(FW["date"])
    od["date"] = pd.to_datetime(od["date"])

    # Build date → (days_array, forward_array) mapping
    FW_grouped = {
        date: (
            grp[days_var].values,
            grp["forwardprice"].values
        )
        for date, grp in FW.groupby("date")
    }

    def _find_prev_date(d):
        available_dates = sorted(FW_grouped)
        prev = [dt for dt in available_dates if dt < d]
        return prev[-1] if prev else None

    def _find_next_date(d):
        available_dates = sorted(FW_grouped)
        next_dates = [dt for dt in available_dates if dt > d]
        return next_dates[0] if next_dates else None

    # def _inter_extrapolate_linear(xp, yp, x):
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         # Compute slopes at endpoints
    #         if len(yp) >= 2:
    #             m_start = (yp[1] - yp[0]) / (xp[1] - xp[0]) if xp[1] != xp[0] else 0
    #             m_end = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 0
    #         else:
    #             m_start = m_end = 0
    #         return np.interp(
    #             x, xp, yp,
    #             left =yp[0]  + (x - xp[0])  * m_start,
    #             right=yp[-1] + (x - xp[-1]) * m_end)

    # def _inter_extrapolate_linear(xp, yp, x):
    #     logging.debug(f"Input xp: {xp}, yp: {yp}, x: {x}")
    #
    #     if len(xp) == 0 or len(yp) == 0:
    #         logging.warning("Empty xp or yp array")
    #         return np.full_like(x, np.nan)
    #
    #     sort_indices = np.argsort(xp)
    #     xp = xp[sort_indices]
    #     yp = yp[sort_indices]
    #     logging.debug(f"Sorted xp: {xp}, yp: {yp}")
    #
    #     if len(yp) == 1:
    #         logging.debug("Single-point extrapolation")
    #         return np.full_like(x, yp[0])
    #
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         m_start = (yp[1] - yp[0]) / (xp[1] - xp[0]) if (xp[1] != xp[0]) else 0.0
    #         m_end = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]) if (xp[-1] != xp[-2]) else 0.0
    #
    #     m_start = np.nan_to_num(m_start, nan=0.0, posinf=0.0, neginf=0.0)
    #     m_end = np.nan_to_num(m_end, nan=0.0, posinf=0.0, neginf=0.0)
    #     logging.debug(f"Slopes: m_start={m_start}, m_end={m_end}")
    #
    #     result = np.interp(
    #         x, xp, yp,
    #         left=yp[0] + (x - xp[0]) * m_start,
    #         right=yp[-1] + (x - xp[-1]) * m_end
    #     )
    #     logging.debug(f"Result: {result}")
    #     return result

    def _inter_extrapolate_linear(xp, yp, x):
        x = np.asarray(x)
        xp = np.asarray(xp)
        yp = np.asarray(yp)

        # Handle edge cases
        if len(xp) < 1 or len(yp) < 1:
            return np.full_like(x, np.nan)
        if len(xp) == 1:
            return np.full_like(x, yp[0])

        # Sort xp and yp
        sort_idx = np.argsort(xp)
        xp = xp[sort_idx]
        yp = yp[sort_idx]

        # Calculate slopes
        with np.errstate(divide='ignore', invalid='ignore'):
            m_start = (yp[1] - yp[0]) / (xp[1] - xp[0]) if (xp[1] != xp[0]) else 0.0
            m_end = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]) if (xp[-1] != xp[-2]) else 0.0

        # Split x into regions
        mask_left = x < xp[0]
        mask_mid = (x >= xp[0]) & (x <= xp[-1])
        mask_right = x > xp[-1]

        # Initialize result array
        result = np.empty_like(x)

        # Interpolate middle region
        result[mask_mid] = np.interp(x[mask_mid], xp, yp)

        # Extrapolate left region
        result[mask_left] = yp[0] + (x[mask_left] - xp[0]) * m_start

        # Extrapolate right region
        result[mask_right] = yp[-1] + (x[mask_right] - xp[-1]) * m_end

        return result


    def _interpolate_per_date(group):
        date = group["date"].iloc[0]

        if date in FW_grouped:
            xp, yp = FW_grouped[date]
            target = group[days_var].values
            fitted = _inter_extrapolate_linear(xp, yp, target)
        else:
            prev = _find_prev_date(date)
            next_date = _find_next_date(date)

            target = group[days_var].values
            fitted = np.full_like(target, np.nan)  # Initialize

            if prev and next_date:
                # Get data from neighboring dates
                xp_prev, yp_prev = FW_grouped[prev]
                xp_next, yp_next = FW_grouped[next_date]

                # Compute forwards from both dates
                forward_prev = _inter_extrapolate_linear(xp_prev, yp_prev, target)
                forward_next = _inter_extrapolate_linear(xp_next, yp_next, target)

                # Check if forwards are within 10% of each other
                avg = (forward_prev + forward_next) / 2
                diff = np.abs(forward_prev - forward_next)
                within_10pct = diff <= 0.1 * avg

                # Time-based interpolation weights
                days_prev = (pd.Timestamp(date) - pd.Timestamp(prev)).days
                days_next = (pd.Timestamp(next_date) - pd.Timestamp(date)).days
                total_days = days_prev + days_next
                weight_prev = days_next / total_days if total_days > 0 else 0.5
                weight_next = 1 - weight_prev

                # Blend forwards where within 10%, else use closer date
                blended = forward_prev * weight_prev + forward_next * weight_next
                fitted = np.where(within_10pct, blended, np.nan)
                if not np.all(within_10pct):
                    warnings.warn(f"Some forwards for {date} differ >10% between {prev} and {next_date}", UserWarning)
            else:
                warnings.warn(f"No forward data for {date}, and no prev/next dates to interpolate between", UserWarning)

        group = group.copy()
        group["F"] = fitted
        return group

    return od.groupby("date", group_keys=False).apply(_interpolate_per_date)




def clean_od_ticker(od):
    days_var = days_type() + "days"

    # od = od.dropna(subset=["K", "IV_om", "cp_flag"])
    # od = od[od["c_days"] <= 365]
    od["spread"] = od["ask"] - od["bid"]
    od["mid"] = od["bid"] + od["spread"] / 2
    od = (
        od
        .dropna(subset=["K", "IV_om", "cp_flag"])
        .query(f"{days_var} <= 365 and spread > 0 and bid > 0") # todo: remove t_days above 252
    )
    # od = od[od["volume"] > 5000]

    od = od.sort_values(
        by=["date", "c_days"]
    ).reset_index(drop=True)
    return od


# def load_clean_and_prepare_od_ticker(ticker, valid_dates, ZCY_curves, IV_type = "od", safe_slow_IV = False):
#     # load data
#     od, FW, returns_and_prices = load_od_FW_ticker(ticker, valid_dates)

#     # returns_and_prices = returns_and_prices[(returns_and_prices["open"] > 0) & (returns_and_prices["close"] > 0)]

#     # add forward (before cleaning such that od_raw has the forward rate, used in options trats for ATM options that might be slightly ITM)
#     od = add_FW_to_od_ticker(od, FW)

#     # add IV to options (bid/ask/mid/om)
#     od["IV"] = vp.add_bid_mid_ask_IV(od, IV_type, safe_slow_IV = safe_slow_IV)
#     od[f"IV_{IV_type}"] = od["IV"]

#     # # Intrinsic value
#     # flag = np.where(od["cp_flag"] == 'C', 1, -1)  # now a numeric array
#     # od["intrinsic"] = np.maximum(flag * (od["F"] - od["K"]), 0)
#     # od["mid_intrinsic_diff"] = od["mid"] - od["intrinsic"]

#     od_raw = od.copy()

#     # clean data (should be looked upon)
#     od = clean_od_ticker(od)

#     # remove if ITM
#     od = od.loc[((od["F"] < od["K"]) & (od["cp_flag"] == "C")) | ((od["F"] > od["K"]) & (od["cp_flag"] == "P"))]

#     # add r to options
#     od = vp.add_r_to_od_parallel(od, ZCY_curves)
#     print("Data loaded")


#     return od, returns_and_prices, od_raw



from pathlib import Path
import numpy as np
import pandas as pd

def load_clean_and_prepare_od_ticker(
    ticker, valid_dates, ZCY_curves,
    IV_type="od", safe_slow_IV=False
):
    # 1) Load + forward
    od, FW, returns_and_prices = load_od_FW_ticker(ticker, valid_dates)
    od = add_FW_to_od_ticker(od, FW)
    # todo: evt. fjern alle optioner der mangler forward for en given dato

    # 2) Add IV
    od["IV"] = vp.add_bid_mid_ask_IV(od, IV_type, safe_slow_IV=safe_slow_IV)
    od[f"IV_{IV_type}"] = od["IV"]

    # 3) Gem rå data
    od_raw = od.copy()

    # 4) Clean + fjern ITM
    od = clean_od_ticker(od)
    od = od.loc[
        ((od["F"] < od["K"]) & (od["cp_flag"] == "C")) |
        ((od["F"] > od["K"]) & (od["cp_flag"] == "P"))
    ]

    # 5) Add rente
    od = vp.add_r_to_od_parallel(od, ZCY_curves)

    # Mål1: relativ strike-afvigelse
    od["rel_dev"] = ((od["K"] - od["F"]) / od["F"]).abs()

    grp = ["date", "exdate", "cp_flag"]
    # groupby-agg: kvantil af rel_dev og median(mid)
    thr = (
        od
        .groupby(grp)
        .agg(
            thr_dev=("rel_dev", "median"), # q75_dev   = ("rel_dev", lambda x: x.quantile(0.50)),
            median_mid=("mid", "median")
        )
        .reset_index()
    )
    # tærskel = max(thr_dev, 0.2)
    thr["thr_dev"] = thr["thr_dev"].clip(lower=0.2)

    # merge tilbage
    od = od.merge(thr[[*grp, "thr_dev", "median_mid"]], on=grp, how="left")

    # 6) Sæt flags
    od["flag1"] = od["rel_dev"] >= od["thr_dev"]   # kigger otm optioner 
    od["flag2"] = od["mid"] > od["median_mid"]*2   # er mid på disse større end median ?
    od["remove"] = od["flag1"] & od["flag2"] 

    # # 7) Gem debug
    # debug_cols = [
    #     "optionid", "date", "exdate", "cp_flag",
    #     "F", "K", "mid", "rel_dev",
    #     "thr_dev", "median_mid",
    #     "flag1", "flag2", "remove"
    # ]
    # debug_dir = Path(load_clean_lib.dirs()["OptionMetrics"] / "Tickers" / "Debug")
    # debug_dir.mkdir(parents=True, exist_ok=True)
    # od[debug_cols].to_csv(debug_dir / f"debug_rel_dev_filter_{ticker}.csv", index=False)

    # 8) Filtrér væk hvor remove==True
    od = od.loc[~od["remove"]].copy()

    # 9) Ryd op og returnér
    od.drop(columns=["rel_dev", "median_mid", "thr_dev", "flag1", "flag2", "remove"], inplace=True)
    
    return od, returns_and_prices, od_raw





def high_low_swap_rates_ticker(summary_dly_df, od_rdy, n_points=200):
    # 1) Calculate var_swap_rate on od_rdy
    # df_swaps = process_od_rdy(od_rdy, replicate_SW, n_points=n_points)
    # Use the parallel version to calculate var_swap_rate
    df_swaps = vp.process_od_rdy_parallel(od_rdy, vp.replicate_SW, n_points=n_points)

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




def create_summary_dly_df_ticker(od, returns_and_prices, RF, n_grid=200):
    first_day = od["date"].min()
    last_day = od["date"].max()

    # Add low/high and create summary (Filters dataset for criteria such as min 3 strikes ... min 8 days...)
    od, summary_dly_df = vp.od_filter_and_summary_creater(od)

    summary_dly_df.reset_index(inplace=True)

    # Add risk-free rate of the given date
    summary_dly_df = summary_dly_df.merge(RF[['date', 'RF']], on='date', how='left')

    # Realized vol calculation
    real_vol = vp.calc_realized_var(returns_and_prices, first_day, last_day)

    # Merge realized vol with summary_dly_df
    summary_dly_df = vp.add_realized_vol_to_summary(summary_dly_df, real_vol)

    # only keep the lowest ("low") and the second lowest ("high") TTMs
    od_rdy = od[(od["low"] == True) | (od["high"] == True)]

    # Calculate high/low Swap Rates
    summary_dly_df = high_low_swap_rates_ticker(summary_dly_df, od_rdy, n_points=n_grid)


    return summary_dly_df, od



def load_analyze_create_swap_ticker(ticker_list, IV_type="om", safe_slow_IV=False):

    print(f"{days_type()} was selected in global_settings.py")
    RF = load_clean_lib.download_factor_df(Factor_list=["FF5"])[['date', 'RF']]

    OM_dir        = load_clean_lib.Option_metrics_path_from_profile()
    Tickers_dir   = OM_dir / "Tickers"
    output_dir    = Tickers_dir / "Output" / days_type()
    sum1_dir      = Tickers_dir / "Sum and orpy" / days_type() / "sum1"
    os.makedirs(sum1_dir, exist_ok=True)

    valid_dates_path = Tickers_dir / "dates.csv"
    valid_dates = pd.read_csv(valid_dates_path, usecols=["DATE"], parse_dates=["DATE"])
    valid_dates.rename(columns={"DATE": "date"}, inplace=True)

    ZCY_curves = vp.load_ZC_yield_curve(Tickers_dir / "ZC yield curve.csv")


    for ticker in ticker_list:

        output_ticker_dir = output_dir / ticker
        if output_ticker_dir.is_dir():
            print(f"{ticker} output data already exists; skipping.")
            continue

        print(f"Analysing {ticker}")
        os.makedirs(output_ticker_dir, exist_ok=True)

        # Load data and clean
        od, returns_and_prices, od_raw = load_clean_and_prepare_od_ticker(ticker, valid_dates, ZCY_curves, IV_type=IV_type, safe_slow_IV=safe_slow_IV)

        # Calculate results such as SW, RV ect.
        summary_dly_df, od_rdy = create_summary_dly_df_ticker(od, returns_and_prices, RF, n_grid=2000)

        summary_dly_df = vp.interpolate_swaps_and_returns(summary_dly_df)
        summary_dly_df = summary_dly_df.reset_index()

        # save
        summary_dly_df.to_csv(f"{sum1_dir}/{ticker}.csv", index=False)

        summary_dly_df.to_csv(f"{output_ticker_dir}/sum1_df.csv", index=False)
        od_raw.to_csv(f"{output_ticker_dir}/od_raw.csv", index=False)
        od.to_csv(f"{output_ticker_dir}/od_rdy.csv", index=False)
    return







import os
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import load_clean_lib
import volpy_func_lib as vp

# bring in your helper functions
from volpy_func_ticker_lib import (
    load_clean_and_prepare_od_ticker,
    create_summary_dly_df_ticker,
    days_type,
)


def _process_one_ticker(
    ticker: str,
    valid_dates: pd.DataFrame,
    ZCY_curves: pd.DataFrame,
    RF: pd.DataFrame,
    base_dir: Path,
    IV_type: str,
    safe_slow_IV: bool,
) -> list[str]:
    """
    Load, analyze, and save results for a single ticker.
    Cleans up large objects at the end to keep memory usage in check.
    """
    import traceback

    logs: list[str] = []
    def L(*msgs):
        # coerce everything to str and join with spaces
        logs.append(" ".join(str(m) for m in msgs))


    tickers_dir   = base_dir / "Tickers"
    out_dir       = tickers_dir / "Output" / days_type() / ticker
    sum1_dir      = tickers_dir / "SumAndOrpy" / days_type() / "sum1"

    # skip if already done
    if out_dir.exists():
        return logs

    try:
        # 1) Load & prepare
        od, returns_and_prices, od_raw = load_clean_and_prepare_od_ticker(
            ticker, valid_dates, ZCY_curves,
            IV_type=IV_type, safe_slow_IV=safe_slow_IV,
        )

        # L("Write log here", od.columns)

        # 2) Summarize into daily frame + ready-to-use od
        summary_dly_df, od_rdy = create_summary_dly_df_ticker(
            od, returns_and_prices, RF, n_grid=2000
        )
        # 3) Interpolate missing swaps/returns, reset index for CSV
        summary_dly_df = vp.interpolate_swaps_and_returns(summary_dly_df).reset_index()

        out_dir.mkdir(parents=True, exist_ok=True)
        sum1_dir.mkdir(parents=True, exist_ok=True)

        summary_dly_df.to_csv(sum1_dir / f"{ticker}.csv", index=False)
        summary_dly_df.to_csv(out_dir / "sum1_df.csv",    index=False)
        od_raw.to_csv(       out_dir / "od_raw.csv",      index=False)
        od_rdy.to_csv(           out_dir / "od_rdy.csv",  index=False)
    finally:
        # Explicitly delete large objects and run garbage collector
        for obj in ("od", "returns_and_prices", "od_raw", "summary_dly_df", "od_rdy"):
            if obj in locals():
                del locals()[obj]
        gc.collect()
        return logs

# import traceback

def load_analyze_create_swap_ticker_parallel(
    ticker_list: list[str],
    IV_type: str = "om",
    safe_slow_IV: bool = False,
    max_max_workers = 4
) -> None:
    """
    Parallelizes load/analysis of multiple tickers.
    Writes per-ticker CSVs under:
      Tickers/Output/<days_type()>/<ticker>/
      Tickers/SumAndOrpy/<days_type()>/sum1/
    """
    base_dir     = load_clean_lib.Option_metrics_path_from_profile()
    tickers_dir  = base_dir / "Tickers"

    # prepare common inputs
    dates_path   = tickers_dir / "dates.csv"
    valid_dates  = (
        pd.read_csv(dates_path, usecols=["DATE"], parse_dates=["DATE"])
          .rename(columns={"DATE": "date"})
    )

    ZCY_curves = vp.load_ZC_yield_curve(tickers_dir / "ZC yield curve.csv")
    RF         = load_clean_lib.download_factor_df(Factor_list=["FF5"])[["date", "RF"]]

    # choose a conservative number of workers so we don't blow out RAM
    n_cpus      = os.cpu_count() or 1
    max_workers = min(n_cpus - 1, max_max_workers)
    print(f"Launching up to {max_workers} parallel workers for {len(ticker_list)} tickers…")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_ticker,
                ticker,
                valid_dates,
                ZCY_curves,
                RF,
                base_dir,
                IV_type,
                safe_slow_IV,
            ): ticker
            for ticker in ticker_list
        }

        for fut in as_completed(futures):
            tkr = futures[fut]
            try:
                logs = fut.result()
                for line in logs:
                    print(line)  # now you _will_ see them

                print(f"[✓] {tkr}")
            except Exception as e:
                # traceback.print_exc()  # Add this import at the top
                print(f"[✗] {tkr} failed: {e}")

    print("All tickers processed.")

    # # Replace the ProcessPoolExecutor block with:
    # for ticker in ticker_list:
    #     try:
    #         logs = _process_one_ticker(ticker, valid_dates, ZCY_curves, RF, base_dir, IV_type, safe_slow_IV)
    #         print(f"[✓] {ticker}")
    #     except Exception:
    #         traceback.print_exc()




import vol_strat_lib as vs
import option_returns as orpy


def import_sum_raw_ticker(ticker_dir):
    """
    Read in sum1_df.csv and od_raw.csv from ticker_dir.
    If the directory or files aren’t present, prints a warning and returns (None, None).
    """
    # 1) Check that the directory exists
    if not ticker_dir.is_dir():
        print(f"Directory {ticker_dir!r} not found; skipping ticker.")
        return None, None

    sum_path = ticker_dir / "sum1_df.csv"
    raw_path = ticker_dir / "od_raw.csv"

    # 2) Check that the files exist
    if not sum_path.exists():
        print(f"File {sum_path!r} not found; skipping ticker.")
        return None, None
    if not raw_path.exists():
        print(f"File {raw_path!r} not found; skipping ticker.")
        return None, None

    # 3) Load and clean
    sum_df = pd.read_csv(sum_path)
    sum_df["date"] = pd.to_datetime(sum_df["date"])
    sum_df["return_next"] = sum_df["return"].shift(-1)
    sum_df.loc[sum_df["SW_0_30"] <= 1e-6, "SW_0_30"] = 1e-6

    od_raw = pd.read_csv(raw_path)
    od_raw["optionid"] = od_raw["optionid"].astype(int)
    od_raw["date"] = pd.to_datetime(od_raw["date"])

    return sum_df, od_raw

def create_option_sgys_ticker(ticker_list, price_type="mid", IV_type="om", OTMs=[0.05, 0.15]):

    base_dir    = load_clean_lib.Option_metrics_path_from_profile()
    sum2_dir     = base_dir / "Tickers" / "SumAndOrpy" / days_type() / "sum2"
    orpy_dir    = base_dir / "Tickers" / "SumAndOrpy" / days_type() / "orpy"
    sum2_dir.mkdir(parents=True, exist_ok=True)
    orpy_dir.mkdir(parents=True, exist_ok=True)

    for ticker in ticker_list:

        ticker_dir = base_dir / "Tickers" / "Output" / days_type() / ticker
        sum_df, od_raw = import_sum_raw_ticker(ticker_dir)

        # load data
        od_hl = vs.create_od_hl(od_raw=od_raw, sum_df=sum_df, price_type=price_type, IV_type=IV_type)
        sum_df = vs.add_F_to_sum_df(od_hl=od_hl, sum_df=sum_df)
        sum_df = vs.add_ATM_options_to_sum_df(sum_df=sum_df, od_hl=od_hl, od_raw=od_raw, OTMs=OTMs)

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
        sum_df.to_csv(ticker_dir / "sum2_df.csv", index=False)
        df_orpy.to_csv(ticker_dir / "df_orpy.csv", index=False)

        sum_df.to_csv(sum2_dir / f"{ticker}.csv", index=False)
        df_orpy.to_csv(orpy_dir / f"{ticker}.csv", index=False)



import os
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from volpy_func_ticker_lib import days_type


def fix_index_returns_bloomberg_OM(ticker_list=["SPX","NDX","OEX","DJX"]):
    base_dir   = load_clean_lib.Option_metrics_path_from_profile()
    input_dir  = base_dir / "Tickers" / "Input"
    index_data_dir = base_dir / "Tickers" / "index data"
    OM_returns_dir = base_dir / "Tickers" / "Returns OM"
    input_ticker_dir = base_dir / "Tickers" / "Input"

    for ticker in ticker_list:
        OM_path      = OM_returns_dir / ticker / 'returns and stock price OM.csv'
        Bloomberg_path = index_data_dir / f'{ticker}.csv'
        if not OM_path.exists():
            print(f"File {OM_path!r} not found; skipping ticker.")
            continue

        # load
        df_OM        = pd.read_csv(OM_path)
        df_Bloomberg = pd.read_csv(Bloomberg_path)

        # parse dates
        df_OM['date']      = pd.to_datetime(df_OM['date'],      dayfirst=False)
        df_Bloomberg['date'] = pd.to_datetime(df_Bloomberg['date'], dayfirst=False)
        df_Bloomberg = df_Bloomberg[df_Bloomberg["return"].isna()==False]

        # # keep old return around
        # df_CRSP['return_old'] = df_CRSP['return']

        # OPTION A: overwrite in-place via index alignment
        df_OM.set_index('date', inplace=True)
        df_OM['return'] = df_Bloomberg.set_index('date')['return']
        df_OM.reset_index(inplace=True)

        # optional: sort by date
        df_OM.sort_values('date', inplace=True)

        # save
        save_dir = input_ticker_dir / ticker / 'returns and stock price.csv'
        df_OM.to_csv(save_dir, index=False)
        print(f"{ticker!r} (returns and stock price.csv) fixed.")



def _process_one_ticker_sgys(
    ticker: str,
    base_dir: Path,
    price_type: str,
    IV_type: str,
    OTMs: list[float],
):
    """
    Build sum2_df and df_orpy for a single ticker and save to disk.
    Skips if output CSV already exists. Raises an error if input files missing.
    """
    tickers_dir = base_dir / "Tickers"
    ticker_dir  = tickers_dir / "Output" / days_type() / ticker
    sum2_path   = ticker_dir / "sum2_df.csv"
    orpy_path   = ticker_dir / "df_orpy.csv"

    # 0) Skip if already done
    if sum2_path.exists() and orpy_path.exists():
        print(f"{ticker}: output CSVs already exist; skipping.")
        return

    # 1) Check input files
    sum1_path = ticker_dir / "sum1_df.csv"
    raw_path  = ticker_dir / "od_raw.csv"
    if not sum1_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing required files for {ticker}: "
                                f"{sum1_path.name}, {raw_path.name}")

    # 2) Load and clean raw data
    sum_df = pd.read_csv(sum1_path)
    sum_df["date"]        = pd.to_datetime(sum_df["date"])
    sum_df["return_next"] = sum_df["return"].shift(-1)
    sum_df.loc[sum_df["SW_0_30"] <= 1e-6, "SW_0_30"] = 1e-6

    od_raw = pd.read_csv(raw_path)
    od_raw["optionid"] = od_raw["optionid"].astype(int)
    od_raw["date"]     = pd.to_datetime(od_raw["date"])

    # 3) Create high/low option frame and extend sum_df
    od_hl  = vs.create_od_hl(
        od_raw=od_raw, sum_df=sum_df,
        price_type=price_type, IV_type=IV_type
    )
    sum_df = vs.add_F_to_sum_df(od_hl=od_hl, sum_df=sum_df)
    sum_df = vs.add_ATM_options_to_sum_df(
        sum_df=sum_df,
        od_hl=od_hl,
        od_raw=od_raw,
        OTMs=OTMs
    )

    # 4) Build trading strategies
    HL30_list = ["30"]
    df_orpy   = orpy.prepare_for_sgys(sum_df.copy(), OTMs)
    Strategies = [
        orpy.add_put_and_call_sgy,
        orpy.add_straddle_strangle_sgy,
        orpy.add_butterfly_spread_sgy,
        orpy.add_condor_strangle_sgy,
        orpy.add_stacked_straddle_sgy,
        orpy.add_full_stacked_straddle_sgy,
        orpy.add_full_stacked_strangle_sgy,
    ]
    for add_sgy in Strategies:
        df_orpy = add_sgy(df_orpy, OTMs, HL30_list)

    # 5) Ensure output dirs exist
    sum2_dir = tickers_dir / "SumAndOrpy" / days_type() / "sum2"
    orpy_dir = tickers_dir / "SumAndOrpy" / days_type() / "orpy"
    sum2_dir.mkdir(parents=True, exist_ok=True)
    orpy_dir.mkdir(parents=True, exist_ok=True)

    # 6) Save results
    sum_df.to_csv(sum1_path.parent / "sum2_df.csv", index=False)
    df_orpy.to_csv(sum1_path.parent / "df_orpy.csv", index=False)
    sum_df.to_csv(sum2_dir / f"{ticker}.csv", index=False)
    df_orpy.to_csv(orpy_dir / f"{ticker}.csv", index=False)

    # 7) Cleanup large objects
    for obj in ("sum_df", "od_raw", "od_hl", "df_orpy"):
        if obj in locals():
            del locals()[obj]
    gc.collect()


def create_option_sgys_ticker_parallel(
    ticker_list: list[str],
    price_type: str = "mid",
    IV_type:    str = "om",
    OTMs:       list[float] = None,
    max_max_workers = 4,
    ) -> None:
    """
    Parallelize creation of sum2_df and df_orpy CSVs for each ticker.
    Skips tickers with existing CSVs and marks missing-input cases as failures.
    """
    if OTMs is None:
        OTMs = [0.05, 0.15]

    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    n_cpus   = os.cpu_count() or 1
    max_workers = max(1, min(n_cpus - 1, max_max_workers))
    print(f"Launching up to {max_workers} workers for {len(ticker_list)} tickers…")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_ticker_sgys,
                ticker, base_dir,
                price_type, IV_type, OTMs
            ): ticker
            for ticker in ticker_list
        }

        for fut in as_completed(futures):
            tkr = futures[fut]
            try:
                fut.result()
                print(f"[✓] {tkr}")
            except FileNotFoundError as e:
                print(f"[✗] {tkr} failed: {e}")
            except Exception as e:
                print(f"[✗] {tkr} error: {e}")

    print("All option-strategy CSVs created.")





def concat_output_ticker_datasets(
    ticker_list: list[str],
    df_name:    str,
) -> pd.DataFrame:
    """
    For each ticker in ticker_list, load
      <Option_metrics_path>/Tickers/Output/<days_type()>/<ticker>/<df_name>.csv
    (with low_memory=False to avoid mixed-dtype warnings)
    and append them into one DataFrame.

    Skips any missing files with a warning.
    Adds a 'ticker' column to identify origin.
    """
    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    out_dir  = base_dir / "Tickers" / "Output" / days_type()
    frames   = []

    for ticker in ticker_list:
        path = out_dir / ticker / f"{df_name}.csv"
        if not path.exists():
            print(f"Warning: {path!r} not found; skipping {ticker}.")
            continue

        # read with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(
            path,
            low_memory=False,
            parse_dates=["date"],   # explicitly parse your date column
        )
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def concat_ticker_datasets(
    ticker_list: list[str],
    df_name:    str,
) -> pd.DataFrame:
    """
    For each ticker in ticker_list, load
      <Option_metrics_path>/Tickers/Output/<days_type()>/<ticker>/<df_name>.csv
    (with low_memory=False to avoid mixed-dtype warnings)
    and append them into one DataFrame.

    Skips any missing files with a warning.
    Adds a 'ticker' column to identify origin.
    """
    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    dir  = base_dir / "Tickers" / "SumAndOrpy" / days_type() / df_name
    frames   = []
    skipped = []

    for ticker in ticker_list:
        path = dir / f"{ticker}.csv"
        if not path.exists():
            skipped.append(ticker)
            # print(f"Warning: {path!r} not found; skipping {ticker}.")
            continue

        # read with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(
            path,
            low_memory=False,
            parse_dates=["date"],   # explicitly parse your date column
        )
        df["ticker"] = ticker
        frames.append(df)

    if skipped:
        print(f"Skipped {len(skipped)}/{len(ticker_list)} tickers:", skipped)  # ← print them all at once

    if not frames:
        return pd.DataFrame()

    df_merged = pd.concat(frames, ignore_index=True)
    # df_merged = df_merged.dropna(subset=["SW_0_30"])
    return df_merged


def create_csv_from_folder(df_name_list = ["sum1", "sum2", "orpy"]):
    for df_name in df_name_list:
        df = concat_ticker_datasets(vp.ALL_tickers, df_name)

        base_dir = load_clean_lib.Option_metrics_path_from_profile()
        dir = base_dir / "Tickers" / "SumAndOrpy" / days_type()

        df.to_csv(dir / f"{df_name}.csv", index=False)


def draw_ticker_sum_orpy(ticker_list, df_name):
    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    data_dir = base_dir / "Tickers" / "SumAndOrpy" / days_type()
    file_path = data_dir / f"{df_name}.csv"

    # 1) Ensure file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {file_path!r}")

    # 2) Read CSV (suppress low-memory warnings)
    df = pd.read_csv(
        file_path,
        low_memory=False,
        parse_dates=["date"] if "date" in pd.read_csv(file_path, nrows=0).columns else None
    )

    # 3) Check for ticker column
    if "ticker" not in df.columns:
        raise KeyError("Loaded DataFrame has no 'ticker' column to filter on.")

    # 4) Filter by ticker_list
    filtered = df[df["ticker"].isin(ticker_list)].copy()
    return filtered


def find_missing_tickers(ticker_list: list[str]) -> list[str]:
    base_dir = load_clean_lib.Option_metrics_path_from_profile()
    sum1_dir = base_dir / "Tickers" / "SumAndOrpy" / days_type() / "sum1"
    missing = []
    for ticker in ticker_list:
        path = sum1_dir / f"{ticker}.csv"
        if not path.exists():
            missing.append(ticker)
    return missing
