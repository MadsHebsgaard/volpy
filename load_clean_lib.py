import numpy as np
from pathlib import Path
import volpy_func_lib as vp
import pandas as pd

def dirs(profile):
    if profile == "Mads":
        Option_metrics_path = Path(r"D:\Finance Data\OptionMetrics")
    elif profile == "Axel":
        Option_metrics_path = Path(r"C:\Users\axell\Desktop\CBS\data\OptionMetrics")
    dir = {
        "OptionMetrics": Option_metrics_path,
        "CarrWu": Option_metrics_path / "1996-2003 (CarrWu2009)",
        # "sp500": Option_metrics_path / "1996-2003 (CarrWu2009)" / "sp500",
        "i4s4": Option_metrics_path / "1996-2003 (CarrWu2009)" / "i4s4",
        "i91": Option_metrics_path / "1996-2003 (CarrWu2009)" / "i91",
        "US ZOO": Option_metrics_path / "1996-2003 (CarrWu2009)" / "US ZOO",
        "i2s1 full": Option_metrics_path / "1996-2003 (CarrWu2009)" / "i2s1 full",
        "axeltest":Option_metrics_path / "Axeltest",
    }
    return dir


def load_od_FW_ZCY(profile, data_folder = "i4s4", tickers = None):
    dir = dirs(profile)


    # Define the i4s4 data files and their loaders (excluding ZCY_curve)
    files_and_loaders = {
        "od": (dir[data_folder] / "option data.csv", vp.load_option_data),  # Option data on i4s4 (4 index, 4 stocks)
        # "RV": (dir[data_folder] / "realized vol.csv", vp.load_realized_volatility),
        # "IV": (dir[data_folder] / "implied vol.csv", vp.load_implied_volatility),
        "FW": (dir[data_folder] / "forward price.csv", vp.load_forward_price),  # Forward data on i4s4
        "ret": (dir[data_folder] / "returns and stock price.csv", vp.load_returns_and_price)
    }

    # Load all i4s4 data into a dictionary
    data = {name: loader(path) for name, (path, loader) in files_and_loaders.items()}

    if tickers is not None:
        data = {key: df[df["ticker"].isin(tickers)] for key, df in data.items()}

    ZCY_curves = vp.load_ZC_yield_curve(dir["CarrWu"] / "ZC yield curve Full.csv")
    ZCY_curves = ZCY_curves[ZCY_curves["date"] <= data["od"]["date"].max()]

    return data["od"], data["FW"], ZCY_curves, data["ret"]




def clean_od(od, first_day = pd.to_datetime("1996-01-04"), last_day = pd.to_datetime("2003-02-28")):
    # od = od.dropna(subset=["K", "IV_om", "cp_flag"])
    # od = od[od["days"] <= 365]
    od["spread"] = od["ask"] - od["bid"]
    od["mid"] = od["bid"] + od["spread"] / 2
    od = (
        od
        .dropna(subset=["K", "IV_om", "cp_flag"])
        .query("days <= 365 and spread > 0 and bid > 0")
    )
    # od = od[od["volume"] > 5000]

    # od = od[(od["date"] >= first_day) & (od["date"] <= last_day)].sort_values(
    #     by="date").reset_index(drop=True)
    od = od[(od["date"] >= first_day) & (od["date"] <= last_day)].sort_values(
        by=["date", "ticker", "days"]
    ).reset_index(drop=True)

    return od

def summary_dly_df_creator(od):
    unique_dates = od["date"].unique()
    unique_tickers = od["ticker"].unique()

    # Initialize the data structure
    index = pd.MultiIndex.from_product([unique_tickers, unique_dates], names=["ticker", "date"])
    summary_dly_df = pd.DataFrame(index=index,
                                  columns=["#days", "low days", "high days", "low #K", "high #K", "#K", "low SW", "high SW", "Active",
                                           "Inactive reason"])
    # Default
    summary_dly_df["low SW"] = np.nan
    summary_dly_df["high SW"] = np.nan
    summary_dly_df["#days"] = np.nan
    summary_dly_df["low days"] = np.nan
    summary_dly_df["high days"] = np.nan
    summary_dly_df["low #K"] = np.nan
    summary_dly_df["high #K"] = np.nan
    summary_dly_df["#K"] = np.nan

    summary_dly_df["Active"] = False
    summary_dly_df["Inactive reason"] = "missing"

    return summary_dly_df


def load_clean_and_prepare_od(data_folder, profile="Mads", tickers=None, first_day=None, last_day=None, IV_type = "od"):
    # load data
    od, FW, ZCY_curves, returns_and_prices = load_od_FW_ZCY(profile, data_folder, tickers=tickers)

    if first_day is None:   first_day = od["date"].min()
    if last_day is None:    last_day = od["date"].max()

    # clean data (should be looked upon)
    od = clean_od(od, first_day=first_day, last_day=last_day)

    # add forward
    od = vp.add_FW_to_od(od, FW)

    # remove if ITM
    od = od.loc[((od["F"] < od["K"]) & (od["cp_flag"] == "C")) | ((od["F"] > od["K"]) & (od["cp_flag"] == "P"))]

    # add r to options
    od = vp.add_r_to_od_parallel(od, ZCY_curves)

    # add IV_bid/IV_mid/IV_ask to options
    od = vp.add_bid_mid_ask_IV(od)
    od["IV"] = od[f"IV_{IV_type}"]

    return od, returns_and_prices


def create_summary_dly_df(od, returns_and_prices, first_day=None, last_day=None):
    if first_day is None:   first_day = od["date"].min()
    if last_day is None:    last_day = od["date"].max()

    # Add low/high and create summary (Filters dataset for criteria such as min 3 strikes ... min 8 days...)
    od, summary_dly_df = vp.od_filter_and_summary_creater(od)

    # Realized vol calculation
    real_vol = vp.calc_realized_var(returns_and_prices, first_day, last_day)

    # Merge realized vol with summary_dly_df
    summary_dly_df = vp.add_realized_vol_to_summary(summary_dly_df, real_vol)

    # only keep the lowest ("low") and the second lowest ("high") TTMs
    od_rdy = od[(od["low"] == True) | (od["high"] == True)]

    summary_dly_df = vp.fill_swap_rates(summary_dly_df, od_rdy, n_points=200)

    return summary_dly_df, od_rdy