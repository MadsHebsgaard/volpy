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
    }
    return dir


def load_od_FW_ZCY(profile, dataset):

    dir = dirs(profile)

    # Define the i4s4 data files and their loaders (excluding ZCY_curve)
    i4s4_files_and_loaders = {
        "od": (dir["i4s4"] / "option data.csv", vp.load_option_data),  # Option data on i4s4 (4 index, 4 stocks)
        # "RV": (dir["i4s4"] / "realized vol.csv", vp.load_realized_volatility),
        # "IV": (dir["i4s4"] / "implied vol.csv", vp.load_implied_volatility),
        "FW": (dir["i4s4"] / "forward price.csv", vp.load_forward_price),  # Forward data on i4s4
        "ret": (dir["i4s4"] / "returns and stock price.csv", vp.load_returns_and_price)
    }

    # Load all i4s4 data into a dictionary
    i4s4 = {name: loader(path) for name, (path, loader) in i4s4_files_and_loaders.items()}
    ZCY_curves = vp.load_ZC_yield_curve(dir["CarrWu"] / "ZC yield curve.csv")

    # Filter SPX data for each i4s4 dataset and create a new sp500 dictionary
    sp500 = {key: df[df["ticker"] == "SPX"] for key, df in i4s4.items()}  # 1 index (sp500): identical to loading just sp500!
    i1s1 = {key: df[(df["ticker"] == "SPX") | (df["ticker"] == "AMZN")] for key, df in i4s4.items()}  # 1 index, 1 stock
    i2s1 = {key: df[(df["ticker"] == "SPX") | (df["ticker"] == "OEX") | (df["ticker"] == "AMZN")] for key, df in i4s4.items()}  # 2 index, 1 stock

    if dataset == "sp500" or dataset == "i1s0":
        data = sp500
    elif dataset == "i4s4":
        data = i4s4
    elif dataset == "i1s1":
        data = i1s1
    elif dataset == "i2s1":
        data = i2s1
    else:
        print("Error: choose a viable dataset such as 'i4s4' or 'sp500'. Default is nan.")
        data = np.nan

    return data["od"], data["FW"], ZCY_curves, data["ret"]




def clean_od(od, first_day = pd.to_datetime("1996-01-04"), last_day = pd.to_datetime("2003-02-28")):
    # od = od.dropna(subset=["K", "impl_volatility", "cp_flag"])
    # od = od[od["days"] <= 365]
    od["spread"] = od["best_offer"] - od["best_bid"]
    od = (
        od
        .dropna(subset=["K", "impl_volatility", "cp_flag"])
        .query("days <= 365 and spread > 0 and best_bid > 0")
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
    index = pd.MultiIndex.from_product([unique_tickers, unique_dates], names=["Ticker", "Date"])
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