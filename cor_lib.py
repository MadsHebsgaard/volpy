import pandas as pd
import numpy as np
import os
import volpy_func_lib as vp
import load_clean_lib as load


def get_index_constituents_and_returns(index: str, force_overwrite = False):
    """
    Henter sammensætning og afkast for index (OEX eller INDU).
    Returnerer:
      - constituents_long: DataFrame med kolonner ['date','ticker']
      - daily_returns_constituents: DataFrame med kolonner ['date','ticker','weight_lag','return']
    hvor weight_lag er gårsdagens vægt.
    """
    # 1) Validér input og sæt filstier
    index = index.upper()
    if index not in ["OEX", "INDU"]:
        raise ValueError("index skal være enten 'OEX' eller 'INDU'")
    
    base_dir = load.dirs()["OptionMetrics"]
    base_path          =  base_dir / "Tickers"
    constituents_path  = os.path.join(base_path, "index data", f"{index.lower()}_constituents_long.csv")
    crsp_path          = os.path.join(base_path, "cor metadata", "crsp_complete_1996_2024.csv")
    output_folder      = os.path.join(base_path, "cor metadata")
    os.makedirs(output_folder, exist_ok=True)
    output_path        = os.path.join(output_folder, f"daily_returns_{index.lower()}.csv")

    # 2) Hvis filen allerede findes, loader vi den
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Fil fundet: {output_path} – loader den i stedet.")
        daily_returns_constituents = pd.read_csv(output_path, parse_dates=["date"])
        constituents_long          = pd.read_csv(constituents_path, parse_dates=["date"])
        return constituents_long, daily_returns_constituents

    # 3) Indlæs constituents og filtrér CRSP
    constituents = pd.read_csv(constituents_path, parse_dates=["date"])
    relevant     = set(constituents["ticker"])
    chunks       = []
    for chunk in pd.read_csv(crsp_path, chunksize=500_000, low_memory=False):
        chunks.append(chunk[chunk["TICKER"].isin(relevant)])
    crsp = pd.concat(chunks, ignore_index=True)

    # 4) Match hver trading day med seneste constituents-dato
    crsp["date"]   = pd.to_datetime(crsp["date"])
    crsp["NEXTDT"] = pd.to_datetime(crsp["NEXTDT"])
    trading_days   = crsp[["date"]].drop_duplicates().sort_values("date")
    cons_dates     = constituents["date"].sort_values().unique()

    lookup = [
        (td, cons_dates[cons_dates <= td][-1])
        for td in trading_days["date"]
        if any(cons_dates <= td)
    ]
    lookup_df = pd.DataFrame(lookup, columns=["date", "constituent_date"])

    expanded = (
        lookup_df
        .merge(constituents, left_on="constituent_date", right_on="date", suffixes=("_td","_const"))
        .rename(columns={"date_td":"date"})
        [["date","ticker"]]
        .sort_values(["date","ticker"])
        .reset_index(drop=True)
    )
    constituents_long = expanded

    # 5) Rens RET og DLRET, og slå dem sammen
    crsp["RET"]   = pd.to_numeric(crsp["RET"], errors="coerce")
    crsp["DLRET"] = pd.to_numeric(crsp["DLRET"], errors="coerce")
    crsp.loc[crsp["RET"].isin([-66,-77,-88,-99]),   "RET"]   = np.nan
    crsp.loc[crsp["DLRET"].isin([-55,-66,-88,-99]), "DLRET"] = np.nan

    dlret_df = (
        crsp[["NEXTDT","TICKER","DLRET"]]
        .dropna()
        .rename(columns={"NEXTDT":"date","TICKER":"ticker"})
    )

    merged = (
        expanded
        .merge(crsp[["date","TICKER","PRC","SHROUT","RET"]],
               left_on=["date","ticker"],
               right_on=["date","TICKER"],
               how="left")
        .merge(dlret_df, on=["date","ticker"], how="left")
    )
    merged["market_cap"]   = merged["PRC"].abs() * merged["SHROUT"]
    merged["RET_combined"] = (1 + merged["RET"]) * (1 + merged["DLRET"].fillna(0)) - 1
    merged = merged.dropna(subset=["RET_combined","market_cap"]) \
                   .drop_duplicates(["date","ticker"])

    # 6) Beregn vægte og shift én dag
    if index == "OEX":
        merged["total_mc"] = merged.groupby("date")["market_cap"].transform("sum")
        merged["weight"]   = merged["market_cap"] / merged["total_mc"]
    else:
        merged["total_pr"] = merged.groupby("date")["PRC"].transform("sum")
        merged["weight"]   = merged["PRC"] / merged["total_pr"]

    merged["weight_lag"] = merged.groupby("ticker")["weight"].shift(1)

    # 7) Byg final_df med weight_lag og return
    final_df = (
        merged[["date","ticker","weight_lag","RET_combined"]]
        .rename(columns={"RET_combined":"return"})
    )
    daily_returns_constituents = final_df.dropna(subset=["weight_lag","return"])

    # 8) Gem til CSV og returnér
    daily_returns_constituents.to_csv(output_path, index=False)
    return constituents_long, daily_returns_constituents





def get_replicated_index_returns(daily_returns_constituents):
    temp = daily_returns_constituents.copy()
    temp["weighted_return"] = temp["weight_lag"] * temp["return"]
    index_returns_replicated = (
        temp.groupby("date")["weighted_return"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_return": "return_replicated"})
    )
    return index_returns_replicated

def add_return_true(df, ticker):
    import pandas as pd
    import os

    # Lav filsti baseret på ticker
    file_path = fr"C:\Users\axell\Desktop\CBS\data\OptionMetrics\Tickers\index data\{ticker.lower()}.csv"

    # Indlæs og forbered indexafkast
    index_returns_true = pd.read_csv(file_path)
    index_returns_true = index_returns_true.rename(columns={"return": "return_true"})
    index_returns_true["date"] = pd.to_datetime(index_returns_true["date"])

    # Merge på dato
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(index_returns_true, on="date", how="left")

    return df



import numpy as np
import matplotlib.pyplot as plt

def plot_log_cumulative_returns(df, start_date, end_date):
    # Filter selected date range
    subset = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    # Compute log cumulative returns
    subset["log_cum_replicated"] = np.log1p(subset["return_replicated"]).cumsum()
    subset["log_cum_true"] = np.log1p(subset["return_true"]).cumsum()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(subset["date"], subset["log_cum_replicated"], label="Replicated Index", color="red", linestyle="-")
    plt.plot(subset["date"], subset["log_cum_true"], label="True Index", color="black", linestyle="--")
    plt.title(f"Cumulative Log Returns ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative log return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_rolling_volatility(df, start_date, end_date, window=30):
    # Compute rolling annualized volatility
    vol_replicated = df["return_replicated"].rolling(window).std() * np.sqrt(252)
    vol_true = df["return_true"].rolling(window).std() * np.sqrt(252)

    # Filter date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    dates_filtered = df.loc[mask, "date"]
    vol_replicated_filtered = vol_replicated.loc[mask]
    vol_true_filtered = vol_true.loc[mask]

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(dates_filtered, vol_replicated_filtered, label="Replicated Index Volatility", color="red")
    plt.plot(dates_filtered, vol_true_filtered, label="True Index Volatility", color="black", linestyle="--")

    plt.title(f"Rolling Annualized Volatility ({window}-Day Window)")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


