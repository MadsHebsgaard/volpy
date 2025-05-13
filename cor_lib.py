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
    if index not in ["OEX", "DJX"]:
        raise ValueError("index skal være enten 'OEX' eller 'DJX'")
    
    base_dir = load.dirs()["OptionMetrics"]
    base_path          =  base_dir / "Tickers"
    constituents_path  = os.path.join(base_path, "index data", f"{index.lower()}_constituents_long.xlsx")
    crsp_path          = os.path.join(base_path, "cor metadata", "crsp_complete_1996_2024.csv")
    output_folder      = os.path.join(base_path, "cor metadata")
    os.makedirs(output_folder, exist_ok=True)
    output_path        = os.path.join(output_folder, f"daily_returns_{index.lower()}.csv")

    # 2) Hvis filen allerede findes, loader vi den
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Fil fundet: {output_path} - loader den i stedet.")
        daily_returns_constituents = pd.read_csv(output_path, parse_dates=["date"])
        constituents_long          = pd.read_excel(constituents_path, parse_dates=["date"])
        return constituents_long, daily_returns_constituents

    # 3) Indlæs constituents og filtrér CRSP
    constituents = pd.read_excel(constituents_path, parse_dates=["date"])
    relevant_permnos     = set(constituents["permno"])
    chunks       = []
    for chunk in pd.read_csv(crsp_path, chunksize=500_000, low_memory=False):
        chunks.append(chunk[chunk["permno"].isin(relevant_permnos)])
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
        crsp[["NEXTDT","permno","DLRET"]]
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




def get_index_constituents_and_returns_v2(index: str, force_overwrite=False):
    """
    Henter sammensætning og daglige afkast (uden delisting returns) for indeks (OEX eller DJX).
    Returnerer:
      - constituents_long: DataFrame med kolonner ['date','ticker']
      - daily_returns_constituents: DataFrame med kolonner ['date','ticker','weight_lag','return']
    """
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # 1) Validér input og sæt filstier
    index = index.upper()
    if index not in ["SPX", "DJX", "OEX"]:
        raise ValueError("index skal være enten 'SPX' eller 'DJX'")

    constituents_index = index
    
    if index == "SPX": constituents_index = "OEX"

    base_dir          = load.dirs()["OptionMetrics"]
    base_path         = base_dir / "Tickers"
    constituents_path = os.path.join(base_path, "index data", f"{constituents_index.lower()}_constituents_long.xlsx")
    crsp_path         = os.path.join(base_path, "cor metadata", "crsp_complete_1996_2024.csv")
    output_folder     = os.path.join(base_path, "cor metadata")
    os.makedirs(output_folder, exist_ok=True)
    output_path       = os.path.join(output_folder, f"daily_returns_{constituents_index.lower()}.csv")
    index_return_path = os.path.join(base_path, "Input", index, "returns and stock price.csv")

    # 2) Load fra cache hvis muligt
    if os.path.exists(output_path) and not force_overwrite:
        cons_long = pd.read_excel(constituents_path, parse_dates=["date"])
        daily_returns = pd.read_csv(output_path, parse_dates=["date"])
        
        # Eksplicit konverter for sikkerhed
        cons_long["date"] = pd.to_datetime(cons_long["date"])
        daily_returns["date"] = pd.to_datetime(daily_returns["date"])
        
        return cons_long, daily_returns

    # 3) Indlæs constituents og filtrér CRSP
    constituents = pd.read_excel(constituents_path, parse_dates=["date"])
    index_returns = pd.read_csv(index_return_path, parse_dates=["date"])
    valid_index_dates = index_returns.loc[index_returns["return"].notna(), "date"]

    relevant_permnos = constituents["permno"].unique()

    # Læs CRSP data i chunks
    chunks = []
    for chunk in pd.read_csv(crsp_path, parse_dates=["date"], chunksize=500_000, low_memory=False):
        chunk.columns = chunk.columns.str.lower()
        chunks.append(chunk[chunk["permno"].isin(relevant_permnos)])
    crsp = pd.concat(chunks, ignore_index=True)

    # 4) Opbyg lookup-tabel mellem trading days og seneste constituent-dato
    trading_days = crsp[["date"]].drop_duplicates().sort_values("date")
    cons_dates = constituents["date"].drop_duplicates().sort_values().values

    # Find seneste constituent-dato for hver handelsdag
    lookup = [
        (td, cons_dates[cons_dates <= td][-1])
        for td in trading_days["date"]
        if any(cons_dates <= td)
    ]
    lookup_df = pd.DataFrame(lookup, columns=["date", "constituent_date"])

    # 5) Udvid constituents til dagligt niveau
    expanded = (
        lookup_df
        .merge(constituents, left_on="constituent_date", right_on="date", suffixes=("_td", "_const"))
        .drop(columns="constituent_date")
        .rename(columns={"date_td": "date"})
        [["date", "permno", "ticker"]]
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )
    constituents_long = expanded[["date", "ticker"]]

    # 6) Merge constituents og CRSP
    merged = (
        expanded
        .merge(crsp[["date", "permno", "prc", "shrout", "ret"]],
               on=["date", "permno"],
               how="left")
    )

    # 7) Rens data
    merged["ret"] = pd.to_numeric(merged["ret"], errors="coerce")
    merged = merged.dropna(subset=["ret", "prc", "shrout"])

    # 8) Beregn vægte
    if index == "OEX":
        merged["market_cap"] = merged["prc"].abs() * merged["shrout"]
        merged["weight"] = merged["market_cap"] / merged.groupby("date")["market_cap"].transform("sum")
    else:
        merged["weight"] = merged["prc"] / merged.groupby("date")["prc"].transform("sum")

    # 9) Beregn lagged vægte
    merged["weight_lag"] = merged.groupby("permno")["weight"].shift(1)

    # 10) Byg final_df
    final_df = (
        merged[["date", "permno", "ticker", "weight_lag", "ret"]]
        .rename(columns={"ret": "return"})
        .dropna(subset=["weight_lag", "return"])
    )

    # 11) Gem og returnér
    final_df["date"] = pd.to_datetime(final_df["date"])  # Sikr korrekt type
    final_df = final_df[final_df["date"].isin(valid_index_dates)]
    final_df.to_csv(output_path, index=False)
    return constituents_long, final_df






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

    base_dir = load.dirs()["OptionMetrics"]
    base_path         = base_dir / "Tickers"
    file_path = os.path.join(base_path, "index data", f"{ticker.lower()}.csv")

    # file_path = fr"C:\Users\axell\Desktop\CBS\data\OptionMetrics\Tickers\index data\{ticker.lower()}.csv"

    # Indlæs og forbered indexafkast
    index_returns_true = pd.read_csv(file_path)
    index_returns_true = index_returns_true.rename(columns={"return": "return_true"})
    index_returns_true["date"] = pd.to_datetime(index_returns_true["date"])

    # Merge på dato
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(index_returns_true, on="date", how="left")

    # Fjern rækker hvor indeksafkastet mangler
    df = df.dropna(subset=["return_true"])

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




import pandas as pd
import numpy as np

def decompose_variance_flexible_weights(
    df: pd.DataFrame,
    window: int = 30,
    average_weights: bool = True,
    mean_method: str = "equal",  # 'equal' eller 'weighted'
    min_pct: float = 0.8
) -> pd.DataFrame:
    """
    Dekomponér rullede variance- og covariance-komponenter vha. fleksible vægte.

    Parametre:
    - df: DataFrame med ["date","ticker","return","weight_lag"]
    - window: rullevindue (dage)
    - average_weights: True -> gennemsnitsvægte, False -> vægte fra sidste dag
    - mean_method: 'equal' eller 'weighted' for beregning af både sigma_mean og rho_mean
    - min_pct: minimum % af dage en ticker skal have retur i vinduet

    Returnerer for hver dato:
      - vol_variance_component
      - vol_covariance_component
      - vol_total_reconstructed
      - rho_mean
      - sigma_mean
      - n_assets_used
      - n_days_used
    """
    df = df.drop_duplicates(subset=["date", "ticker"])

    df = df.sort_values(["date", "ticker"])
    dates = df["date"].drop_duplicates().sort_values().to_list()

    R = df.pivot(index="date", columns="ticker", values="return")
    W = df.pivot(index="date", columns="ticker", values="weight_lag")

    out = []
    req = int(np.ceil(window * min_pct))

    for i in range(window, len(dates)):
        dt  = dates[i]
        win = dates[i-window+1 : i+1]

        r_win = R.loc[win]
        w_win = W.loc[win]

        # 1) filtrér tickers med utilstrækkelig data
        valid = r_win.columns[r_win.notna().sum() >= req]
        if len(valid) < 2:
            continue

        r_sel = r_win[valid].dropna(axis=0)
        w_sel = w_win[valid].loc[r_sel.index]

        n_days = len(r_sel)
        if n_days < 2:
            continue

        # 2) vægte
        w_ts = w_sel.mean(axis=0) if average_weights else w_sel.iloc[-1]

        # 3) beregn sigma og corr
        sigma_t = r_sel.std(ddof=0)
        corr_t  = r_sel.corr()

        # 4) trim til tickers med positiv sigma
        ok = sigma_t.index[sigma_t > 0]
        if len(ok) < 2:
            continue

        wv = w_ts[ok].values
        sv = sigma_t[ok].values
        Cv = corr_t.loc[ok,ok].values

        # 5) variance‐komponenter
        ws = wv * sv
        var_diag    = np.sum((wv**2) * (sv**2))
        var_offdiag = np.sum(np.outer(ws, ws) * Cv) - np.sum(ws**2)
        var_tot     = var_diag + var_offdiag

        # 6) sigma_mean
        if mean_method == "equal":
            sigma_mean = sv.mean()
        else:  # weighted
            # normaliser vægte
            w_norm = wv / wv.sum()
            sigma_mean = np.dot(w_norm, sv)

        # 7) rho_mean
        mask = ~np.eye(len(ok), dtype=bool)
        if mean_method == "equal":
            rho_mean = Cv[mask].mean()
        else:
            # vægtet gennemsnit af off‐diag elementer
            # lav matrix af wv[i]*wv[j]
            W2 = np.outer(wv, wv)
            rho_sum = np.sum(W2[mask] * Cv[mask])
            rho_mean = rho_sum / np.sum(W2[mask])

        out.append({
            "date":                     dt,
            "vol_variance_component":   np.sqrt(var_diag    * 252),
            "vol_covariance_component": np.sqrt(var_offdiag * 252),
            "vol_total_reconstructed":  np.sqrt(var_tot     * 252),
            "rho_mean":                 rho_mean,
            "sigma_mean":               sigma_mean * np.sqrt(252),
            "n_assets_used":            len(ok),
            "n_days_used":              n_days
        })

    return pd.DataFrame(out)


def add_all_variance_components(
    index_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    window: int = 30,
    average_weights: bool = True,
    mean_method: str = "equal",
    min_pct: float = 0.8
) -> pd.DataFrame:
    """
    Tilføj alle volatility‐kolonner til index_df:
      - vol_rolling_true
      - vol_rolling_replicated
      - vol_variance_component
      - vol_covariance_component
      - vol_total_reconstructed
      - rho_mean
      - sigma_mean
      - n_assets_used
      - n_days_used
    """
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx["vol_rolling_true"]       = (
        idx["return_true"]
        .rolling(window)
        .std(ddof=0)
        * np.sqrt(252)
    )
    idx["vol_rolling_replicated"] = (
        idx["return_replicated"]
        .rolling(window)
        .std(ddof=0)
        * np.sqrt(252)
    )

    var_corr_df = decompose_variance_flexible_weights(
        constituents_df,
        window=window,
        average_weights=average_weights,
        mean_method=mean_method,
        min_pct=min_pct
    )
    var_corr_df["date"] = pd.to_datetime(var_corr_df["date"])

    return idx.merge(var_corr_df, on="date", how="left")


import matplotlib.pyplot as plt

def plot_reconstructed_vs_replicated_vol(df, index_ticker, start_date=None, end_date=None):
    # Filtrér datoer hvis angivet
    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.to_datetime(end_date)]

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["date"], plot_df["vol_total_reconstructed"],
            label="Reconstructed Volatility (diagonal + off-diagonal)",
            color="green", linewidth=1)

    plt.plot(plot_df["date"], plot_df["vol_rolling_replicated"],
            label="Realized Replicated Volatility (replicated return series)",
            color="red", linewidth=1)

    plt.plot(plot_df["date"], plot_df["vol_rolling_true"],
            label="Realized True Volatility (actual index return series)",
            color="blue", linewidth=1)

    
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_reconstructed_vs_replicated_vol_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')


    plt.show()


import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_rho_mean(df, index_ticker, start_date=None, end_date=None, plot_change=False, highlight_periods=None):
    """
    Plots the average correlation (rho_mean) over time, optionally including changes
    and multiple shaded highlight periods.

    Parameters:
    - df                : DataFrame with columns ['date','rho_mean']
    - index_ticker      : str, used in filename
    - start_date        : optional str "YYYY-MM-DD" for plot start
    - end_date          : optional str "YYYY-MM-DD" for plot end
    - plot_change       : True to also plot changes in rho_mean
    - highlight_periods : optional list of dicts with 'start', 'end', and 'label' keys
    """
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.to_datetime(end_date)]

    plot_df["rho_change"] = plot_df["rho_mean"].diff()

    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["date"], plot_df["rho_mean"], color="purple", label="ρ (Average)")

    if plot_change:
        plt.plot(plot_df["date"], plot_df["rho_change"], color="orange", linestyle="--", label="Δρ (Change)")

    # Highlight multiple periods
    if highlight_periods:
        for period in highlight_periods:
            start = pd.to_datetime(period["start"])
            end = pd.to_datetime(period["end"])
            label = period.get("label", "")

            plt.axvspan(start, end, color="grey", alpha=0.3)
            plt.text(
                x=start + (end - start) / 10,
                y=plot_df["rho_mean"].max() * 0.95,
                s=label,
                fontsize=10,
                color="black",
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

        # for i, period in enumerate(highlight_periods):
        #     start = pd.to_datetime(period["start"])
        #     end = pd.to_datetime(period["end"])
        #     label = period.get("label", "")

        #     plt.axvspan(start, end, color="grey", alpha=0.3)

        #     # Varier y-positionen for at undgå overlap
        #     y_offset = 0.95 - (i % 5) * 0.12  # Cirkulerer over 5 højder

        #     plt.text(
        #         x=start + (end - start) / 10,
        #         y=plot_df["rho_mean"].max() * y_offset,
        #         s=label,
        #         fontsize=9,
        #         color="black",
        #         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        #     )
            
    plt.xlabel("Date")
    plt.ylabel("ρ")
    plt.grid(True)
    plt.legend(loc="upper left")    
    plt.tight_layout()

    os.makedirs("figures/Analysis/CRP", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_rho_mean_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()




import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_sigma_mean(df, index_ticker, start_date=None, end_date=None, plot_change=False, highlight_periods=None):
    """
    Plots the average ticker volatility (sigma_mean) over time, optionally including changes
    and shaded highlight periods.

    Parameters:
    - df                : DataFrame with columns ['date','sigma_mean']
    - index_ticker      : str, used in filename
    - start_date        : optional str "YYYY-MM-DD" for plot start
    - end_date          : optional str "YYYY-MM-DD" for plot end
    - plot_change       : True to also plot changes in sigma_mean
    - highlight_periods : optional list of dicts with 'start', 'end', and 'label' keys
    """
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.to_datetime(end_date)]

    plot_df["sigma_change"] = plot_df["sigma_mean"].diff()

    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["date"], plot_df["sigma_mean"], color="orange", label="σ (Average)")

    if plot_change:
        plt.plot(plot_df["date"], plot_df["sigma_change"], color="black", linestyle="--", label="Δσ (Change)")

    # Highlight multiple periods
    if highlight_periods:
        for period in highlight_periods:
            start = pd.to_datetime(period["start"])
            end = pd.to_datetime(period["end"])
            label = period.get("label", "")

            plt.axvspan(start, end, color="grey", alpha=0.3)
            plt.text(
                x=start + (end - start) / 10,
                y=plot_df["sigma_mean"].max() * 0.95,
                s=label,
                fontsize=10,
                color="black",
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

    plt.xlabel("Date")
    plt.ylabel("σ")
    plt.grid(True)
    plt.legend(loc="upper left")    
    plt.tight_layout()

    os.makedirs("figures/Analysis/CRP", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_sigma_mean_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()



import pandas as pd
import numpy as np

def decompose_total_variance_fixed_elements_v3(
    df: pd.DataFrame,
    window: int = 30,
    fix_to_constant: bool = False,
    min_pct: float = 0.8,
    min_obs: int = 20, fix_method = "mean"
) -> pd.DataFrame:
    """
    Version 3: Samme som v2, men TOTAL_VARIANS (= vol_total_reconstructed)
    og de faste komponenter beregnes *kun på off‐diagonal* (altså uden idiosynkratisk/diagonal‐risk).

    Parametre:
    - df: DataFrame med ["date","ticker","return","weight_lag"]
    - window: rullevindue i dage
    - fix_to_constant: True -> brug konstante σ̄ og ρ̄ ved off‐diag‐beregningen
    - min_pct: min % af dage en ticker skal have data i vinduet
    - min_obs: min antal fælles dage for at en par‐vis ρ/σ tæller med i fixed‐værdier
    """
    df = df.sort_values(["date", "ticker"])
    dates = df["date"].drop_duplicates().sort_values().to_list()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first")

    # Pivotér returns og laggede vægte
    R = df.pivot(index="date", columns="ticker", values="return")
    W = df.pivot(index="date", columns="ticker", values="weight_lag")

    # Build matrix over fælles observationer
    present = R.notna().astype(int)
    counts  = present.T.dot(present)

    # Rå kovarianser/korrelationer
    raw_cov  = R.cov()
    raw_corr = R.corr()

    # Maskér fixed‐parametre ved min_obs
    fixed_cov  = raw_cov.where(counts >= min_obs)
    fixed_corr = raw_corr.where(counts >= min_obs)

    # Udtræk faste σ fra diagonalen af fixed_cov
    fixed_sigma = pd.Series(
        np.sqrt(np.diag(fixed_cov.values)),
        index=fixed_cov.index
    )

    # Globale gennemsnit, hvis vi fikser konstant
    if fix_to_constant:
        if fix_method == "mean":
            sigma_bar = fixed_sigma.mean()
            rho_bar   = fixed_corr.stack().mean()
        elif fix_method == "median":
            sigma_bar = fixed_sigma.median()
            rho_bar   = fixed_corr.stack().median()
        else:
            raise ValueError("fix_method skal være enten 'mean' eller 'median'")

    out = []
    req = int(np.ceil(window * min_pct))

    for i in range(window, len(dates)):
        dt  = dates[i]
        win = dates[i-window : i]

        r_win = R.loc[win]
        w_win = W.loc[win]

        # 1) filtrér tickers med for få dage
        valid = r_win.columns[r_win.notna().sum() >= req]
        if len(valid) < 2:
            continue

        # 2) drop days with any NaN in these tickers
        r_sel = r_win[valid].dropna(axis=0)
        w_sel = w_win[valid].loc[r_sel.index]
        n_days = len(r_sel)
        if n_days < 2:
            continue

        # 3) vægte fra sidste dag
        w_t = w_sel.iloc[-1]

        # 4) population σ og ρ på renset data
        sigma_t = r_sel.std(ddof=0)
        corr_t  = r_sel.corr()

        # Tickere med positiv σ
        ok = sigma_t.index[sigma_t > 0]
        if len(ok) < 2:
            continue

        # Arrays
        wv = w_t[ok].values
        sv = sigma_t[ok].values
        Cv = corr_t.loc[ok, ok].values
        fsv = fixed_sigma[ok].values
        fcv = fixed_corr.loc[ok, ok].values

        if fix_to_constant:
            fsv[:] = sigma_bar
            fcv[:] = rho_bar
            np.fill_diagonal(fcv, 1.0)

        # Byg covariance matricer
        cov_t    = np.outer(sv,    sv)    * Cv
        cov_fsig = np.outer(fsv,   fsv)   * Cv
        cov_frho = np.outer(sv,    sv)    * fcv

        # Beregn bidrag
        # diag‐bidrag, som vi nu vil ekskludere:
        var_diag = np.sum((wv**2) * (sv**2))

        # off‐diag‐bidrag:
        var_offdiag      = np.dot(wv, cov_t    @ wv) - var_diag
        var_offdiag_sig  = np.dot(wv, cov_fsig @ wv) - np.sum((wv**2) * (fsv**2))
        var_offdiag_rho  = np.dot(wv, cov_frho @ wv) - var_diag  # diag for cov_frho = sv^2

        out.append({
            "date":                    dt,
            # kun off‐diag bidrag annualiseret:
            "vol_total_reconstructed": np.sqrt(max(0, var_offdiag     * 252)),
            "vol_total_fixed_sigma":   np.sqrt(max(0, var_offdiag_sig * 252)),
            "vol_total_fixed_rho":     np.sqrt(max(0, var_offdiag_rho * 252)),
            "n_assets_used":           len(ok),
            "n_days_used":             n_days
        })

    return pd.DataFrame(out)


import pandas as pd
import matplotlib.pyplot as plt

def plot_total_vs_fixed(
    df: pd.DataFrame,
    fixed_col: str,
    fixed_label: str,
    start_date: str = None,
    end_date: str = None
):
    """
    Plot Total vs. én af de faste volatilitetserier.

    Args:
      df           : DataFrame med kolonner ['date','vol_total_reconstructed', fixed_col]
      fixed_col    : kolonnenavn for fixed‐serie (f.eks. 'vol_total_fixed_sigma')
      fixed_label  : etiketten i legend (f.eks. 'Fixed‐Sigma Volatility')
      start_date   : (valgfrit) ISO‐string ’YYYY‐MM‐DD’
      end_date     : (valgfrit) ISO‐string ’YYYY‐MM‐DD’
    """
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(12,6))
    plt.plot(
        plot_df["date"],
        plot_df["vol_total_reconstructed"],
        label="Reconstructed Total Volatility",
        color="black",
        lw=2
    )
    plt.plot(
        plot_df["date"],
        plot_df[fixed_col],
        label=fixed_label,
        color="tab:blue" if "sigma" in fixed_col else "tab:green",
        lw=2
    )

    plt.title(f"Total vs. {fixed_label}")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_both_decompositions(
    df: pd.DataFrame,
    start_date: str = None,
    end_date: str = None
):
    """
    Laver to plots i træk:
      1) Total vs. Fixed‐Sigma
      2) Total vs. Fixed‐Rho
    """
    # 1) Total vs. sigma
    plot_total_vs_fixed(
        df,
        fixed_col="vol_total_fixed_sigma",
        fixed_label="Fixed‐Sigma Volatility",
        start_date=start_date,
        end_date=end_date
    )

    # 2) Total vs. rho
    plot_total_vs_fixed(
        df,
        fixed_col="vol_total_fixed_rho",
        fixed_label="Fixed‐Rho Volatility",
        start_date=start_date,
        end_date=end_date
    )



def plot_volatility_decomposition(
    df: pd.DataFrame,
    start_date: str = None,
    end_date:   str = None
):
    plot_df = df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])

    if start_date:
        plot_df = plot_df[plot_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df['date'] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(14, 6))
    plt.plot(
        plot_df['date'],
        plot_df['vol_total_reconstructed'],
        label='Reconstructed Total Volatility',
        color='black'
    )
    plt.plot(
        plot_df['date'],
        plot_df['vol_total_fixed_sigma'],
        label='Fixed‐Sigma Volatility',
        color='blue'
    )
    plt.plot(
        plot_df['date'],
        plot_df['vol_total_fixed_rho'],
        label='Fixed‐Rho Volatility',
        color='green'
    )

    plt.title('Portfolio Volatility Decomposition')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_rolling_corr(df, window=21*3, from_date='1996-01-01', to_date='2023-12-31'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filtrér på datoer hvis angivet
    if from_date:
        from_date = pd.to_datetime(from_date)
        df = df[df['date'] >= from_date]
    if to_date:
        to_date = pd.to_datetime(to_date)
        df = df[df['date'] <= to_date]

    ts = df.set_index('date')[[
        'vol_total_reconstructed',
        'vol_total_fixed_sigma',
        'vol_total_fixed_rho'
    ]]

    corr_sig = ts['vol_total_reconstructed'].rolling(window).corr(ts['vol_total_fixed_sigma'])
    corr_rho = ts['vol_total_reconstructed'].rolling(window).corr(ts['vol_total_fixed_rho'])

    plt.figure(figsize=(12,5))
    plt.plot(corr_sig.index, corr_sig, label='Corr(Total, Fixedσ)', color='blue')
    plt.plot(corr_rho.index, corr_rho, label='Corr(Total, Fixedρ)', color='green')
    plt.title(f'Rolling {window}-day Correlation with Total Volatility')
    plt.xlabel('Date'); plt.ylabel('Correlation')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_rolling_corr_with_shading(
    df: pd.DataFrame,
    window:     int    = 21*3,
    thresh:     float  = 0.3,
    from_date:  str    = None,
    to_date:    str    = None,
    alpha:      float  = 0.2
):
    """
    Plot rullende korrelation mellem total vol og fixed-σ / fixed-ρ.
    Skygger perioder hvor |corr_sig - corr_rho| > thresh,
    med blå baggrund hvis corr_sig > corr_rho, ellers grøn.
    """
    # Klargør data
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    if from_date:
        df2 = df2[df2['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df2 = df2[df2['date'] <= pd.to_datetime(to_date)]
    ts = df2.set_index('date')[
        ['vol_total_reconstructed','vol_total_fixed_sigma','vol_total_fixed_rho']
    ]

    # Rullende korrelation
    corr_sig = ts['vol_total_reconstructed'].rolling(window).corr(ts['vol_total_fixed_sigma'])
    corr_rho = ts['vol_total_reconstructed'].rolling(window).corr(ts['vol_total_fixed_rho'])
    diff     = corr_sig - corr_rho

    # Plot basis
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(corr_sig.index, corr_sig, label='Corr(Total, Fixedσ)', color='blue')
    ax.plot(corr_rho.index, corr_rho, label='Corr(Total, Fixedρ)', color='green')

    # Helper til shading
    def shade_segments(mask, color):
        in_seg = False
        seg_start = None
        for dt, flag in mask.items():        # <-- brug items() her
            if flag and not in_seg:
                seg_start, in_seg = dt, True
            elif not flag and in_seg:
                ax.axvspan(seg_start, dt, color=color, alpha=alpha)
                in_seg = False
        if in_seg:
            ax.axvspan(seg_start, mask.index[-1], color=color, alpha=alpha)

    # Skift først perioder
    shade_segments(diff >  thresh, 'blue')
    shade_segments(diff < -thresh, 'green')

    ax.set_title(f"Rolling {window}-day Correlation vs. Total Volatility\nShaded when |Δcorr| > {thresh}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_variance_covariance_decomposition(index_df, index_ticker, start_date=None, end_date=None):
    """
    Plots the additive decomposition of total variance into diagonal (variance)
    and off-diagonal (covariance) components over time, with an overlay of total variance.

    Parameters
    ----------
    index_df : pd.DataFrame
        DataFrame containing at least these columns:
            - 'date'                        : datetime-like index or column
            - 'vol_variance_component'      : variance component (to be squared)
            - 'vol_covariance_component'    : covariance component (to be squared)
            - 'vol_total_reconstructed'     : total volatility (to be squared)
    start_date : str or datetime-like, optional
        Lower bound for filtering on 'date' (inclusive).
    end_date   : str or datetime-like, optional
        Upper bound for filtering on 'date' (inclusive).
    """
    # Make a copy and ensure 'date' is datetime
    df = index_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date if requested
    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    # Compute squared components
    v_var = df['vol_variance_component'] ** 2
    v_cov = df['vol_covariance_component'] ** 2
    total_var = df['vol_total_reconstructed'] ** 2
    
    # Plot stacked variance + covariance
    plt.figure(figsize=(6, 5))
    plt.stackplot(
        df['date'],
        v_var,
        v_cov,
        labels=['Diagonal', 'Off-diagonal'],
        colors=['#1f77b4', '#ff7f0e']
    )
    
    # Overlay total variance
    plt.plot(
        df['date'],
        total_var,
        color='black',
        label='Total Variance'
    )
    
    plt.xlabel('Date')
    plt.ylabel('Variance')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_variance_covariance_decomposition_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')


    plt.show()


def plot_variance_covariance_percent(index_df, index_ticker, start_date=None, end_date=None):
    """
    Plots the percentage decomposition of total variance into diagonal (variance)
    and off-diagonal (covariance) components over time.

    Parameters
    ----------
    index_df : pd.DataFrame
        DataFrame containing at least these columns:
            - 'date'                        : datetime-like index or column
            - 'vol_variance_component'      : variance component (to be squared)
            - 'vol_covariance_component'    : covariance component (to be squared)
            - 'vol_total_reconstructed'     : total volatility (to be squared)
    start_date : str or datetime-like, optional
        Lower bound for filtering on 'date' (inclusive).
    end_date   : str or datetime-like, optional
        Upper bound for filtering on 'date' (inclusive).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = index_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Compute squared components (variances)
    v_var = df['vol_variance_component'] ** 2
    v_cov = df['vol_covariance_component'] ** 2
    total_var = df['vol_total_reconstructed'] ** 2

    # Avoid division by zero
    total_var = total_var.replace(0, 1e-10)

    # Compute percentage contributions
    var_pct = 100 * v_var / total_var
    cov_pct = 100 * v_cov / total_var

    # Plot stacked area in percent
    plt.figure(figsize=(6, 5))
    plt.stackplot(
        df['date'],
        var_pct,
        cov_pct,
        labels=['Diagonal %', 'Off-diagonal %'],
        colors=['#1f77b4', '#ff7f0e']
    )

    plt.xlabel('Date')
    plt.ylabel('Percent Contribution to Total Variance')
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_variance_covariance_percent_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    
    plt.show()




import volpy_func_ticker_lib as vtp


def get_sum_df_for_cor(index_ticker, index_tickerlist):

    if index_ticker == "SPX": index_tickerlist = index_tickerlist + ["SPX"]
    sum_df = vtp.concat_ticker_datasets(index_tickerlist, "sum1")
    sum_df["date"] = pd.to_datetime(sum_df["date"])
    sum_df = sum_df.dropna(subset=["SW_0_30", "RV"])

    return sum_df




def calculate_summary_df_active_constituents(sum_df, daily_returns_constituents, ticker_list):
    """
    Beregner equal- og cap-weighted gennemsnit af SW_0_30 og RV for en given liste af tickers.
    Returnerer en DataFrame med 'date' som almindelig kolonne og inkluderer Variance Risk Premium (VRP).
    """



    # Begræns CRSP-data til datoer som findes i sum_df
    valid_dates = sum_df['date'].unique()
    daily_returns_constituents = daily_returns_constituents[
        daily_returns_constituents['date'].isin(valid_dates)
    ]

    final_df = daily_returns_constituents.merge(
        sum_df[["date", "ticker", "SW_0_30", "RV", "close"]],
        on=["date", "ticker"],
        how="left"
    )

    # Filtrer ønskede tickers
    filtered_df = final_df[final_df["ticker"].isin(ticker_list)]

    # Equal-weighted
    equal_weighted = (
        filtered_df.dropna(subset=["SW_0_30", "RV"])
        .groupby("date", as_index=False)
        .agg(
            SW_0_30_equalweight=("SW_0_30", "mean"),
            RV_equalweight=("RV", "mean"),
            n_tickers=("SW_0_30", "count")  # tæller hvor mange SW_0_30 der er
        )
    )

    # Cap-weighted med justering for NaNs
    cap_weighted = (
        filtered_df.dropna(subset=["SW_0_30", "RV", "weight_lag"])
        .groupby("date", group_keys=False)
        .apply(lambda x: pd.Series({
            "SW_0_30_capweight": (x["SW_0_30"] * x["weight_lag"] / x["weight_lag"].sum()).sum(),
            "RV_capweight": (x["RV"] * x["weight_lag"] / x["weight_lag"].sum()).sum()
        }), include_groups=False)  # Pandas >= 2.2 kræver dette for stilhed
        .reset_index()
    )

    
    # Merge på 'date'
    summary_df = pd.merge(equal_weighted, cap_weighted, how="outer", on="date")
    summary_df = summary_df.sort_values("date")

    # Beregn VRP (kun hvor begge komponenter findes)
    summary_df["VRP_equalw"] = summary_df["SW_0_30_equalweight"] - summary_df["RV_equalweight"]
    summary_df["VRP_capw"] = summary_df["SW_0_30_capweight"] - summary_df["RV_capweight"]

    # Fjern rækker uden data i hovedkolonner
    summary_df = summary_df.dropna(how="all", subset=[
        "SW_0_30_equalweight",
        "RV_equalweight",
        "SW_0_30_capweight",
        "RV_capweight"
    ])

    summary_df["date"] = pd.to_datetime(summary_df["date"])
    # Fjern dage med for få aktive tickere
    summary_df = summary_df[summary_df["n_tickers"] >= 1]

    return summary_df


def add_index_vrp(sum_df, sum_df_indexfiltered, ticker="OEX"):
    index_data = sum_df[sum_df["ticker"] == ticker][["date", "SW_0_30", "RV"]].copy()
    index_data = index_data.rename(columns={
        "SW_0_30": "SW_index",
        "RV": "RV_index"
    })
    index_data["VRP_index"] = index_data["SW_index"] - index_data["RV_index"]
    merged = sum_df_indexfiltered.merge(index_data, on="date", how="left")
    return merged





# def calculate_summary_df_active_constituents(sum_df, daily_returns_constituents, ticker_list, topN=None):
#     """
#     Beregner equal- og cap-weighted gennemsnit af SW_0_30 og RV for en given liste af tickers.
#     Returnerer en DataFrame med 'date' som almindelig kolonne og inkluderer Variance Risk Premium (VRP).
#     """

#     sum_df = sum_df[sum_df['ticker'].isin(ticker_list)].copy()
#     sum_df["min_k"] = np.minimum(sum_df["high #K"], sum_df["low #K"])

#     if topN is not None:
#             # sum_df = sum_df.sort_values(['date', '#K'], ascending=[True, False]).groupby('date').head(topN).copy()
#             sum_df_reduced = sum_df.copy()
#             sum_df = sum_df.sort_values(['date', 'min_k'], ascending=[True, False]).groupby('date').head(topN).copy()

#     # Begræns CRSP-data til datoer som findes i sum_df
#     valid_dates = sum_df['date'].unique()
#     daily_returns_constituents = daily_returns_constituents[
#         daily_returns_constituents['date'].isin(valid_dates)
#     ]

#     final_df = daily_returns_constituents.merge(
#         sum_df[["date", "ticker", "SW_0_30", "RV", "close"]],
#         on=["date", "ticker"],
#         how="left"
#     )
#     final_df = final_df.dropna(subset=["SW_0_30", "RV"])

#     # Filtrer ønskede tickers
#     filtered_df = final_df[final_df["ticker"].isin(ticker_list)]

#     # Equal-weighted
#     equal_weighted = (
#         filtered_df.dropna(subset=["SW_0_30", "RV"])
#         .groupby("date", as_index=False)
#         .agg(
#             SW_0_30_equalweight=("SW_0_30", "mean"),
#             RV_equalweight=("RV", "mean"),
#             n_tickers=("SW_0_30", "count")  # tæller hvor mange SW_0_30 der er
#         )
#     )

#     # Cap-weighted med justering for NaNs
#     cap_weighted = (
#         filtered_df.dropna(subset=["SW_0_30", "RV", "weight_lag"])
#         .groupby("date", group_keys=False)
#         .apply(lambda x: pd.Series({
#             "SW_0_30_capweight": (x["SW_0_30"] * x["weight_lag"] / x["weight_lag"].sum()).sum(),
#             "RV_capweight": (x["RV"] * x["weight_lag"] / x["weight_lag"].sum()).sum()
#         }), include_groups=False)  # Pandas >= 2.2 kræver dette for stilhed
#         .reset_index()
#     )

    
#     # Merge på 'date'
#     summary_df = pd.merge(equal_weighted, cap_weighted, how="outer", on="date")
#     summary_df = summary_df.sort_values("date")

#     # Beregn VRP (kun hvor begge komponenter findes)
#     summary_df["VRP_equalw"] = summary_df["SW_0_30_equalweight"] - summary_df["RV_equalweight"]
#     summary_df["VRP_capw"] = summary_df["SW_0_30_capweight"] - summary_df["RV_capweight"]

#     # Fjern rækker uden data i hovedkolonner
#     summary_df = summary_df.dropna(how="all", subset=[
#         "SW_0_30_equalweight",
#         "RV_equalweight",
#         "SW_0_30_capweight",
#         "RV_capweight"
#     ])

#     summary_df["date"] = pd.to_datetime(summary_df["date"])
#     # Fjern dage med for få aktive tickere
#     summary_df = summary_df[summary_df["n_tickers"] >= 1]

#     return summary_df, filtered_df, sum_df_reduced


# def add_index_vrp(sum_df, sum_df_indexfiltered, ticker="OEX"):
#     index_data = sum_df[sum_df["ticker"] == ticker][["date", "SW_0_30", "RV"]].copy()
#     index_data = index_data.rename(columns={
#         "SW_0_30": "SW_index",
#         "RV": "RV_index"
#     })
#     index_data["VRP_index"] = index_data["SW_index"] - index_data["RV_index"]
#     merged = sum_df_indexfiltered.merge(index_data, on="date", how="left")
#     return merged




def plot_sw_vs_rv_index(dataframe, ticker, from_date="1996-01-01", to_date="2023-12-31", sigma_scale=1):
    
    df = dataframe.copy()

    # Filtrér på datoer hvis angivet
    if from_date:
        from_date = pd.to_datetime(from_date)
        df = df[df['date'] >= from_date]
    if to_date:
        to_date = pd.to_datetime(to_date)
        df = df[df['date'] <= to_date]

    df["SW_index"] = df["SW_index"] ** (sigma_scale / 2)
    df["RV_index"] = df["RV_index"] ** (sigma_scale / 2)

    # Plot
    plt.figure(figsize=(12, 4))
    if sigma_scale == 2:
        label_text1 = f"SW"
        label_text1 = f"RV"
    else:
        label_text1 = f"$\\sqrt{{\\mathrm{{SW}}}}$"
        label_text2 = f"$\\sqrt{{\\mathrm{{RV}}}}$"

    plt.plot(df["date"], df["SW_index"], label=label_text1, linewidth=1, color="red")
    plt.plot(df["date"], df["RV_index"], label=label_text2, linewidth=1, color="black")


    plt.xlabel("Date")
    plt.ylabel("Volatility" if sigma_scale == 1 else "Variance" if sigma_scale == 2 else "Scaled Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_sw_vs_rv_index_{ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')


    plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_sw_vs_rv_summary(summary_df, index_ticker, from_date=None, to_date=None, sigma_scale=1, weighting='equal'):
    """
    Plotter SW_0_30 og RV over tid fra en summary_df.
    
    Parameters:
    - summary_df: DataFrame med både equal og cap-weighted kolonner
    - from_date, to_date: Dato-grænser (strings eller datetime)
    - sigma_scale: 1 for volatilitet, 2 for varians, andet for skalering
    - weighting: 'equal' eller 'cap' for vægtet metode
    """
    assert weighting in ['equal', 'cap'], "weighting skal være 'equal' eller 'cap'"

    df = summary_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Filtrér datoer
    if from_date:
        df = df[df["date"] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df["date"] <= pd.to_datetime(to_date)]

    df = df.set_index("date").sort_index()

    # Vælg kolonnenavne baseret på weighting
    sw_col = f"SW_0_30_{weighting}weight"
    rv_col = f"RV_{weighting}weight"

    # Anvend sigma-skala
    df["SW_scaled"] = df[sw_col] ** (sigma_scale / 2)
    df["RV_scaled"] = df[rv_col] ** (sigma_scale / 2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))

    if sigma_scale == 2:
        label_text1 = f"SW"
        label_text1 = f"RV"
    else:
        label_text1 = f"$\\sqrt{{\\mathrm{{SW}}}}$"
        label_text2 = f"$\\sqrt{{\\mathrm{{RV}}}}$"

    # ax.plot(df.index, df["SW_scaled"], label=f"{weighting.title()}-weighted SW_0_30")
    # ax.plot(df.index, df["RV_scaled"], label=f"{weighting.title()}-weighted RV", linestyle="--")

    ax.plot(df.index, df["SW_scaled"], label=label_text1, color = "red", linewidth=1)
    ax.plot(df.index, df["RV_scaled"], label=label_text2, color = "black", linewidth=1)

    ax.set_xlabel("Date")
    ylabel = "Volatility" if sigma_scale == 1 else "Variance" if sigma_scale == 2 else "Scaled value"
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('center')

    fig.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_sw_vs_rv_summary_{index_ticker}_{weighting}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_n_tickers_over_time(df, from_date=None, to_date=None):
    """
    Plotter n_tickers over tid med valgfrit datointerval.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filtrér datoer hvis angivet
    if from_date:
        df = df[df['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['date'] <= pd.to_datetime(to_date)]

    df = df.sort_values('date')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['n_tickers'], marker='o')
    plt.xlabel('Dato')
    plt.ylabel('Antal tickere')
    plt.title(f'Antal tickere over tid ({from_date} til {to_date})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_return_and_sw_coverage_old(daily_returns_constituents, index_ticker, sum_df, from_date=None, to_date=None):
    """
    Plots number of assets with valid return data and number of assets with valid SW_0_30 (after merge)
    using daily_returns_constituents as base and merging in SW from sum_df.

    Parameters
    ----------
    daily_returns_constituents : pd.DataFrame
        Must contain ['date', 'ticker', 'return']
        
    sum_df : pd.DataFrame
        Must contain ['date', 'ticker', 'SW_0_30']
        
    from_date : str or datetime, optional
        Start of date filter (inclusive)
        
    to_date : str or datetime, optional
        End of date filter (inclusive)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Copy and ensure datetime format
    df = daily_returns_constituents.copy()
    sw = sum_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    sw['date'] = pd.to_datetime(sw['date'])

    # Merge SW into returns on ['date', 'ticker']
    df = df.merge(sw[['date', 'ticker', 'SW_0_30']], on=['date', 'ticker'], how='left')

    # Apply optional date filtering
    if from_date:
        df = df[df['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['date'] <= pd.to_datetime(to_date)]

    # Count total assets with return per day
    n_returns = df.groupby('date')['return'].count()

    # Count assets with non-null SW_0_30 per day (i.e. valid implied variance)
    n_sw = df.dropna(subset=['SW_0_30']).groupby('date')['ticker'].nunique()

    # Combine
    combined = pd.DataFrame({'n_returns': n_returns, 'n_sw': n_sw}).sort_index()

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(combined.index, combined['n_returns'], label='Assets with return data')
    plt.plot(combined.index, combined['n_sw'], label='Assets with SW (implied variance)')
    plt.xlabel('Date')
    plt.ylabel('Number of assets')
    plt.title('Data Coverage: Return vs. Implied Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/return_coverage_with_SW_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Output min counts
    print(f"Minimum assets with return data: {n_returns.min()}")
    print(f"Minimum assets with SW_0_30: {n_sw.min()}")


def plot_return_and_sw_coverage(daily_returns_constituents, index_ticker, sum_df, from_date=None, to_date=None, sw_ma_window=21):
    """
    Plots number of assets with valid return data and number of assets with valid SW_0_30 (after merge),
    including a rolling average for SW coverage.

    Parameters
    ----------
    daily_returns_constituents : pd.DataFrame
        Must contain ['date', 'ticker', 'return']
        
    sum_df : pd.DataFrame
        Must contain ['date', 'ticker', 'SW_0_30']
        
    from_date : str or datetime, optional
        Start of date filter (inclusive)
        
    to_date : str or datetime, optional
        End of date filter (inclusive)

    sw_ma_window : int, optional
        Window size for moving average of SW coverage (default is 21 trading days)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Copy and ensure datetime format
    df = daily_returns_constituents.copy()
    sw = sum_df[['date', 'ticker', 'SW_0_30']].copy()
    df['date'] = pd.to_datetime(df['date'])
    sw['date'] = pd.to_datetime(sw['date'])

    # Merge SW into returns on ['date', 'ticker']
    df = df.merge(sw[['date', 'ticker', 'SW_0_30']], on=['date', 'ticker'], how='left')

    # Apply optional date filtering
    if from_date:
        df = df[df['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['date'] <= pd.to_datetime(to_date)]

    # Count total assets with return per day
    n_returns = df.groupby('date')['return'].count()

    # Count assets with non-null SW_0_30 per day (i.e. valid implied variance)
    n_sw = df.dropna(subset=['SW_0_30']).groupby('date')['ticker'].nunique()

    # Combine
    combined = pd.DataFrame({'n_returns': n_returns, 'n_sw': n_sw}).sort_index()
    combined['n_sw_ma'] = combined['n_sw'].rolling(sw_ma_window, min_periods=5).mean()

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(combined.index, combined['n_returns'], color='black', label='Assets with return data')
    plt.plot(combined.index, combined['n_sw'], color='gray', alpha=0.4, label='Assets with SW (implied variance)')
    plt.plot(combined.index, combined['n_sw_ma'], color='gray', linestyle='--', label=f'SW {sw_ma_window}-day MA')
    
    plt.xlabel('Date')
    plt.ylabel('Number of assets')
    plt.title('Data Coverage: Return vs. Implied Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    os.makedirs("figures/Analysis/CRP", exist_ok=True)
    filename = f"figures/Analysis/CRP/return_coverage_with_SW_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()

    # Output min counts
    print(f"Minimum assets with return data: {n_returns.min()}")
    print(f"Minimum assets with SW_0_30: {n_sw.min()}")








def plot_return_coverage(daily_returns_constituents, index_ticker, from_date=None, to_date=None, save=True):
    """
    Plots and optionally saves the number of assets with valid return data over time.

    Parameters
    ----------
    daily_returns_constituents : pd.DataFrame
        Must contain columns ['date', 'return']
        
    from_date : str or datetime, optional
        Start date for filtering
        
    to_date : str or datetime, optional
        End date for filtering

    save : bool, default True
        If True, saves the figure as a PDF in 'figures/Analysis/'
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = daily_returns_constituents.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filter by date
    if from_date:
        df = df[df['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['date'] <= pd.to_datetime(to_date)]

    # Count non-NaN returns per day
    assets_per_day = df.groupby("date")["return"].count()

    # Print min
    print(f"Minimum number of assets with return data: {assets_per_day.min()}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(assets_per_day.index, assets_per_day.values)
    plt.xlabel("Date")
    plt.ylabel("Number of assets with return")
    plt.title("Return Data Coverage Over Time")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/return_coverage_{index_ticker}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()




import pandas as pd
import numpy as np

def construct_swap_pnl(
    daily_returns_constituents: pd.DataFrame,
    sum_df: pd.DataFrame,
    index_ticker: str,
    target_notional: float = 1.0
) -> pd.DataFrame:

    # 1) Merge swap-strike og RV på
    df = daily_returns_constituents.merge(
        sum_df[['date', 'ticker', 'SW_0_30', 'RV']],
        on=['date', 'ticker'],
        how='left'
    ).dropna(subset=['SW_0_30', 'RV', 'weight_lag'])

    # 2) Normalize weights pr. dag
    df['weight_lag'] = (
        df
        .groupby('date', group_keys=False)['weight_lag']
        .transform(lambda w: w / w.sum())
    )

    # 3) Merge index‐data
    index_df = (
        sum_df[sum_df['ticker'] == index_ticker]
        .rename(columns={'SW_0_30': 'K_index', 'RV': 'RV_index'})
        [['date', 'K_index', 'RV_index']]
    )

    df = df.merge(index_df, on='date', how='left')
    df = df.dropna(subset=['K_index', 'RV_index'])

    # 4) Beregn lambda og a_i
    df['denom'] = df['weight_lag']**2 * df['SW_0_30']
    df['denom'] = df.groupby('date')['denom'].transform('sum')
    df['lambda'] = df['K_index'] / df['denom']
    df['a_i'] = df['lambda'] * df['weight_lag']**2

    # 5) Tilføj hjælpe‐kolonner for aggregering
    df['a_i_SW'] = df['a_i'] * df['SW_0_30']
    df['a_i_RV'] = df['a_i'] * df['RV']

    # 6) Aggreger
    agg_df = df.groupby('date', as_index=False).agg(
        SW_market=('K_index', 'first'),
        RV_market=('RV_index', 'first'),
        sum_a_i=('a_i', 'sum'),
        SW_portfolio=('a_i_SW', 'sum'),
        RV_portfolio=('a_i_RV', 'sum'),
    )

    # 7) Beregn PnL
    agg_df['PnL'] = agg_df['RV_portfolio'] - agg_df['RV_market']

    # 8) Skaler til target notional
    agg_df['scaling_factor'] = target_notional / agg_df['SW_market']
    agg_df['SW_portfolio_scaled'] = agg_df['SW_portfolio'] * agg_df['scaling_factor']
    agg_df['RV_portfolio_scaled'] = agg_df['RV_portfolio'] * agg_df['scaling_factor']
    agg_df['RV_market_scaled'] = agg_df['RV_market'] * agg_df['scaling_factor']
    agg_df['SW_market_scaled'] = agg_df['SW_market'] * agg_df['scaling_factor']

    return agg_df

def plot_carry_cumsum(agg_df, index_ticker, from_date="1996-01-01", to_date="2023-12-31", scale=True):
    df = agg_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df = df[(df['date'] >= pd.to_datetime(from_date)) & (df['date'] <= pd.to_datetime(to_date))]

    if scale:
        sw_port = 'SW_portfolio_scaled'
        sw_mkt = 'SW_market_scaled'
        rv_port = 'RV_portfolio_scaled'
        rv_mkt  = 'RV_market_scaled'
    else:
        sw_port = 'SW_portfolio'
        sw_mkt = 'SW_market'
        rv_port = 'RV_portfolio'
        rv_mkt  = 'RV_market'

    # Beregn daglige carry
    df['carry_market'] = df[sw_mkt] - df[rv_mkt]
    df['carry_portfolio'] = df[sw_port] - df[rv_port]

    # Akkumuler over tid
    df['carry_market_cum'] = df['carry_market'].cumsum()
    df['carry_portfolio_cum'] = df['carry_portfolio'].cumsum()
    df['carry_strategy_cum'] = df['carry_market_cum'] - df['carry_portfolio_cum']

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df['date'], df['carry_portfolio_cum'], label='Portfolio of underlying (SW - RV)', color='blue', linewidth=1.8)
    plt.plot(df['date'], df['carry_market_cum'], label='Market (SW - RV)', color='grey', linewidth=1.8)
    plt.plot(df['date'], df['carry_strategy_cum'], label='Strategy (Market - Portfolio)', color='red', linewidth=2)

    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Accumulated PnL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_carry_cumsum_{index_ticker}_{from_date}_{to_date}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
 

    plt.show()




def construct_swap_pnl_rolling_risk_estimate(
    daily_returns_constituents: pd.DataFrame,
    sum_df: pd.DataFrame,
    index_ticker: str,
    target_notional: float = 1.0
):

    # 1) Merge swap-strike og RV på
    df = daily_returns_constituents.merge(
        sum_df[['date', 'ticker', 'SW_0_30', 'RV']],
        on=['date', 'ticker'],
        how='left'
    ).dropna(subset=['SW_0_30', 'RV', 'weight_lag'])

    # 2) Normalize weights pr. dag
    df['weight_lag'] = (
        df
        .groupby('date', group_keys=False)['weight_lag']
        .transform(lambda w: w / w.sum())
    )

    # # 3) Merge index‐data
    # index_df = (
    #     sum_df[sum_df['ticker'] == index_ticker]
    #     .rename(columns={'SW_0_30':'K_index','RV':'RV_index'})
    #     [['date','K_index','RV_index']]
    # )


    # 3) Merge index‐data
    index_df = (
        sum_df[sum_df['ticker'] == index_ticker]
        .rename(columns={'SW_m1_29':'K_index','RV':'RV_index', 'SW_0_30':'SW_index'})
        [['date','K_index','RV_index', 'SW_index']]
    )

    index_df = index_df.sort_values('date')
    index_df['K_index_roll21'] = (
        index_df['K_index']
        .rolling(window=21, min_periods=5)
        .mean()
    )

    # Merge med df – og omdøb den rullende til at erstatte K_index
    df = df.merge(
        index_df[['date','K_index_roll21','RV_index', 'SW_index']],
        on='date',
        how='left'
    ).rename(columns={'K_index_roll21':'K_index'})
    # df = df.dropna(subset=['K_index', 'CF_index'])



    # df = df.merge(index_df, on='date', how='left')



    df = df.dropna(subset=['K_index', 'RV_index'])




    # 4) Beregn lambda og a_i
    df['denom'] = df['weight_lag']**2 * df['SW_0_30']
    df['denom'] = df.groupby('date')['denom'].transform('sum')
    df['lambda'] = df['SW_index'] / df['denom']
    df['a_i'] = df['lambda'] * df['weight_lag']**2

    # 5) Tilføj hjælpe‐kolonner for aggregering
    df['a_i_SW'] = df['a_i'] * df['SW_0_30']
    df['a_i_RV'] = df['a_i'] * df['RV']

    # 6) Aggreger med groupby.agg
    agg_df = df.groupby('date', as_index=False).agg(
        SW_market       = ('SW_index', 'first'),
        RV_market       = ('RV_index', 'first'),
        K_roll = ('K_index', 'first'),
        sum_a_i         = ('a_i', 'sum'),
        SW_portfolio    = ('a_i_SW', 'sum'),
        RV_portfolio    = ('a_i_RV', 'sum'),
    )

    # 7) Beregn PnL
    agg_df['PnL'] = agg_df['RV_portfolio'] - agg_df['RV_market']

    # Beregn skalar
    agg_df['scaling_factor'] = target_notional / agg_df['K_roll']

    # Tilføj skalerede kolonner
    agg_df['SW_portfolio_scaled'] = agg_df['SW_portfolio'] * agg_df['scaling_factor']
    agg_df['RV_portfolio_scaled'] = agg_df['RV_portfolio'] * agg_df['scaling_factor']
    agg_df['RV_market_scaled'] = agg_df['RV_market'] * agg_df['scaling_factor']
    agg_df['SW_market_scaled'] = agg_df['SW_market'] * agg_df['scaling_factor']


    return agg_df
    


def construct_pnl_dly_cf(
    daily_returns_constituents: pd.DataFrame,
    sum_df: pd.DataFrame,
    index_ticker: str,
    target_notional: float = 1.0,
    lambda_equal_1 = False
):

    # 1) Merge swap-strike og RV på
    df = daily_returns_constituents.merge(
        sum_df[['date', 'ticker', 'SW_0_30', 'CF_30_SW_day', 'SW_m1_29']],
        on=['date', 'ticker'],
        how='left'
    ).dropna(subset=['CF_30_SW_day', 'SW_m1_29', 'weight_lag', 'SW_0_30'])

    # 2) Normalize weights pr. dag
    df['weight_lag'] = (
        df
        .groupby('date', group_keys=False)['weight_lag']
        .transform(lambda w: w / w.sum())
    )

    # 3) Merge index‐data
    index_df = (
        sum_df[sum_df['ticker'] == index_ticker]
        .rename(columns={'SW_m1_29':'K_index', 'CF_30_SW_day':'CF_index'})
        [['date','K_index','CF_index']]
    )
    index_df = index_df.sort_values('date')
    index_df['K_index_roll21'] = (
        index_df['K_index']
        .rolling(window=21, min_periods=5)
        .mean()
    )

    # Merge med df – og omdøb den rullende til at erstatte K_index
    df = df.merge(
        index_df[['date','K_index_roll21','CF_index']],
        on='date',
        how='left'
    ).rename(columns={'K_index_roll21':'K_index'})
    df = df.dropna(subset=['K_index', 'CF_index'])

    # 4) Beregn lambda og a_i 
    df['denom'] = df['weight_lag']**2 * df['SW_m1_29']
    df['denom'] = df.groupby('date')['denom'].transform('sum')
    df['lambda'] = df['K_index'] / df['denom']

    if lambda_equal_1:
        df['lambda'] = 1.0

    df['a_i'] = df['lambda'] * df['weight_lag']**2

    # 5) Tilføj hjælpe‐kolonner for aggregering
    df['a_i_SW'] = df['a_i'] * df['CF_30_SW_day']
    # df['a_i_RV'] = df['a_i'] * df['RV']

    # 6) Aggreger med groupby.agg
    agg_df = df.groupby('date', as_index=False).agg(
        SW_market = ('K_index', 'first'),
        sum_a_i = ('a_i', 'sum'),
        CF_portfolio = ('a_i_SW', 'sum'),
        CF_index = ('CF_index', 'first')
    )

    # Beregn skalar
    agg_df['scaling_factor'] = target_notional / agg_df['SW_market']

    # Tilføj skalerede kolonner
    agg_df['PnL_portfolio_scaled'] = agg_df['CF_portfolio'] * agg_df['scaling_factor']
    agg_df['PnL_market_scaled'] = agg_df['CF_index'] * agg_df['scaling_factor']


    return agg_df

def plot_negative_cumulative_pnl_with_diff(agg_df, index_ticker, legs_matched, start_date=None, end_date=None):

    df = agg_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Filtrér på dato, hvis angivet
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    # Beregn negativt akkumuleret PnL
    df["cum_PnL_portfolio"] = -df["PnL_portfolio_scaled"].cumsum()
    df["cum_PnL_market"]    = -df["PnL_market_scaled"].cumsum()
    # Beregn akkumuleret difference
    df["cum_diff"] =  df["cum_PnL_market"] - df["cum_PnL_portfolio"] 

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["cum_PnL_portfolio"], label="Portfolio PnL", color="blue")
    plt.plot(df["date"], df["cum_PnL_market"],    label="Market PnL",    color="gray")
    plt.plot(df["date"], df["cum_diff"],          label="Difference (Mkt - Port)",     color="red")
    plt.xlabel("Date")
    plt.ylabel("Accumulated PnL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_negative_cumulative_pnl_with_diff_{index_ticker}_{legs_matched}_{start_date}_{end_date}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')


    plt.show()



def plot_cumulative_pnl_dly(agg_df, index_ticker, legs_matched, from_date=None, to_date=None):
    # Sikr at 'date' er datetime
    agg_df = agg_df.copy()
    agg_df['date'] = pd.to_datetime(agg_df['date'])

    # Tilføj PnL-kolonner
    agg_df['PnL_portfolio'] = agg_df['r_portfolio']
    agg_df['PnL_market'] = agg_df['r_index']

    # Filtrér på dato
    if from_date:
        from_date = pd.to_datetime(from_date)
        agg_df = agg_df[agg_df['date'] >= from_date]
    if to_date:
        to_date = pd.to_datetime(to_date)
        agg_df = agg_df[agg_df['date'] <= to_date]

    # Gruppér og summer per dag
    daily_sum = agg_df.groupby('date')[['PnL_portfolio', 'PnL_market']].sum().sort_index()

    # Beregn forskel
    daily_sum['PnL_diff'] = daily_sum['PnL_market'] - daily_sum['PnL_portfolio']

    # Akkumuler med negativt fortegn
    daily_sum['PnL_portfolio_cum'] = -daily_sum['PnL_portfolio'].cumsum()
    daily_sum['PnL_market_cum'] = -daily_sum['PnL_market'].cumsum()
    daily_sum['PnL_diff_cum'] = -daily_sum['PnL_diff'].cumsum()

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(daily_sum.index, daily_sum['PnL_portfolio_cum'], label='Portfolio (-cumulative)', color = "blue")
    plt.plot(daily_sum.index, daily_sum['PnL_market_cum'], label='Market (-cumulative)', color = "grey")
    plt.plot(daily_sum.index, daily_sum['PnL_diff_cum'], label='Difference (Market - Portfolio)', color = "red")

    plt.xlabel('Dato')
    plt.ylabel('Accumulated PnL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    from_str = from_date.strftime('%Y-%m-%d') if from_date else "start"
    to_str = to_date.strftime('%Y-%m-%d') if to_date else "end"
    filename = f"figures/Analysis/CRP/plot_cumulative_pnl_dly_{index_ticker}_{legs_matched}_{from_str}_{to_str}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
 
    plt.show()


# import pandas as pd

# def construct_portfolio_cf_dly_scaled(
#     daily_returns_constituents: pd.DataFrame,
#     sum_df: pd.DataFrame,
#     index_ticker: str,
#     risk_col: str,
#     return_col: str
# ) -> pd.DataFrame:
#     """
#     Konstruerer risikojusteret portefølje og marked baseret på swap return og volatilitet.

#     Parametre
#     ---------
#     daily_returns_constituents : pd.DataFrame
#         Indeholder ['date', 'ticker', 'weight_lag', ...]
#     sum_df : pd.DataFrame
#         Indeholder ['date', 'ticker', return_col, risk_col, 'SW_m1_29']
#     index_ticker : str
#         Ticker der bruges som benchmark/index
#     risk_col : str
#         Kolonnenavn for volatilitet (f.eks. 'EWMA SW .20')
#     return_col : str
#         Kolonnenavn for swap return (f.eks. 'r_30_SW_day .20')

#     Returnerer
#     ---------
#     agg_df : pd.DataFrame
#         Aggregerede og risikojusterede værdier for portefølje og benchmark
#     """

#     # 1) Merge swap-strike og return på
#     df = daily_returns_constituents.merge(
#         sum_df[['date', 'ticker', return_col, risk_col, 'SW_m1_29']],
#         on=['date', 'ticker'],
#         how='left'
#     ).dropna(subset=[return_col, 'weight_lag'])

#     # Omdøb til standardiserede navne
#     df = df.rename(columns={risk_col: 'risk', return_col: 'r_30_SW'})

#     # 2) Normalize weights per dag
#     df['weight_lag'] = (
#         df
#         .groupby('date', group_keys=False)['weight_lag']
#         .transform(lambda w: w / w.sum())
#     )

#     # 3) Merge index-data
#     index_df = (
#         sum_df[sum_df['ticker'] == index_ticker]
#         .rename(columns={
#             'SW_m1_29': 'K_index',
#             return_col: 'r_index',
#             risk_col: 'risk_index'
#         })
#         [['date', 'K_index', 'r_index', 'risk_index']]
#     ).sort_values('date')
#     index_df['K_index_2'] = index_df['K_index'] / index_df['risk_index']

#     df = df.merge(
#         index_df[['date', 'r_index', 'risk_index', 'K_index_2']],
#         on='date',
#         how='left'
#     )
#     df = df.dropna(subset=['K_index_2', 'r_index', 'risk_index'])

#     # 4) Beregn lambda og a_i
#     df['denom'] = df['weight_lag']**2 * df['SW_m1_29'] / df['risk']
#     df['denom'] = df.groupby('date')['denom'].transform('sum')
#     df['lambda'] = df['K_index_2'] / df['denom']
#     df['a_i'] = df['lambda'] * df['weight_lag']**2

#     # 5) Hjælpekolonner
#     df['a_i_r'] = df['a_i'] * df['r_30_SW']
#     df['a_i_SW'] = df['a_i'] * df['SW_m1_29'] / df['risk']
#     df['N_market'] = 1 / df['risk_index']
#     df['a_i_2'] = df['a_i'] / df['risk']

#     # 6) Aggreger til dagligt niveau
#     agg_df = df.groupby('date', as_index=False).agg(
#         Fixed_leg_market=('K_index_2', 'first'),
#         Fixed_leg_portfolio=('a_i_SW', 'sum'),
#         sum_a_i=('a_i_2', 'sum'),
#         N_market=('N_market', 'first'),
#         r_portfolio=('a_i_r', 'sum'),
#         r_index=('r_index', 'first'),
#         lambdaa=('lambda', 'first')
#     )

#     return agg_df


def construct_portfolio_return_scaled(
    daily_returns_constituents: pd.DataFrame,
    sum_df: pd.DataFrame,
    index_ticker: str,
    risk_col: str,
    return_col: str
):

    # === Parametrisering ===
    risk_col = 'EWMA SW .20'         # Risiko: rullende EWMA-volatilitet
    return_col = 'r_30_SW_day .20'   # Afkast: 30-dages swap return

    # 1) Merge swap-strike og RV på
    df = daily_returns_constituents.merge(
        sum_df[['date', 'ticker', return_col, risk_col, 'SW_m1_29']],
        on=['date', 'ticker'],
        how='left'
    ).dropna(subset=[return_col, 'weight_lag'])

    # Omdøb til generiske navne
    df = df.rename(columns={risk_col: 'risk', return_col: 'r_30_SW'})

    # 2) Normalize weights pr. dag
    df['weight_lag'] = (
        df
        .groupby('date', group_keys=False)['weight_lag']
        .transform(lambda w: w / w.sum())
    )

    # 3) Merge index‐data
    index_df = (
        sum_df[sum_df['ticker'] == index_ticker]
        .rename(columns={
            'SW_m1_29': 'K_index',
            return_col: 'r_index',
            risk_col: 'risk_index'
        })
        [['date', 'K_index', 'r_index', 'risk_index']]
    )
    index_df = index_df.sort_values('date')
    index_df["K_index_2"] = index_df["K_index"] / index_df["risk_index"]

    # Merge med df – og omdøb den rullende til at erstatte K_index
    df = df.merge(
        index_df[['date', 'r_index', 'risk_index', 'K_index_2']],
        on='date',
        how='left'
    )
    df = df.dropna(subset=['K_index_2', 'r_index', 'risk_index'])

    # 4) Beregn lambda og a_i
    df['denom'] = df['weight_lag']**2 * df['SW_m1_29'] * 1 / df['risk'] # exposure to ith swap
    df['denom'] = df.groupby('date')['denom'].transform('sum') # total exposure to swaps
    df['lambda'] = df['K_index_2'] / df['denom'] # how much to scale?
    df['a_i'] = df['lambda'] * df['weight_lag']**2 # new portfolio weights to match index exposure fixed
    
    # 5) Tilføj hjælpekolonner
    df['a_i_r'] = df['a_i'] * df['r_30_SW']
    df['a_i_SW'] = df['a_i'] * df['SW_m1_29'] * 1 / df["risk"] # we bought 1/risk units
    df['N_market'] = 1 / df['risk_index']
    df['a_i_2'] = df['a_i'] / df["risk"]


    # 6) Aggreger
    agg_df = df.groupby('date', as_index=False).agg(
        Fixed_leg_market=('K_index_2', 'first'),
        Fixed_leg_portfolio=('a_i_SW', 'sum'),
        sum_a_i=('a_i_2', 'sum'),
        N_market=('N_market', 'first'),
        r_portfolio=('a_i_r', 'sum'),
        r_index=('r_index', 'first'),
        lambdaa=('lambda', 'first')
    )
    return agg_df


    # # 5) Tilføj hjælpe‐kolonner for aggregering
    # df['a_i_SW'] = df['a_i'] * df['CF_30_SW_day']
    # # df['a_i_RV'] = df['a_i'] * df['RV']

    # # 6) Aggreger med groupby.agg
    # agg_df = df.groupby('date', as_index=False).agg(
    #     SW_market = ('K_index', 'first'),
    #     sum_a_i = ('a_i', 'sum'),
    #     CF_portfolio = ('a_i_SW', 'sum'),
    #     CF_index = ('CF_index', 'first')
    # )

    # # Beregn skalar
    # agg_df['scaling_factor'] = target_notional / agg_df['SW_market']

    # # Tilføj skalerede kolonner
    # agg_df['PnL_portfolio_scaled'] = agg_df['CF_portfolio'] * agg_df['scaling_factor']
    # agg_df['PnL_market_scaled'] = agg_df['CF_index'] * agg_df['scaling_factor']


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scale_variance_to_unit(agg_df: pd.DataFrame, plot: bool = False):

    df = agg_df.copy()

    # 1. Beregn corstrat
    df["PnL_corstrat_scaled"] = df["PnL_market_scaled"] - df["PnL_portfolio_scaled"]

    # 2. Definér kolonner og beregn oprindelig varians
    cols = ["PnL_market_scaled", "PnL_portfolio_scaled", "PnL_corstrat_scaled"]
    vars_orig = df[cols].var()
    target_var = 1.0
    scale_factors = np.sqrt(target_var / vars_orig)

    # 3. Skaler og lav -cumsum
    for col in cols:
        scaled_col = col + "_varscaled"
        cum_col = scaled_col + "_cum"
        df[scaled_col] = df[col] * scale_factors[col]
        df[cum_col] = -df[scaled_col].cumsum()

    # 4. Plot hvis ønsket
    if plot:
        plt.figure(figsize=(12, 6))
        for col in cols:
            cum_col = col + "_varscaled_cum"
            plt.plot(df.index, df[cum_col], label=col.replace("_scaled", "") + " (-cum. scaled)")
        plt.title("Negativ akkumuleret, varians-skaleret PnL over tid")
        plt.xlabel("Tid")
        plt.ylabel("-Akkumuleret PnL (skaleret)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df




import pandas as pd
import numpy as np

def compute_realized_vs_implied_correlation(
    daily_returns_constituents: pd.DataFrame,
    sum_df: pd.DataFrame,
    index_ticker: str,
    window: int = 21,
    min_assets: int = 20
) -> pd.DataFrame:
    """
    Computes time series of realized vs. implied correlation for an index based on its constituents.

    Parameters
    ----------
    daily_returns_constituents : pd.DataFrame
        Must contain columns: ['date', 'ticker', 'return', 'weight_lag']
    sum_df : pd.DataFrame
        Must contain columns: ['date', 'ticker', 'SW_0_30', 'RV']
    index_ticker : str
        The ticker symbol representing the index.
    window : int, default 21
        The rolling window (in days) for realized variance computation.
    min_assets : int, default 20
        Minimum number of valid assets required for a given date.

    Returns
    -------
    corr_df : pd.DataFrame
        DataFrame with columns ['date', 'realized_correlation', 'implied_correlation', 'n_assets_used']
    """
    # --- Step 0: Datetime formatting
    for df in (daily_returns_constituents, sum_df):
        df['date'] = pd.to_datetime(df['date'])

    # --- Step 1: Merge implied variance and normalize weights
    merged_df = (
        daily_returns_constituents
        .merge(sum_df[['date', 'ticker', 'SW_0_30', 'RV']], on=['date', 'ticker'], how='left')
        .dropna(subset=['SW_0_30'])
        .drop_duplicates(['date', 'ticker'])
    )

    merged_df['weight_norm'] = (
        merged_df.groupby('date')['weight_lag']
        .transform(lambda w: w / w.sum())
    )


    # --- Step 2: Merge index implied variance
    index_df = sum_df.loc[sum_df['ticker'] == index_ticker, ['date', 'SW_0_30']]
    index_df = index_df.rename(columns={'SW_0_30': 'SW_0_30_index'})
    merged_df = merged_df.merge(index_df, on='date', how='left').dropna(subset=['SW_0_30_index'])

    # --- Step 3: Prepare return pivot and lookups
    returns_df = pd.concat([
        daily_returns_constituents[['date', 'ticker', 'return']].drop_duplicates(['date', 'ticker']),
        sum_df.loc[sum_df['ticker'] == index_ticker, ['date', 'return']]
            .drop_duplicates(['date']).assign(ticker=index_ticker)
    ], ignore_index=True)

    pivot = returns_df.pivot(index='date', columns='ticker', values='return')
    pivot_index_lookup = {d: i for i, d in enumerate(pivot.index)}
    ticker_index_lookup = {t: i for i, t in enumerate(pivot.columns)}
    pivot_values = pivot.values

    # --- Step 4: Rolling correlation loop
    dates = merged_df['date'].unique()
    out = []

    for date in dates:
        idx = pivot_index_lookup.get(date)
        if idx is None:
            continue
        start = idx + 1
        end = start + window
        if end > pivot_values.shape[0]:
            continue

        fut_vals = pivot_values[start:end, :]
        idx_col = ticker_index_lookup.get(index_ticker)
        if idx_col is None:
            continue
        ivar = np.nanvar(fut_vals[:, idx_col], ddof=0)

        grp = merged_df[merged_df['date'] == date]
        valid_tickers = [t for t in grp['ticker'] if t in ticker_index_lookup]
        if len(valid_tickers) < min_assets:
            continue

        cols = [ticker_index_lookup[t] for t in valid_tickers]
        fut_sub = fut_vals[:, cols]
        if np.isnan(fut_sub).any():
            continue

        # Vægte og implied varians
        grp = grp.set_index('ticker').loc[valid_tickers]
        w_raw = grp['weight_norm'].values
        w = w_raw / w_raw.sum()

        # Realized correlation
        vars_ = np.nanvar(fut_sub, axis=0, ddof=0)
        std = np.sqrt(vars_)
        diag_r = np.dot(w**2, vars_)
        ws_r = np.dot(w, std)
        off_r = ws_r**2 - diag_r
        rho_realized = (ivar - diag_r) / off_r if off_r != 0 else np.nan

        # Implied correlation
        s2 = grp['SW_0_30'].values
        s2i = grp['SW_0_30_index'].iloc[0]
        diag_i = np.dot(w**2, s2)
        ws_i = np.dot(w, np.sqrt(s2))
        off_i = ws_i**2 - diag_i
        rho_implied = (s2i - diag_i) / off_i if off_i != 0 else np.nan

        out.append({
            'date': date,
            'realized_correlation': rho_realized,
            'implied_correlation': rho_implied,
            'n_assets_used': len(valid_tickers)
        })

    corr_df = pd.DataFrame(out).sort_values('date').reset_index(drop=True)
    return corr_df


def plot_correlation_comparison(
    corr_df,
    index_ticker,
    from_date=None,
    to_date=None,
    smooth_window=21,
    plot_difference=False
):
    df = corr_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if from_date:
        fd = pd.to_datetime(from_date)
        df = df[df['date'] >= max(fd, df['date'].min())]
    if to_date:
        td = pd.to_datetime(to_date)
        df = df[df['date'] <= min(td, df['date'].max())]
    if df.empty:
        print("Ingen data i intervallet.")
        return

    df['implied_smooth'] = df['implied_correlation'].rolling(smooth_window, min_periods=smooth_window).mean()
    df['realized_smooth'] = df['realized_correlation'].rolling(smooth_window, min_periods=smooth_window).mean()
    df['difference'] = df['implied_smooth'] - df['realized_smooth']

    # Fjern rækker med NaN fra glidende gennemsnit
    df = df.dropna(subset=['implied_smooth', 'realized_smooth'])

    smooth_str = f' (smoothed {smooth_window}d)' if smooth_window > 1 else ''

    plt.figure(figsize=(12, 3))
    plt.plot(df['date'], df['implied_smooth'], label=f'Implied{smooth_str}', linewidth=1, color = 'red')
    plt.plot(df['date'], df['realized_smooth'], label=f'Realized{smooth_str}', linewidth=1, color = "black")

    if plot_difference:
        plt.plot(df['date'], df['difference'], linestyle='--', color='grey', label='Difference (Implied - Realized)')

    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("figures/Analysis", exist_ok=True)
    filename = f"figures/Analysis/CRP/plot_correlation_comparison_{index_ticker}_{smooth_window}d.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()



def fast_bootstrap_volatility(df, window=30, n_bootstrap=500, sample_pct=0.7, min_pct=0.9):
    """
    Fast bootstrap volatility reconstruction using NumPy arrays.

    Parameters:
    - df: DataFrame with ['date', 'ticker', 'return', 'weight_lag']
    - window: size of rolling window in days
    - n_bootstrap: number of bootstrap iterations
    - sample_pct: share of stocks to sample
    - min_pct: min % of window days an asset must have to be included

    Returns:
    - DataFrame with ['date', 'vol_median', 'vol_lower', 'vol_upper']
    """
    import pandas as pd
    import numpy as np

    df = df.drop_duplicates(subset=['date', 'ticker']).copy()
    df['date'] = pd.to_datetime(df['date'])

    all_dates = df['date'].sort_values().unique()
    results = []

    for i in range(window, len(all_dates)):
        end_date = all_dates[i]
        start_date = all_dates[i - window + 1]
        win_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        pivot_r = win_df.pivot(index='date', columns='ticker', values='return')
        pivot_w = win_df.pivot(index='date', columns='ticker', values='weight_lag')

        # Keep only tickers with sufficient data
        valid_tickers = pivot_r.columns[pivot_r.notna().sum() >= int(window * min_pct)]
        if len(valid_tickers) < 5:
            continue

        R = pivot_r[valid_tickers].dropna(axis=0)
        W = pivot_w[valid_tickers].loc[R.index]

        if len(R) < 2:
            continue

        weights = W.mean().values  # shape: (n_assets,)
        returns = R.values         # shape: (n_days, n_assets)
        sigmas = returns.std(axis=0, ddof=0)
        valid_idx = sigmas > 0

        if valid_idx.sum() < 5:
            continue

        weights = weights[valid_idx]
        sigmas = sigmas[valid_idx]
        returns = returns[:, valid_idx]

        rho = np.corrcoef(returns.T)

        n_assets = len(sigmas)
        vol_list = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n_assets, int(n_assets * sample_pct), replace=False)
            w = weights[idx]
            s = sigmas[idx]
            w = w / np.sum(w)  # ✅ Normalisér vægtene så de summer til 1
            r = rho[np.ix_(idx, idx)]

            ws = w * s
            var_diag = np.sum((w ** 2) * (s ** 2))
            var_offdiag = np.sum(np.outer(ws, ws) * r) - np.sum(ws ** 2)
            var_total = var_diag + var_offdiag
            vol_list.append(np.sqrt(var_total * 252))

        vol_arr = np.array(vol_list)
        results.append({
            'date': end_date,
            'vol_median': np.median(vol_arr),
            'vol_lower': np.percentile(vol_arr, 5),
            'vol_upper': np.percentile(vol_arr, 95)
        })

    return pd.DataFrame(results)



def plot_bootstrap_stacked(bootstrap_dfs,true_vol_df, index_ticker,from_date=None,to_date=None,true_label='True index volatility'
):
    """
    Plot multiple bootstrap results in stacked subplots, sharing x-axis.

    Parameters:
    - bootstrap_dfs: dict of {label: DataFrame} for each sample_pct
    - true_vol_df: DataFrame with ['date', 'vol_rolling_true']
    - from_date, to_date: optional date filters
    - true_label: label for the true volatility line
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    n = len(bootstrap_dfs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, (label, df) in zip(axes, bootstrap_dfs.items()):
        merged = pd.merge(df, true_vol_df[['date', 'vol_rolling_true']], on='date', how='inner')

        if from_date:
            merged = merged[merged['date'] >= pd.to_datetime(from_date)]
        if to_date:
            merged = merged[merged['date'] <= pd.to_datetime(to_date)]

        ax.plot(merged['date'], merged['vol_rolling_true'], label=true_label, color='black')
        ax.fill_between(
            merged['date'],
            merged['vol_lower'],
            merged['vol_upper'],
            color='gray',
            alpha=0.3,
            label=f'90% CI ({label} sample)'
        )
        ax.set_ylabel('Annualized Volatility')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Date')
    plt.tight_layout()

    # Save to file
    os.makedirs("figures/Analysis/CRP", exist_ok=True)
    sample_labels = "_".join([str(label).replace('%','') for label in bootstrap_dfs.keys()])
    filename = f"figures/Analysis/CRP/bootstrap_stacked_{sample_labels}_{index_ticker}.pdf"    
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()
