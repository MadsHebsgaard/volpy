import pandas as pd
import numpy as np
import os
import volpy_func_lib as vp
from load_clean_lib import ticker_to_vol
from volpy_func_lib import Cross_AM_tickers


def CarrWu_tickers():
    ticker_order = [
        "SPX", "OEX", "DJX", "NDX", "QQQ", "MSFT", "INTC", "IBM", "AMER",
        "DELL", "CSCO", "GE", "CPQ", "YHOO", "SUNW", "MU", "MO", "AMZN",
        "ORCL", "LU", "TRV", "WCOM", "TYC", "AMAT", "QCOM", "TXN", "PFE",
        "MOT", "EMC", "HWP", "AMGN", "BRCM", "MER", "NOK", "CHL", "UNPH",
        "EBAY", "JNPR", "CIEN", "BRCD"
    ]
    return ticker_order

def CarrWu_order(df):
    ticker_order = CarrWu_tickers()
    # Create a mapping from ticker to its order
    order_dict = {ticker: idx for idx, ticker in enumerate(ticker_order)}

    # Map each ticker in the DataFrame to its order; tickers not in ticker_order become NaN.
    df['sort_order'] = df['ticker'].map(order_dict)

    # Replace NaN (tickers not in ticker_order) with a value that puts them after the defined tickers.
    df['sort_order'] = df['sort_order'].fillna(len(ticker_order))

    # Now sort by this order
    df_ordered = df.sort_values('sort_order')

    # Optionally, drop the auxiliary sort column if it's no longer needed
    df_ordered = df_ordered.drop(columns=['sort_order'])
    return df_ordered


# def CarrWu2009_table_1(df, name):
#     if name == "VIX":
#         df = df[df["ticker"] != "TLT"]
#
#     df_nonan = df[df["SW_0_30"].notna()]
#
#     # add Q5_K aggregation
#     table_df = (
#         df_nonan
#         .groupby("ticker")
#         .agg(
#             Starting_date=("date", "min"),
#             Ending_date  =("date", "max"),
#             N            =("SW_0_30", "count"),
#             NK           =("#K",       "mean"),
#             Q5_K         =("#K",       lambda x: x.quantile(0.05)),
#             Q10_K         =("#K", lambda x: x.quantile(0.10))
#     )
#         .reset_index()
#     )
#
#     # format dates and types
#     table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"]).dt.strftime("%d-%b-%Y")
#     table_df["Ending_date"]   = pd.to_datetime(table_df["Ending_date"]).dt.strftime("%d-%b-%Y")
#     table_df["N"]   = table_df["N"].astype(int)
#     table_df["NK"]  = table_df["NK"].astype(float)
#     table_df["Q5_K"] = table_df["Q5_K"].astype(float)
#     table_df["Q10_K"] = table_df["Q10_K"].astype(float)
#
#     # table_df = CarrWu_order(table_df)
#     table_df = table_df.sort_values("NK", ascending=False).reset_index(drop=True)
#
#     # include Q5_K in the LaTeX and save
#     latex_code = table_df.to_latex(
#         columns=["ticker", "Starting_date", "Ending_date", "N", "NK", "Q5_K", "Q10_K"],
#         index=False,
#         header=["ticker", "Starting date", "Ending date", "N", "NK", "5\\% q. NK", "10\\% q. NK"],
#         float_format="%.1f"
#     )
#
#     full_table = (
#         r"\begin{table}[ht]" "\n"
#         r"\centering" "\n\n"
#         + latex_code + "\n"
#         rf"\caption{{List of stocks and stock indexes (ETFs) in the {name} sample.}}" "\n"
#         rf"\label{{tab:data_summary_table_{name}}}" "\n"
#         r"\end{table}"
#     )
#
#     # write out to a .txt
#     out_path = f'figures/summary/data_summary_table_{name}.tex'
#     with open(out_path, 'w') as f:
#         f.write(full_table)
#     print(f"Wrote full {name} LaTeX table to {out_path}")
#
#     return table_df



def table_dataset_list_strike_count(df, name, width_scale=0.75):
    if name == "VIX":
        df = df[df["ticker"] != "TLT"]

    df_nonan = df[df["SW_0_30"].notna()]

    table_df = (
        df_nonan
        .groupby("ticker")
        .agg(
            Starting_date=("date", "min"),
            Ending_date  =("date", "max"),
            N            =("date", "count"),
            NK           =("#K",   "mean"),
            Q1_K         =("#K", lambda x: x.quantile(0.01)),
            Q5_K         =("#K", lambda x: x.quantile(0.05)),
            Q10_K        =("#K", lambda x: x.quantile(0.10))
        )
        .reset_index()
    )

    table_df = table_df.sort_values("NK", ascending=False).reset_index(drop=True)

    # ─── insert row-number column ─────────────────────────────────────────────
    table_df.insert(0, "No.", table_df.index + 1)
    # ──────────────────────────────────────────────────────────────────────────

    # format dates and types
    table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"]).dt.strftime("%d-%b-%Y")
    table_df["Ending_date"]   = pd.to_datetime(table_df["Ending_date"]).dt.strftime("%d-%b-%Y")
    table_df["N"]    = table_df["N"].astype(int)
    table_df["NK"]   = table_df["NK"].astype(float)
    table_df["Q1_K"] = table_df["Q1_K"].astype(float)
    table_df["Q5_K"] = table_df["Q5_K"].astype(float)
    table_df["Q10_K"]= table_df["Q10_K"].astype(float)

    # get raw LaTeX
    raw = table_df.to_latex(
        index=False,
        header=False,
        float_format="%.1f"
    )

    # pull out only the data rows
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == r'\midrule') + 1
    end   = next(i for i, l in enumerate(lines) if l.strip() == r'\bottomrule')
    body  = "\n".join(lines[start:end])

    # build full table with updated header (added "No." at left)
    full_table = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\renewcommand{\arraystretch}{1}" "\n"
        rf"\adjustbox{{max width={width_scale}\textwidth}}{{" "\n"
        # changed {lllrrrrr} → {rlllrrrrr} to make room for No.
        r"\begin{tabular}{rlllrrrrr}" "\n"
        r"\toprule" "\n"
        r"No.\ & Ticker & Starting date & Ending date & Days & \multicolumn{4}{c}{Number of strikes} \\" "\n"
        r"\cmidrule(lr){6-9}" "\n"
        r" &  &  &  &  & Mean & 1\% & 5\% & 10\% \\" "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        rf"\caption{{List of stocks and stock indexes in the {name} sample.}}" "\n"
        rf"\label{{tab:data_summary_table_{name}}}" "\n"
        r"\end{table}"
    )

    out_path = f'figures/summary/data_summary_table_{name}.tex'
    with open(out_path, 'w') as f:
        f.write(full_table)

    return table_df


def table_dataset_list_strike_count_pages(df, name, width_scale=0.75):

    df = df.dropna(subset=["RV", "SW_0_30"]).copy()

    table_df = (
        df
        .groupby("ticker")
        .agg(
            Starting_date=("date", "min"),
            Ending_date=("date", "max"),
            N=("date", "count"),
            NK=("#K", "mean"),
            Q1_K=("#K", lambda x: x.quantile(0.01)),
            Q5_K=("#K", lambda x: x.quantile(0.05)),
            Q10_K=("#K", lambda x: x.quantile(0.10))
        )
        .reset_index()
    )
    table_df = sort_table_df(table_df, df, name)

    table_df.insert(0, "No.", table_df.index + 1)

    # Formatting
    table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"]).dt.strftime("%d-%b-%Y")
    table_df["Ending_date"] = pd.to_datetime(table_df["Ending_date"]).dt.strftime("%d-%b-%Y")
    table_df["N"] = table_df["N"].astype(int)
    table_df["NK"] = table_df["NK"].round(1)
    table_df["Q1_K"] = table_df["Q1_K"].round(1)
    table_df["Q5_K"] = table_df["Q5_K"].round(1)
    table_df["Q10_K"] = table_df["Q10_K"].round(1)
    table_df["Name"] = [name.replace("&", r"\&") for name in vp.ticker_list_to_ordered_map(table_df["ticker"])["name"]]

    # LaTeX generation
    raw = table_df.to_latex(index=False, header=False, float_format="%.1f")
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == r'\midrule') + 1
    end = next(i for i, l in enumerate(lines) if l.strip() == r'\bottomrule')
    body = "\n".join(lines[start:end])

    num_rows = len(table_df)

    # Font size mapping (approximate scaling)
    font_sizes = {
        0.6: r"\scriptsize",
        0.7: r"\footnotesize",
        0.75: r"\small",
        0.9: r"\normalsize",
        1.0: r"\normalsize"
    }
    font_size = font_sizes.get(width_scale, r"\small")
    if num_rows > 50:
        full_table = (
            rf"{font_size}" + "\n"
            r"\begin{longtable}{@{}rlrlrrrrrl@{}}" + "\n"
            rf"\caption{{List of stocks and stock indexes in the {name} sample}} \\" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-9}" + "\n"
            r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
            r"\midrule" + "\n"
            r"\endfirsthead" + "\n\n"
            r"\multicolumn{10}{c}{{\tablename\ \thetable{} -- Continued}} \\" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-9}" + "\n"
            r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
            r"\midrule" + "\n"
            r"\endhead" + "\n\n"
            r"\midrule" + "\n"
            r"\multicolumn{10}{r}{{Continued}} \\" + "\n"
            r"\endfoot" + "\n\n"
            r"\bottomrule" + "\n"
            r"\endlastfoot" + "\n\n"
            + body + "\n"
            rf"\caption{{List of stocks and stock indexes in the {name} sample}}" + "\n"
            rf"\label{{tab:data_summary_{name}}}" + "\n"
            r"\end{longtable}" + "\n"
            r"\normalsize"  # Reset font size
        )
    else:
        full_table = (
            r"\begin{table}[ht]" + "\n"
            r"\centering" + "\n"
            rf"{font_size}" + "\n"
            r"\begin{tabular}{rlrlrrrrrl}" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-9}" + "\n"
            r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
            r"\midrule" + "\n"
            + body + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            rf"\caption{{List of stocks and stock indexes in the {name} sample}}" + "\n"
            rf"\label{{tab:data_summary_{name}}}" + "\n"
            r"\end{table}" + "\n"
            r"\normalsize"
        )

    out_path = f'figures/summary/data_summary_table_{name}.tex'
    with open(out_path, 'w') as f:
        f.write(full_table)
    return table_df

sort_map = {
    "VIX":      True,
    "Cross-AM": True,
    "OEX":      True,
    "Liquid":   True,
    "CarrWu":   True,
    "DJX":      True,
}

def sort_table_df(table_df, sum_df, name):
    """
    Given a DataFrame with an 'NK' column and a 'ticker' column,
    optionally sorts by asset class & group (Index→ETF→Other) then NK;
    otherwise just by NK descending.
    Drops helper columns and preserves original column order.
    """

    df = table_df.copy()
    idx_CAM_set = set(vp.Index_tickers)
    etf_set     = set(vp.ETF_tickers)
    orig_cols   = df.columns.tolist()

    # 1) compute temp-NK
    means = sum_df.groupby("ticker")["#K"].mean()
    df["NK_tmp"] = df["ticker"].map(means)

    if sort_map[name]:
        # assign group: 0=Index, 1=ETF, 2=Other
        df["group"]       = df["ticker"].map(
            lambda t: 0 if t in idx_CAM_set
                      else (1 if t in etf_set else 2)
        )
        # add asset_class (0–4) for sorting
        df["asset_class"] = df["ticker"].map(vp.ticker_to_asset_code)

        # sort by asset_class ↑, group ↑, NK_tmp ↓
        df = (
            df
            .sort_values(
                ["asset_class", "group", "NK_tmp"],
                ascending=[True, True, False]
            )
            .reset_index(drop=True)
        )
        # restore only the original columns (dropping helpers)
        df = df[orig_cols]
    else:
        df = (
            df
            .sort_values("NK_tmp", ascending=False)
            .reset_index(drop=True)
        )

    return df



import pandas as pd

def CarrWu2009_table_2(df, name):
    df = df.dropna(subset=["RV", "SW_0_30"]).copy()

    def compute_stats(sub_df, col):
        scaled = sub_df[col] * 100
        return pd.Series({
            "Mean":     scaled.mean(),
            "Std. dev.":scaled.std(),
            "Auto":     scaled.autocorr(lag=1),
            "Skew":     scaled.skew(),
            "Kurt":     scaled.kurt()
        })

    # Panel A
    df_rv = (df[df["RV"].notna()]
             .groupby("ticker")
             .apply(lambda g: compute_stats(g, "RV"))
             .reset_index()
            )
    df_rv.columns = ["ticker","Mean_RV","Std_RV","Auto_RV","Skew_RV","Kurt_RV"]

    # Panel B
    df_sw = (df[df["SW_0_30"].notna()]
             .groupby("ticker")
             .apply(lambda g: compute_stats(g, "SW_0_30"))
             .reset_index()
            )
    df_sw.columns = ["ticker","Mean_SW","Std_SW","Auto_SW","Skew_SW","Kurt_SW"]

    # merge panels
    out = pd.merge(df_rv, df_sw, on="ticker", how="inner")

    # sort
    out = sort_table_df(out, df, name)
    out["ticker"] = vp.ticker_list_to_ordered_map(out["ticker"])["ticker_out"]

    # 3) generate & save LaTeX
    latex = CarrWu2009_table_2_latex_v2(out, name)   # now returns a string
    with open(f"figures/Analysis/{name}_table_2.tex","w") as f:
        f.write(latex)

    return out

def CarrWu2009_table_2_latex_v2(table_df, name):
    # ——— reorder & set up MultiIndex (unchanged) ———
    table_df = table_df[
        ["ticker",
         "Mean_RV","Std_RV","Auto_RV","Skew_RV","Kurt_RV",
         "Mean_SW","Std_SW","Auto_SW","Skew_SW","Kurt_SW"]
    ]
    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "No."),
        ("",    "Ticker"),
        ("Panel A: Realized variance, RV×100",  "Mean"),
        ("Panel A: Realized variance, RV×100",  "Std. dev."),
        ("Panel A: Realized variance, RV×100",  "Auto"),
        ("Panel A: Realized variance, RV×100",  "Skew"),
        ("Panel A: Realized variance, RV×100",  "Kurt"),
        ("Panel B: Variance swap rate, SW×100",  "Mean"),
        ("Panel B: Variance swap rate, SW×100",  "Std. dev."),
        ("Panel B: Variance swap rate, SW×100",  "Auto"),
        ("Panel B: Variance swap rate, SW×100",  "Skew"),
        ("Panel B: Variance swap rate, SW×100",  "Kurt"),
    ])

    # ——— generate a longtable instead of a tabular ———
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            f"Summary statistics for the realized variance and the synthetic swap rate for the {name} dataset. "
            "Autocorrelation (\"Auto\") is not adjusted for serial dependence, hence its high value, "
            "and kurtosis (\"Kurt\") is excessive."
        ),
        label=f"tab:analysis:Summary statistics realized variance and swap ({name})",
        column_format='rlrrrrr||rrrrr',
        multicolumn_format="c"
    )

    # ——— inject the cmidrule lines under the top header ———
    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            # after \toprule and the header row comes \midrule;
            # so insert two lines down
            lines.insert(i + 2, r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}')
            break

    # ——— add size commands & float barrier ———
    lines.insert(0, r'\scriptsize')
    lines.append(r'\normalsize')

    return "\n".join(lines)


def CarrWu2009_table_2_latex(table_df):
    # reorder & set up MultiIndex (same as before)
    table_df = table_df[
        ["ticker",
         "Mean_RV","Std_RV","Auto_RV","Skew_RV","Kurt_RV",
         "Mean_SW","Std_SW","Auto_SW","Skew_SW","Kurt_SW"]
    ]

    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))

    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "No."),
        ("",    "Ticker"),
        ("Panel A: Realized variance, RV×100",  "Mean"),
        ("Panel A: Realized variance, RV×100",  "Std. dev."),
        ("Panel A: Realized variance, RV×100",  "Auto"),
        ("Panel A: Realized variance, RV×100",  "Skew"),
        ("Panel A: Realized variance, RV×100",  "Kurt"),
        ("Panel B: Variance swap rate, SW×100",  "Mean"),
        ("Panel B: Variance swap rate, SW×100",  "Std. dev."),
        ("Panel B: Variance swap rate, SW×100",  "Auto"),
        ("Panel B: Variance swap rate, SW×100",  "Skew"),
        ("Panel B: Variance swap rate, SW×100",  "Kurt"),
    ])

    # generate base LaTeX with vertical separator
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        column_format='rlrrrrr||rrrrr',   # r for No., l for Ticker, etc.
        multicolumn_format="c",
    )

    # inject the cmidrule lines
    lines = latex.splitlines()
    # find where to insert (after the multicolumn header)
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            insert_pos = i + 2
            break
    lines.insert(insert_pos, r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}')

    lines.insert(0, r'\scriptsize')
    lines.append(r'\normalsize')

    return '\n'.join(lines)





def newey_west_t_stat(series, lag=30):
    """
    Computes the Newey–West adjusted t-statistic for the mean of the series.

    Parameters:
      series : array-like
          The data series.
      lag : int
          The lag length to use for the Newey–West estimator.

    Returns:
      t_stat : float
          The t-statistic for the mean.
    """
    x = np.asarray(series)
    n = len(x)
    if n == 0:
        return np.nan
    x_bar = np.mean(x)

    # gamma(0): sample variance
    gamma0 = np.mean((x - x_bar) ** 2)

    # Compute autocovariances for lags 1 through lag
    gamma_s = []
    for s in range(1, lag + 1):
        if n - s <= 0:
            break
        gamma = np.sum((x[s:] - x_bar) * (x[:-s] - x_bar)) / n
        gamma_s.append(gamma)

    # Apply weights: 1 - s/(lag+1)
    weights = [1 - (s / (lag + 1)) for s in range(1, lag + 1)]

    # Newey–West variance of the sample mean:
    nw_var = (gamma0 + 2 * np.sum([w * g for w, g in zip(weights, gamma_s)])) / n
    nw_se = np.sqrt(nw_var)

    return x_bar / nw_se if nw_se > 0 else np.nan


def CarrWu2009_table_3(df, name):
    # ensure output dir
    os.makedirs("figures/Analysis", exist_ok=True)

    # 1) Prep the data
    df = df.dropna(subset=["RV", "SW_0_30"]).copy()
    df["diff"]    = df["RV"] - df["SW_0_30"]
    df["lnratio"] = np.log(df["RV"]) - np.log(df["SW_0_30"])

    # 1) add this helper right after newey_west_t_stat
    def newey_west_std(series, lag=21):
        x = np.asarray(series)
        n = len(x)
        if n == 0:
            return np.nan
        x_bar = np.mean(x)
        # γ₀ and γ_s as in your t-stat
        gamma0 = np.mean((x - x_bar) ** 2)
        gamma_s = [
            np.sum((x[s:] - x_bar) * (x[:-s] - x_bar)) / n
            for s in range(1, lag + 1) if n - s > 0
        ]
        weights = [1 - s / (lag + 1) for s in range(1, lag + 1)]
        # long-run variance of the series
        longrun_var = gamma0 + 2 * sum(w * g for w, g in zip(weights, gamma_s))
        return np.sqrt(longrun_var)

    # 2) in your compute_stats(), capture this NW std
    def compute_stats(series):
        mean_ = series.mean()
        std_ = series.std()  # you can keep this if you still want “Std”
        auto_ = series.autocorr(lag=21)
        skew_ = series.skew()
        kurt_ = series.kurt()
        t_ = newey_west_t_stat(series, lag=21)
        nw_std = newey_west_std(series, lag=21)
        return pd.Series({
            "Mean": mean_,
            "Std": std_,
            "Auto": auto_,
            "Skew": skew_,
            "Kurt": kurt_,
            "t": t_,
            "NW_std": nw_std  # ← new column
        })


    # Panel A: (RV - SW_0_30) × 100
    df_diff = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["diff"] * 100))
        .reset_index()
    )
    df_diff = df_diff.drop(columns=["NW_std"])
    df_diff.columns = [
        "ticker",
        "Mean_diff", "Std_diff", "Auto_diff",
        "Skew_diff", "Kurt_diff", "t_diff"
    ]

    # Panel B: ln(RV / SW_0_30)
    df_ln = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["lnratio"]))
        .reset_index()
    )
    df_ln.columns = ["ticker", "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln", "NW_std"]

    # Compute Sharpe ratios for Panel B
    df_ln["SR_30d"] = -df_ln["Mean_ln"] / df_ln["NW_std"]
    df_ln["SR_ann"] = df_ln["SR_30d"] * np.sqrt(12)
    df_ln = df_ln.drop(columns=["NW_std"])


    # Merge & sort
    out = pd.merge(df_diff, df_ln, on="ticker", how="inner")
    out = sort_table_df(out, df, name)
    out["ticker"] = vp.ticker_list_to_ordered_map(out["ticker"])["ticker_out"]

    # 3) Generate LaTeX and write file
    latex = CarrWu2009_table_3_latex_v2(out, name)
    with open(f"figures/Analysis/{name}_table_3.tex", "w") as f:
        f.write(latex)

    return out


def CarrWu2009_table_3_latex_v2(table_df, name):
    """
    Generates a longtable with:
      - Line number
      - Panel A: (RV - SW) × 100 (Mean, Std. dev., Auto, Skew, Kurt, t)
      - Panel B: ln(RV / SW) (Mean, Std. dev., Auto, Skew, Kurt, t, SR)
    """
    # 1) Reorder & set up MultiIndex
    table_df = table_df[
        [
            "ticker",
            "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff",
            "Mean_ln",   "Std_ln",   "Auto_ln",   "Skew_ln",   "Kurt_ln",   "t_ln",  "SR_ann"
        ]
    ]
    # insert line numbers
    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))
    # rename columns to a two‐level header
    table_df.columns = pd.MultiIndex.from_tuples([
        ("",                           "No."),
        ("",                       "Ticker"),
        ("Panel A: (RV – SW) × 100",   "Mean"),
        ("Panel A: (RV – SW) × 100",   "Std. dev."),
        ("Panel A: (RV – SW) × 100",   "Auto"),
        ("Panel A: (RV – SW) × 100",   "Skew"),
        ("Panel A: (RV – SW) × 100",   "Kurt"),
        ("Panel A: (RV – SW) × 100",     "t"),
        ("Panel B: ln(RV / SW)",        "Mean"),
        ("Panel B: ln(RV / SW)",        "Std. dev."),
        ("Panel B: ln(RV / SW)",        "Auto"),
        ("Panel B: ln(RV / SW)",        "Skew"),
        ("Panel B: ln(RV / SW)",        "Kurt"),
        ("Panel B: ln(RV / SW)",        "t"),
        ("Panel B: ln(RV / SW)",        "SR")
    ])

    # 2) Export as longtable
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            rf"Summary statistics for (RV – SW) $\times$ 100 and ln(RV / SW) for the {name} dataset using monthly values. "
            "Each panel shows Mean, Std. dev., Auto(1), Skew, Kurt, and Newey West (1987) t-statistic; "
            "Panel B also reports an annualized Sharpe ratio (SR) of going short the variance swap."
        ),
        label=f"tab:analysis:Summary statistics diff and logratio ({name})",
        column_format='rlcccccc||ccccccc',
        multicolumn_format="c"
    )

    # 3) Inject cmidrule under the top header
    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            # after \toprule and the next header line
            lines.insert(i + 2, r'\cmidrule(lr){3-8}\cmidrule(lr){9-15}')
            break

    # 4) Add size commands & float barrier
    lines.insert(0, r'\scriptsize')
    lines.append(r'\normalsize')

    return "\n".join(lines)


def CarrWu2009_table_3_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: (RV - SW_0_30) x 100
      - Panel B: ln(RV / SW_0_30)
    Each panel displays: Mean, Std. dev., Auto, Skew, Kurt, and t.
    """
    # 1) Reorder & set up MultiIndex (as you already do)…
    table_df = table_df[
        [
            "ticker",
            "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff",
            "Mean_ln",   "Std_ln",   "Auto_ln",   "Skew_ln",   "Kurt_ln",   "t_ln", "SR_ann"
        ]
    ]
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "Ticker"),
        ("Panel A: (RV - SW) x 100", "Mean"),
        ("Panel A: (RV - SW) x 100", "Std. dev."),
        ("Panel A: (RV - SW) x 100", "Auto"),
        ("Panel A: (RV - SW) x 100", "Skew"),
        ("Panel A: (RV - SW) x 100", "Kurt"),
        ("Panel A: (RV - SW) x 100", "t"),
        ("Panel B: ln(RV / SW)", "Mean"),
        ("Panel B: ln(RV / SW)", "Std. dev."),
        ("Panel B: ln(RV / SW)", "Auto"),
        ("Panel B: ln(RV / SW)", "Skew"),
        ("Panel B: ln(RV / SW)", "Kurt"),
        ("Panel B: ln(RV / SW)", "t"),
        ("Panel B: ln(RV / SW)", "SR")
    ])

    # 2) Generate the LaTeX with a 1 cm gap and centered Panel A header
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        multicolumn_format="c",                  # center the multicolumns by default
        column_format='lcccccc||ccccccc'
    )

    # 3) Split into lines and drop pandas' automatic full-width \midrule
    lines = [L for L in latex.splitlines() if L.strip() != r'\midrule']


    # inject the cmidrule lines
    lines = latex.splitlines()
    # find where to insert (after the multicolumn header)
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            insert_pos = i + 2
            break
    lines.insert(insert_pos, r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}')

    return '\n'.join(lines)


import load_clean_lib

def vix_table(df):
    results = []
    df = df[df["ticker"] != "TLT"]

    for ticker, vol in load_clean_lib.ticker_to_vol.items():
        sw_col = f"SW_0_30"  # Adjust based on actual column names
        iv_col = vol
        df_ticker = df[df["ticker"] == ticker].copy()
        df_ticker[iv_col] = df_ticker[iv_col]**2

        # Filter rows where both SW and IV are present
        mask = df_ticker[sw_col].notna() & df_ticker[iv_col].notna()
        df_filtered = df_ticker.loc[mask]

        if df_filtered.empty:
            continue

        # Compute statistics
        sw_mean = df_filtered[sw_col].mean()
        iv_mean = df_filtered[iv_col].mean()
        epsilon = df_filtered[iv_col] - df_filtered[sw_col]
        epsilon_mean = epsilon.mean()
        epsilon_std = epsilon.std()
        epsilon_over_iv = (epsilon_mean / iv_mean) # Convert to percentage
        correlation = df_filtered[[sw_col, iv_col]].corr().iloc[0, 1]

        # Get start and end dates
        start_date = df_filtered['date'].min()
        end_date = df_filtered['date'].max()

        results.append({
            'ETF/Index': ticker,
            'Vol Index': vol,
            'Underlying': vp.ticker_list_to_ordered_map([ticker])["name"],
            'start_date': start_date,
            'end_date': end_date,
            'SW': sw_mean * 100,
            'IV': iv_mean * 100,
            'mean_epsilon': epsilon_mean * 100,
            'std_epsilon': epsilon_std * 100,
            'eps_correlation': correlation * 100,
            'epsilon_over_iv': epsilon_over_iv * 100,
            'corr_K_eps': np.corrcoef(df_filtered[iv_col], df_filtered["#K"]/epsilon)[0,1] * 100,
        })
    results = pd.DataFrame(results)

    # Compute averages for relevant columns
    avg_row = results[['SW', 'IV', 'mean_epsilon', 'std_epsilon', 'epsilon_over_iv', 'eps_correlation', 'corr_K_eps']].mean()
    avg_row['ETF/Index'] = 'Average'
    avg_row['Vol Index'] = ''
    avg_row['Underlying'] = ''
    avg_row['start_date'] = np.nan
    avg_row['end_date'] = np.nan
    results = pd.concat([results, pd.DataFrame([avg_row])], ignore_index=True)

    return results



def latex_vix_table(df):
    # 1) format dates as strings
    df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
    df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')

    # 2) rename columns to match your header (and inject LaTeX math)
    results_tex = df.rename(columns={
        'ETF/Index': 'Ticker',
        'Vol Index': 'Vol Index',
        'Underlying': 'Underlying',
        'start_date': 'Start date',
        'end_date': 'End date',
        'SW': 'SW',
        'IV': r'$\mathrm{IV}^{\mathrm{Cboe}}$',
        'mean_epsilon': r'mean $\varepsilon$',
        'epsilon_over_iv': r'mean $\varepsilon^*$',
        'std_epsilon': r'st. \ $\varepsilon$',
        'eps_correlation': r'$corr_{\text{SW}}^{\mathrm{IV}^{\mathrm{Cboe}}}$',
        'corr_K_eps': r'$corr_{\varepsilon^*}^{#K}$',
    })

    # 3) dump to LaTeX
    body = results_tex.to_latex(
        index=False,
        escape=False,
        column_format='lllll|lllllll',
        float_format="%.2f"
    )

    full_table = "\n".join([
        r"\begin{table}[ht]",
        r"  \centering",
        r"\adjustbox{max width=\textwidth}{",
        body,
        r"}",
        r"  \caption{ETFs/Indices with Cboe-Branded 30-Day Volatility Indices. Numbers are in percent. To save space we note the scaled percentage error as $\varepsilon^* = \displaystyle\frac{\varepsilon}{\mathrm{IV}^{\mathrm{Cboe}}}$.}",
        r"  \label{tab:Robustness analysis:summary_table_volatility_index_comparison}",
        r"\end{table}"
    ])

    # 4) Write to file
    output_dir = "figures/vix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vix_table.tex")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_table)




import os
def table_3_daily_EWMA(df, name, alpha):
    alpha = int(round(alpha*100))
    # ensure output dir
    os.makedirs("figures/Analysis", exist_ok=True)

    # 1) Prep the data
    df = df.dropna(subset=["CF_30_SW_day", f"r_30_SW_day .{alpha}"]).copy()

    # 2) in your compute_stats(), capture this NW std
    def compute_stats(series):
        n = len(series)
        mean_ = series.mean()
        std_ = series.std(ddof=1)
        se_ = std_ / np.sqrt(n)  # standard error of the mean
        auto_ = series.autocorr(lag=1)
        skew_ = series.skew()
        kurt_ = series.kurt()
        t_ = mean_/se_
        return pd.Series({
            "Mean": mean_,
            "Std": std_,
            "Auto": auto_,
            "Skew": skew_,
            "Kurt": kurt_,
            "t": t_,
        })

    # Panel A: (RV - SW_0_30) × 100
    df_CF = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["CF_30_SW_day"] * 100 * 252))
        .reset_index()
    )
    df_CF.columns = [
        "ticker",
        "Mean_CF", "Std_CF", "Auto_CF",
        "Skew_CF", "Kurt_CF", "t_CF"
    ]

    # Panel B: ln(RV / SW_0_30)
    df_r = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g[f"r_30_SW_day .{alpha}"]))
        .reset_index()
    )
    df_r.columns = ["ticker", "Mean_r", "Std_r", "Auto_r", "Skew_r", "Kurt_r", "t_r"]

    # Compute Sharpe ratios for Panel B
    df_r["SR_ann"] = -df_r["Mean_r"] / df_r["Std_r"] * np.sqrt(252)

    # Merge & sort
    out = pd.merge(df_CF, df_r, on="ticker", how="inner")
    out = sort_table_df(out, df, name)
    out["ticker"] = vp.ticker_list_to_ordered_map(out["ticker"])["ticker_out"]

    # 3) Generate LaTeX and write file
    latex = table_3_daily_EWMA_latex(out, name, alpha)
    with open(f"figures/Analysis/{name}_table_3_dailyEWMA_{alpha}.tex", "w") as f:
        f.write(latex)

    return out


def table_3_daily_EWMA_latex(table_df, name, alpha):
    """
    Generates a longtable with:
      - Line number
      - Panel A: (RV - SW) × 100 (Mean, Std. dev., Auto, Skew, Kurt, t)
      - Panel B: ln(RV / SW) (Mean, Std. dev., Auto, Skew, Kurt, t, SR)
    """
    # 1) Reorder & set up MultiIndex
    table_df = table_df[
        [
            "ticker",
            "Mean_CF", "Std_CF", "Auto_CF", "Skew_CF", "Kurt_CF", "t_CF",
            "Mean_r",   "Std_r",   "Auto_r",   "Skew_r",   "Kurt_r",   "t_r",  "SR_ann"
        ]
    ]
    # insert line numbers
    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))
    # rename columns to a two‐level header
    table_df.columns = pd.MultiIndex.from_tuples([
        ("",                           "No."),
        ("",                       "Ticker"),
        ("Panel A: (RV – SW) × 100",   "Mean"),
        ("Panel A: (RV – SW) × 100",   "Std. dev."),
        ("Panel A: (RV – SW) × 100",   "Auto"),
        ("Panel A: (RV – SW) × 100",   "Skew"),
        ("Panel A: (RV – SW) × 100",   "Kurt"),
        ("Panel A: (RV – SW) × 100",     "t"),
        ("Panel B: ln(RV / SW)",        "Mean"),
        ("Panel B: ln(RV / SW)",        "Std. dev."),
        ("Panel B: ln(RV / SW)",        "Auto"),
        ("Panel B: ln(RV / SW)",        "Skew"),
        ("Panel B: ln(RV / SW)",        "Kurt"),
        ("Panel B: ln(RV / SW)",        "t"),
        ("Panel B: ln(RV / SW)",        "SR")
    ])

    # 2) Export as longtable
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            rf"Summary statistics for daily cashflow of rolling a variance swap (CF $\times$ 100) and the return on the strategy (CF / $\overline{{SW}}$) for the {name} dataset. "
            f"We let $\overline{{SW}}$ be the exponentially weighted moving average swaprate with decay of {round(alpha)}\%. "
            "Each panel shows the annualized Mean, Std. dev., Auto(1), Skew, Kurt, and t-statistic; "
            "Panel B also reports an annualized Sharpe ratio (SR) of going short the rolling variance swap strategy."
        ),
        label=f"tab:analysis:Summary statistics CF and r ({name}, EWMA)",
        column_format='rlcccccc||ccccccc',
        multicolumn_format="c"
    )

    # 3) Inject cmidrule under the top header
    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            # after \toprule and the next header line
            lines.insert(i + 2, r'\cmidrule(lr){3-8}\cmidrule(lr){9-15}')
            break

    # 4) Add size commands & float barrier
    lines.insert(0, r'\scriptsize')
    lines.append(r'\normalsize')

    return "\n".join(lines)


import os
def table_3_daily(df, name, y_var):
    # ensure output dir
    os.makedirs("figures/Analysis", exist_ok=True)

    # 1) Prep the data
    df = df.dropna(subset=["CF_30_SW_day", "r_30_SW_day"]).copy()

    # 2) in your compute_stats(), capture this NW std
    def compute_stats(series):
        n = len(series)
        mean_ = series.mean()
        std_ = series.std(ddof=1)
        se_ = std_ / np.sqrt(n)  # standard error of the mean
        auto_ = series.autocorr(lag=1)
        skew_ = series.skew()
        kurt_ = series.kurt()
        t_ = mean_/se_
        return pd.Series({
            "Mean": mean_,
            "Std": std_,
            "Auto": auto_,
            "Skew": skew_,
            "Kurt": kurt_,
            "t": t_,
        })

    # Panel A: (RV - SW_0_30) × 100
    df_CF = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["CF_30_SW_day"] * 100))
        .reset_index()
    )
    df_CF.columns = [
        "ticker",
        "Mean_CF", "Std_CF", "Auto_CF",
        "Skew_CF", "Kurt_CF", "t_CF"
    ]

    # Panel B: ln(RV / SW_0_30)
    df_r = (
        df
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["r_30_SW_day"]))
        .reset_index()
    )
    df_r.columns = ["ticker", "Mean_r", "Std_r", "Auto_r", "Skew_r", "Kurt_r", "t_r"]

    # Compute Sharpe ratios for Panel B
    df_r["SR_ann"] = -df_r["Mean_r"] / df_r["Std_r"] * np.sqrt(252)

    # Merge & sort
    out = pd.merge(df_CF, df_r, on="ticker", how="inner")
    out = sort_table_df(out, df, name)
    out["ticker"] = vp.ticker_list_to_ordered_map(out["ticker"])["ticker_out"]

    # 3) Generate LaTeX and write file
    latex = table_3_daily_latex(out, name)
    with open(f"figures/Analysis/{name}_{y_var}_table_3_daily.tex", "w") as f:
        f.write(latex)

    return out


def table_3_daily_latex(table_df, name):
    """
    Generates a longtable with:
      - Line number
      - Panel A: (RV - SW) × 100 (Mean, Std. dev., Auto, Skew, Kurt, t)
      - Panel B: ln(RV / SW) (Mean, Std. dev., Auto, Skew, Kurt, t, SR)
    """
    # 1) Reorder & set up MultiIndex
    table_df = table_df[
        [
            "ticker",
            "Mean_CF", "Std_CF", "Auto_CF", "Skew_CF", "Kurt_CF", "t_CF",
            "Mean_r",   "Std_r",   "Auto_r",   "Skew_r",   "Kurt_r",   "t_r",  "SR_ann"
        ]
    ]
    # insert line numbers
    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))
    # rename columns to a two‐level header
    table_df.columns = pd.MultiIndex.from_tuples([
        ("",                           "No."),
        ("",                       "Ticker"),
        ("Panel A: (RV – SW) × 100",   "Mean"),
        ("Panel A: (RV – SW) × 100",   "Std. dev."),
        ("Panel A: (RV – SW) × 100",   "Auto"),
        ("Panel A: (RV – SW) × 100",   "Skew"),
        ("Panel A: (RV – SW) × 100",   "Kurt"),
        ("Panel A: (RV – SW) × 100",     "t"),
        ("Panel B: ln(RV / SW)",        "Mean"),
        ("Panel B: ln(RV / SW)",        "Std. dev."),
        ("Panel B: ln(RV / SW)",        "Auto"),
        ("Panel B: ln(RV / SW)",        "Skew"),
        ("Panel B: ln(RV / SW)",        "Kurt"),
        ("Panel B: ln(RV / SW)",        "t"),
        ("Panel B: ln(RV / SW)",        "SR")
    ])

    # 2) Export as longtable
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            rf"Summary statistics for..."
            #rf"Summary statistics for (RV – SW) $\times$ 100 and ln(RV / SW) for the {name} dataset. "
            #"Each panel shows Mean, Std. dev., Auto(1), Skew, Kurt, and t-statistic; "
            #"Panel B also reports an annualized Sharpe ratio (SR)."
        ),
        label=f"tab:analysis:Summary statistics CF and r ({name})",
        column_format='rlcccccc||ccccccc',
        multicolumn_format="c"
    )

    # 3) Inject cmidrule under the top header
    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            # after \toprule and the next header line
            lines.insert(i + 2, r'\cmidrule(lr){3-8}\cmidrule(lr){9-15}')
            break

    # 4) Add size commands & float barrier
    lines.insert(0, r'\scriptsize')
    lines.append(r'\normalsize')

    return "\n".join(lines)



import numpy as np
import pandas as pd
import statsmodels.api as sm
import volpy_func_lib as vp
from table_lib import sort_table_df

def capm_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    TRADING_DAYS = 252
    records = []
    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}

        # Panel A: SPX
        gA = g.dropna(subset=[y_var, "SPX"])
        if len(gA) > 0:
            X = sm.add_constant(gA["SPX"])
            if max_lags is None:
                fit = sm.OLS(gA[y_var], X).fit()
            else:
                fit = sm.OLS(gA[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
            a, b = fit.params
            ta, tb = fit.tvalues
            r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan

        ann_a = a * TRADING_DAYS

        rec.update({
            ("Panel A: SPX", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
            ("Panel A: SPX", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
            ("Panel A: SPX", r"\(R^2\)"): f"{r2:.3f}"
        })

        # Panel B: Mkt
        gB = g.dropna(subset=[y_var, "Mkt"])
        if len(gB) > 0:
            X = sm.add_constant(gB["Mkt"])
            if max_lags is None:
                fit = sm.OLS(gB[y_var], X).fit()
            else:
                fit = sm.OLS(gB[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
            a, b = fit.params
            ta, tb = fit.tvalues
            r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan

        ann_a = a * TRADING_DAYS

        rec.update({
            ("Panel B: Mkt", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
            ("Panel B: Mkt", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
            ("Panel B: Mkt", r"\(R^2\)"): f"{r2:.3f}"
        })

        records.append(rec)

    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)
    df = df.set_index("Ticker")
    # ensure proper MultiIndex on columns
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    latex = df.to_latex(
        index=True,
        escape=False,
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            "Explaining variance risk premiums with CAPM beta. "
            "Each panel reports the GMM estimates (and t-statistics in parentheses) of "
            "$r_{30\\_SW\\_day}=\\alpha+\\beta\\times \\mathrm{(SPX\\ or\\ Mkt)} + e$, "
            "and unadjusted $R^2$."
        ),
        label=f"tab:analysis:beta{name}",
        column_format="l" + "ccc" * 2,
        multicolumn_format="c",
        float_format="%.3f",
    ).splitlines()

    # insert cmidrules under the top header
    for i, L in enumerate(latex):
        if L.strip().startswith(r"\toprule"):
            # after \toprule and the next header line
            latex.insert(i + 2, r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
            break

    # add sizing & float barrier
    out = ["\scriptsize"] + latex + ["\\normalsize", r"\FloatBarrier"]
    return "\n".join(out), df


def save_capm_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = capm_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/{name}_{y_var}_table_CAPM.tex", "w") as f:
        f.write(tex)
    return df


# def ff3_factor_table_dly_mly(returns_df, name):
    # "Comparison of Fama-French factor models for monthly variance risk premiums (Panel A) "
    # "and daily returns (Panel B). Panel A estimates $\\ln RV_{t,\\tau}/SW_{t,\\tau}$ using HAC "
    # "standard errors with 21 lags. Panel B estimates $r_{t,\\tau}$ using OLS standard errors. "
    # "T-statistics shown in parentheses."



def ff3_factor_table(returns_df, name,
                          y_var="r_30_SW_day", max_lags=0):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    TRADING_DAYS = 252
    records = []
    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}
        # drop any row with a missing regressor or dependent
        g0 = g.dropna(subset=[y_var, "Mkt", "SMB", "HML"])
        if len(g0) > 0:
            X = sm.add_constant(g0[["Mkt", "SMB", "HML"]])
            if max_lags is None:
                fit = sm.OLS(g0[y_var], X).fit()
            else:
                fit = sm.OLS(g0[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

            params = fit.params
            tvals  = fit.tvalues
            r2     = fit.rsquared
            # pull out each coefficient + t-stat
            alpha, βm, s, h = params["const"]*TRADING_DAYS, params["Mkt"], params["SMB"], params["HML"]
            tα, tβm, ts, th = tvals["const"], tvals["Mkt"], tvals["SMB"], tvals["HML"]
        else:
            alpha = βm = s = h = tα = tβm = ts = th = r2 = np.nan

        rec.update({
            r"\(\alpha\)": f"{alpha:.3f}\n({tα:.3f})",
            r"\(ER^m\)"   : f"{βm:.3f}\n({tβm:.3f})",
            r"\mathrm{SMB}": f"{s:.3f}\n({ts:.3f})",
            r"\mathrm{HML}": f"{h:.3f}\n({th:.3f})",
            r"\(R^2\)"    : f"{r2:.3f}",
        })
        records.append(rec)

    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)
    df.set_index("Ticker")

    caption = (
        "Explaining variance risk premiums with Fama–French risk factors. "
        "Entries report the GMM estimates (and t-statistics in parentheses) of "
        r"$\ln RV_{t,\tau}/SW_{t,\tau} = \alpha + \beta\,ER^m_{t,\tau} + s\,SMB_{t,\tau} + h\,HML_{t,\tau} + e$, "
        "and unadjusted $R^2$."
    )

    latex = (
        df.to_latex(index=True, escape=False, longtable=True,
                    caption=caption,
                    label=f"tab:analysis:ff{name}",
                    column_format="lccccc",
                    float_format="%.3f")
          .splitlines()
    )

    # wrap in scriptsize + float barrier
    out = ["\\scriptsize"] + latex + ["\\normalsize", "\\FloatBarrier"]
    return "\n".join(out), df

def save_ff3_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = ff3_factor_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/{name}_{y_var}_table_ff3.tex", "w") as f:
        f.write(tex)
    return df


def save_ff3_table_dly_mly(returns_df, name):
    tex, df = ff3_factor_table_dly_mly(returns_df, name)
    with open(f"figures/Analysis/{name}_dly_mly_table_ff3.tex", "w") as f:
        f.write(tex)
    return df


def ff3_factor_table_dly_mly(returns_df, name):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    TRADING_DAYS = 252
    records = []
    panel_A = r"Panel A: $r_{\text{SW}}^{\text{month}}$"
    panel_B = r"Panel B: $r_{\text{SW}}^{\text{day}}$"

    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}
        X_vars = ["SPX", "SMB", "HML"]

        # Panel A: y_var="RV-SW 30", HAC with max_lags=21
        y_var_A = "RV-SW 30"
        gA = g.dropna(subset=[y_var_A] + X_vars)
        if len(gA) > 0:
            X = sm.add_constant(gA[X_vars])
            fitA = sm.OLS(gA[y_var_A], X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
            pA, tA = fitA.params, fitA.tvalues
            r2A = fitA.rsquared
            αA = pA["const"] * TRADING_DAYS
        else:
            pA = {k: np.nan for k in ["const"] + X_vars}
            tA = {k: np.nan for k in ["const"] + X_vars}
            r2A = αA = np.nan

        rec.update({
            (panel_A, r"\(\alpha\)"):    f"{αA:.3f}\n({tA['const']:.3f})",
            (panel_A, r"\(\beta^{SPX}\)"): f"{pA['SPX']:.3f}\n({tA['SPX']:.3f})",
            (panel_A, r"\(\beta^{SMB}\)"): f"{pA['SMB']:.3f}\n({tA['SMB']:.3f})",
            (panel_A, r"\(\beta^{HML}\)"): f"{pA['HML']:.3f}\n({tA['HML']:.3f})",
            (panel_A, r"\(R^2\)"):        f"{r2A:.3f}",
        })

        # Panel B: y_var="r_30_SW_day .20", plain OLS (max_lags=None)
        y_var_B = "r_30_SW_day .20"
        gB = g.dropna(subset=[y_var_B] + X_vars)
        if len(gB) > 0:
            X = sm.add_constant(gB[X_vars])
            fitB = sm.OLS(gB[y_var_B], X).fit()
            pB, tB = fitB.params, fitB.tvalues
            r2B = fitB.rsquared
            αB = pB["const"] * TRADING_DAYS
        else:
            pB = {k: np.nan for k in ["const"] + X_vars}
            tB = {k: np.nan for k in ["const"] + X_vars}
            r2B = αB = np.nan

        rec.update({
            (panel_B, r"\(\alpha\)"):    f"{αB:.3f}\n({tB['const']:.3f})",
            (panel_B, r"\(\beta^{SPX}\)"): f"{pB['SPX']:.3f}\n({tB['SPX']:.3f})",
            (panel_B, r"\(\beta^{SMB}\)"): f"{pB['SMB']:.3f}\n({tB['SMB']:.3f})",
            (panel_B, r"\(\beta^{HML}\)"): f"{pB['HML']:.3f}\n({tB['HML']:.3f})",
            (panel_B, r"\(R^2\)"):        f"{r2B:.3f}",
        })

        records.append(rec)

    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)
    df = df.set_index("Ticker")
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    latex = df.to_latex(
        index=True, escape=False, multicolumn=True, multirow=True, longtable=True,
        caption=(
            "Explaining variance risk premiums with Fama–French risk factors. "
            "Panel A reports estimates of $r_{\\text{SW}}^{\\text{month}} = RV_{-30,0}-SW_{-30,0} = \\alpha + \\beta Mkt + s SMB + h HML + e$ using HAC standard errors (max lags=21). "
            "Panel B reports estimates of $r_{\\text{SW}}^{\\text{day}} = RV_{-1,29} - SW_{-0,29} - \\text{RV}_{-1,0} = \\alpha + \\beta Mkt + s SMB + h HML + e$ using OLS. "
            "Each panel presents GMM estimates, t-statistics (parentheses), and unadjusted $R^2$. "
            "$r_{\\text{SW}}^{\\text{month}}$ and $r_{\\text{SW}}^{\\text{day}}$ are the returns on the strategies of holding the swap for the full month and a single day, respectively."
        ),
        label=f"tab:analysis:ff3_dly_mly_{name}",
        column_format="l" + "ccccc" * 2,
        multicolumn_format="c",
        float_format="%.3f"
    ).splitlines()

    # insert cmidrules
    for i, line in enumerate(latex):
        if line.strip().startswith(r"\toprule"):
            latex.insert(i + 2, r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}")
            break

    out = ["\\begingroup", "\\setlength{\\tabcolsep}{3pt}", "\\tiny"] + latex + ["\\endgroup"]
    return "\n".join(out), df




def ff5_factor_table(returns_df, name,
                         y_var="r_30_SW_day", max_lags=0):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    TRADING_DAYS = 252
    records = []
    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}

        # require all five factors + the dep var
        g0 = g.dropna(subset=[y_var, "Mkt", "SMB", "HML", "BAB", "UMD"])
        if len(g0):
            X = sm.add_constant(g0[["Mkt","SMB","HML","BAB","UMD"]])

            if max_lags is not None:
                fit = sm.OLS(g0[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
            else:
                fit = sm.OLS(g0[y_var], X).fit()

            ps = fit.params * np.array([TRADING_DAYS] + [1]*5)
            ts = fit.tvalues
            r2 = fit.rsquared

            α, βm, s, h, b_bab, m_umd = ps
            tα, tβm, tsmb, thml, tbab, tumd = (
                ts["const"], ts["Mkt"], ts["SMB"], ts["HML"], ts["BAB"], ts["UMD"]
            )
        else:
            α = βm = s = h = b_bab = m_umd = np.nan
            tα = tβm = tsmb = thml = tbab = tumd = r2 = np.nan

        rec.update({
            r"\(\alpha\)":      f"{α:.3f}\n({tα:.3f})",
            r"\(ER^m\)"   :      f"{βm:.3f}\n({tβm:.3f})",
            r"\mathrm{SMB}":     f"{s:.3f}\n({tsmb:.3f})",
            r"\mathrm{HML}":     f"{h:.3f}\n({thml:.3f})",
            r"\mathrm{BAB}":     f"{b_bab:.3f}\n({tbab:.3f})",
            r"\mathrm{UMD}":     f"{m_umd:.3f}\n({tumd:.3f})",
            r"\(R^2\)"    :      f"{r2:.3f}",
        })
        records.append(rec)

    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)

    df.set_index("Ticker")

    caption = (
        "Explaining variance risk premiums with Fama–French factors plus Betting-Against-Beta and Momentum. "
        "Entries report the GMM estimates (and t-statistics in parentheses) of "
        r"$\ln RV_{t,\tau}/SW_{t,\tau}="
        r"\alpha+\beta ER^m_{t,\tau}+s\,SMB_{t,\tau}+h\,HML_{t,\tau}"
        r"+b\,BAB_{t,\tau}+m\,UMD_{t,\tau}+e$, "
        "and unadjusted $R^2$."
    )

    latex = (
        df.to_latex(index=True, escape=False, longtable=True,
                    caption=caption,
                    label=f"tab:analysis:ff{name}_5f",
                    column_format="lccccccc",
                    float_format="%.3f")
          .splitlines()
    )

    return "\n".join(["\\scriptsize"] + latex + ["\\normalsize","\\FloatBarrier"]), df



def save_ff5_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = ff5_factor_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/{name}_{y_var}_table_ff5.tex", "w") as f:
        f.write(tex)
    return df


# def save_ff5_table_dly_mly(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
#     tex, df = ff5_factor_table_dly_mly(returns_df, name)
#     with open(f"figures/Analysis/{name}_{y_var}_table_ff5.tex", "w") as f:
#         f.write(tex)
#     return df




# def generate_latex_for_pairs(df):
#     """
#     Generate LaTeX code for figure pairs given a DataFrame with a 'ticker' column,
#     and save it to 'figures/vix/vix_figure_latex_code.tex'.
#     """
#     output_dir = "figures/vix"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "vix_figure_latex_code.tex")
#
#     tickers = list(df["ticker"].unique())
#     pairs = [tickers[i:i + 2] for i in range(0, len(tickers), 2)]
#
#     figs = [
#         {"suffix": " vs vix", "caption": "Level"},
#         {"suffix": " - vix", "caption": r"Difference ($\varepsilon = \text{IV}^{\text{Cboe}} - \text{SW}$)"},
#         {"suffix": " - vix scaled", "caption": r"Scaled difference ($\frac{\varepsilon}{\text{IV}^{\text{Cboe}}} = \frac{\text{IV}^{\text{Cboe}} - \text{SW}}{\text{IV}^{\text{Cboe}}}$)"}
#     ]
#
#     latex = []
#     for t1, t2 in pairs:
#         u1 = etf_to_underlying.get(t1, t1)
#         u2 = etf_to_underlying.get(t2, t2)
#         v1 = ticker_to_vol.get(t1, "")
#         v2 = ticker_to_vol.get(t2, "")
#
#         latex.append(f"\\subsection{{{u1} and {u2}}}")
#         for fig in figs:
#             latex.append("\\begin{figure}[!ht]")
#             latex.append("  \\hspace{0.4cm}")
#             latex.append("  \\makebox[\\textwidth][c]{%")
#             for idx, (t, u, v) in zip(['a', 'b'], [(t1, u1, v1), (t2, u2, v2)]):
#                 suffix = fig["suffix"]
#                 fname = f"ticker SW{suffix} ({t}).pdf"
#                 latex.append(f"    \\begin{{minipage}}[b]{{0.55\\linewidth}}")
#                 latex.append("      \\centering")
#                 latex.append(
#                     f"      \\includegraphics[width=\\linewidth]{{figures/vix/{fname}}}")
#                 latex.append("      \\captionsetup{skip=0pt}")
#                 latex.append(f"      \\caption*{{{idx}) {u} ({t}/{v})}}")
#                 latex.append("    \\end{minipage}")
#                 if idx == 'a':
#                     latex.append("    \\hfill")
#             latex.append("  }")
#             caption = fig["caption"]
#             label_key = caption.split()[0].lower()
#             latex.append(f"  \\caption{{{caption}}}")
#             latex.append(f"  \\label{{fig:{t1}_{t2}_{label_key}}}")
#             latex.append("\\end{figure}")
#             latex.append("")
#         latex.append("\\clearpage")
#
#     # Write the LaTeX to the file
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(latex))


































