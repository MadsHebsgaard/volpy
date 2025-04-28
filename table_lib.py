import pandas as pd
import numpy as np
import os

from load_clean_lib import etf_to_underlying, ticker_to_vol


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



def table_dataset_list_strike_count(df, name):
    if name == "VIX":
        df = df[df["ticker"] != "TLT"]

    df_nonan = df[df["SW_0_30"].notna()]

    # add Q5_K aggregation
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

    # table_df = CarrWu_order(table_df)
    table_df = table_df.sort_values("NK", ascending=False).reset_index(drop=True)

    # format dates and types
    table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"]).dt.strftime("%d-%b-%Y")
    table_df["Ending_date"]   = pd.to_datetime(table_df["Ending_date"]).dt.strftime("%d-%b-%Y")
    table_df["N"]   = table_df["N"].astype(int)
    table_df["NK"]  = table_df["NK"].astype(float)
    table_df["Q1_K"] = table_df["Q1_K"].astype(float)
    table_df["Q5_K"] = table_df["Q5_K"].astype(float)
    table_df["Q10_K"] = table_df["Q10_K"].astype(float)

    # 1) get a raw LaTeX snippet (with its own environment)
    raw = table_df.to_latex(
        index=False,
        header=False,
        float_format="%.1f"
    )

    # 2) pull out only the lines between \midrule and \bottomrule
    lines = raw.splitlines()
    # find where the data rows start and end
    start = next(i for i, l in enumerate(lines) if l.strip() == r'\midrule') + 1
    end   = next(i for i, l in enumerate(lines) if l.strip() == r'\bottomrule')
    body  = "\n".join(lines[start:end])

    # 3) build your full table with the two‐row header
    full_table = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lllrrrrr}" "\n"
        r"\toprule" "\n"
        r"Ticker & Starting date & Ending date & Days & \multicolumn{4}{c}{Number of strikes} \\" "\n"
        r"\cmidrule(lr){5-8}" "\n"
        r" &  &  &  & Mean & 1\% & 5\% & 10\% \\" "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        rf"\caption{{List of stocks and stock indexes in the {name} sample.}}" "\n"
        rf"\label{{tab:data_summary_table_{name}}}" "\n"
        r"\end{table}"
    )

    # 4) write out
    out_path = f'figures/summary/data_summary_table_{name}.tex'
    with open(out_path, 'w') as f:
        f.write(full_table)


    return table_df



import pandas as pd

def CarrWu2009_table_2(summary_dly_df, name):
    df = summary_dly_df

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

    # 1) compute NK and merge
    nk = df.groupby("ticker")["#K"].mean().rename("NK")
    out = out.merge(nk, on="ticker")

    # 2) sort by NK desc, then drop the column
    out = out.sort_values("NK", ascending=False).reset_index(drop=True)
    out = out.drop(columns="NK")

    # 3) generate & save LaTeX
    latex = CarrWu2009_table_2_latex(out)   # now returns a string
    with open(f"figures/Analysis/{name}_table_2.tex","w") as f:
        f.write(latex)

    return out


def CarrWu2009_table_2_latex(table_df):
    # reorder & set up MultiIndex (same as before)
    table_df = table_df[
        ["ticker",
         "Mean_RV","Std_RV","Auto_RV","Skew_RV","Kurt_RV",
         "Mean_SW","Std_SW","Auto_SW","Skew_SW","Kurt_SW"]
    ]
    table_df.columns = pd.MultiIndex.from_tuples([
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
        column_format='lrrrrr||rrrrr',
        multicolumn_format = "c",
    )

    # inject the cmidrule lines
    lines = latex.splitlines()
    # find where to insert (after the multicolumn header)
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\toprule'):
            insert_pos = i + 2
            break
    lines.insert(insert_pos, r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}')

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


def CarrWu2009_table_3(summary_dly_df, name):
    # ensure output dir
    os.makedirs("figures/Analysis", exist_ok=True)

    # 1) Prep the data
    df = summary_dly_df
    df_nonan = df.dropna(subset=["RV", "SW_0_30"]).copy()
    df_nonan["diff"]    = df_nonan["RV"] - df_nonan["SW_0_30"]
    df_nonan["lnratio"] = np.log(df_nonan["RV"]) - np.log(df_nonan["SW_0_30"])

    # 1) add this helper right after newey_west_t_stat
    def newey_west_std(series, lag=30):
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
        df_nonan
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
        df_nonan
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["lnratio"]))
        .reset_index()
    )
    df_ln.columns = ["ticker", "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln", "NW_std"]

    # Compute Sharpe ratios for Panel B
    df_ln["SR_30d"] = -df_ln["Mean_ln"] / df_ln["NW_std"]
    df_ln["SR_ann"] = df_ln["SR_30d"] * np.sqrt(12)
    df_ln = df_ln.drop(columns=["NW_std"])


    # Merge & order
    out = pd.merge(df_diff, df_ln, on="ticker", how="inner")
    out = CarrWu_order(out)

    # 3) Generate LaTeX and write file
    latex = CarrWu2009_table_3_latex(out)
    with open(f"figures/Analysis/{name}_table_3.tex", "w") as f:
        f.write(latex)

    return out



import re

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
etf_to_underlying = load_clean_lib.etf_to_underlying
ticker_to_vol = load_clean_lib.ticker_to_vol

def vix_table(df):
    results = []
    df = df[df["ticker"] != "TLT"]

    for ticker, vol in ticker_to_vol.items():
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
            'Underlying': etf_to_underlying[ticker],
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




def generate_latex_for_pairs(df):
    """
    Generate LaTeX code for figure pairs given a DataFrame with a 'ticker' column,
    and save it to 'figures/vix/vix_figure_latex_code.tex'.
    """
    output_dir = "figures/vix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vix_figure_latex_code.tex")

    tickers = list(df["ticker"].unique())
    pairs = [tickers[i:i + 2] for i in range(0, len(tickers), 2)]

    figs = [
        {"suffix": " vs vix", "caption": "Level"},
        {"suffix": " - vix", "caption": r"Difference ($\varepsilon = \text{IV}^{\text{Cboe}} - \text{SW}$)"},
        {"suffix": " - vix scaled", "caption": r"Scaled difference ($\frac{\varepsilon}{\text{IV}^{\text{Cboe}}} = \frac{\text{IV}^{\text{Cboe}} - \text{SW}}{\text{IV}^{\text{Cboe}}}$)"}
    ]

    latex = []
    for t1, t2 in pairs:
        u1 = etf_to_underlying.get(t1, t1)
        u2 = etf_to_underlying.get(t2, t2)
        v1 = ticker_to_vol.get(t1, "")
        v2 = ticker_to_vol.get(t2, "")

        latex.append(f"\\subsection{{{u1} and {u2}}}")
        for fig in figs:
            latex.append("\\begin{figure}[!ht]")
            latex.append("  \\hspace{0.4cm}")
            latex.append("  \\makebox[\\textwidth][c]{%")
            for idx, (t, u, v) in zip(['a', 'b'], [(t1, u1, v1), (t2, u2, v2)]):
                suffix = fig["suffix"]
                fname = f"ticker SW{suffix} ({t}).pdf"
                latex.append(f"    \\begin{{minipage}}[b]{{0.55\\linewidth}}")
                latex.append("      \\centering")
                latex.append(
                    f"      \\includegraphics[width=\\linewidth]{{figures/vix/{fname}}}")
                latex.append("      \\captionsetup{skip=0pt}")
                latex.append(f"      \\caption*{{{idx}) {u} ({t}/{v})}}")
                latex.append("    \\end{minipage}")
                if idx == 'a':
                    latex.append("    \\hfill")
            latex.append("  }")
            caption = fig["caption"]
            label_key = caption.split()[0].lower()
            latex.append(f"  \\caption{{{caption}}}")
            latex.append(f"  \\label{{fig:{t1}_{t2}_{label_key}}}")
            latex.append("\\end{figure}")
            latex.append("")
        latex.append("\\clearpage")

    # Write the LaTeX to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
