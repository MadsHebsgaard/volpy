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


def CarrWu2009_table_1(summary_dly_df, print_latex=False):
    # 1. Convert index to columns:
    # df = summary_dly_df.reset_index()
    df = summary_dly_df
    # Now 'date' and 'ticker' are columns in df.

    # 2. Filter out rows where SW_0_30 is NaN:
    df_nonan = df[df["SW_0_30"].notna()]

    # 3. Group by ticker and aggregate:
    table_df = (
        df_nonan
        .groupby("ticker")
        .agg(
            Starting_date=("date", "min"),
            Ending_date=("date", "max"),
            N=("SW_0_30", "count"),
            NK=("#K", "mean")  # adjust aggregation if needed
        )
        .reset_index()
    )

    # Convert the date columns to datetime
    table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"])
    table_df["Ending_date"] = pd.to_datetime(table_df["Ending_date"])

    # Format the dates as dd-mmm-yyyy
    table_df["Starting_date"] = table_df["Starting_date"].dt.strftime("%d-%b-%Y")
    table_df["Ending_date"] = table_df["Ending_date"].dt.strftime("%d-%b-%Y")

    # Set the numeric columns to their appropriate types:
    table_df["N"] = table_df["N"].astype(int)  # N as an integer
    table_df["NK"] = table_df["NK"].astype(float)  # NK remains float

    table_df = CarrWu_order(table_df)

    # Generate the LaTeX table. The float_format only applies to float columns,
    # so N will be printed without decimals.


    latex_code = table_df.to_latex(
        columns=["ticker", "Starting_date", "Ending_date", "N", "NK"],
        index=False,
        header=["ticker", "Starting date", "Ending date", "N", "NK"],
        float_format="%.2f"
    )

    if print_latex:
        print(latex_code)
    return table_df



def CarrWu2009_table_2(summary_dly_df, print_latex=False):
    """
    Creates a summary DataFrame with two panels:
      - Panel A: Realized variance, RV×100
      - Panel B: Variance swap rate, SW_0_30×100
    Columns calculated in each panel: Mean, Std. dev., Auto, Skew, Kurt.
    """
    # Convert index to columns if ticker is in the index
    df = summary_dly_df

    def compute_stats(sub_df, col):
        """Helper function to compute scaled stats for the given column."""
        scaled = sub_df[col] * 100  # scale by 100 for Mean/Std.
        return pd.Series({
            "Mean": scaled.mean(),
            "Std. dev.": scaled.std(),
            "Auto": scaled.autocorr(lag=1),  # dimensionless
            "Skew": scaled.skew(),
            "Kurt": scaled.kurt()
        })

    # --- Panel A: Realized variance (RV) ---
    df_rv = df[df["RV"].notna()].groupby("ticker") \
        .apply(lambda g: compute_stats(g, "RV")) \
        .reset_index()

    # Rename columns so we can distinguish them before merging
    df_rv.columns = ["ticker", "Mean_RV", "Std_RV", "Auto_RV", "Skew_RV", "Kurt_RV"]

    # --- Panel B: Variance swap rate (SW_0_30) ---
    df_sw = df[df["SW_0_30"].notna()].groupby("ticker") \
        .apply(lambda g: compute_stats(g, "SW_0_30")) \
        .reset_index()

    df_sw.columns = ["ticker", "Mean_SW", "Std_SW", "Auto_SW", "Skew_SW", "Kurt_SW"]

    # Merge realized variance and swap rate stats on ticker
    # Use how="inner" so we only keep tickers that have both RV and SW_0_30
    out = pd.merge(df_rv, df_sw, on="ticker", how="inner")

    out = CarrWu_order(out)


    # Optionally print LaTeX
    if print_latex:
        CarrWu2009_table_2_latex(out)

    return out

def CarrWu2009_table_2_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: Realized variance, RV×100
      - Panel B: Variance swap rate, SW_0_30×100
    Each panel has Mean, Std. dev., Auto, Skew, Kurt.
    """
    # Reorder columns
    table_df = table_df[
        [
            "ticker",
            "Mean_RV", "Std_RV", "Auto_RV", "Skew_RV", "Kurt_RV",
            "Mean_SW", "Std_SW", "Auto_SW", "Skew_SW", "Kurt_SW"
        ]
    ]

    # Create multi-level column headers
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "ticker"),
        ("Panel A: Realized variance, RV×100", "Mean"),
        ("Panel A: Realized variance, RV×100", "Std. dev."),
        ("Panel A: Realized variance, RV×100", "Auto"),
        ("Panel A: Realized variance, RV×100", "Skew"),
        ("Panel A: Realized variance, RV×100", "Kurt"),
        ("Panel B: Variance swap rate, SW×100", "Mean"),
        ("Panel B: Variance swap rate, SW×100", "Std. dev."),
        ("Panel B: Variance swap rate, SW×100", "Auto"),
        ("Panel B: Variance swap rate, SW×100", "Skew"),
        ("Panel B: Variance swap rate, SW×100", "Kurt"),
    ])


    # Generate LaTeX code
    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",  # 2 decimals for numeric columns
        multicolumn=True,  # allow multi-column formatting
        multirow=True  # allow multi-row formatting
    )
    print(latex_code)



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


def CarrWu2009_table_3(summary_dly_df, print_latex=False):
    # Ensure ticker is a column
    df = summary_dly_df

    # Keep only rows where both RV and SW_0_30 are present
    df_nonan = df.dropna(subset=["RV", "SW_0_30"]).copy()

    # Create new columns:
    df_nonan["diff"] = df_nonan["RV"] - df_nonan["SW_0_30"]  # (RV - SW_0_30)
    df_nonan["lnratio"] = np.log(df_nonan["RV"]) - np.log(df_nonan["SW_0_30"])  # ln(RV/SW_0_30)

    # Helper function to compute statistics including Newey-West adjusted t-stat
    def compute_stats(series):
        mean_ = series.mean()
        std_ = series.std()
        auto_ = series.autocorr(lag=1)
        skew_ = series.skew()
        kurt_ = series.kurt()
        t_ = newey_west_t_stat(series, lag=30)
        return pd.Series({
            "Mean": mean_,
            "Std. dev.": std_,
            "Auto": auto_,
            "Skew": skew_,
            "Kurt": kurt_,
            "t": t_
        })

    # Panel A: (RV - SW_0_30) x 100 (scale diff by 100)
    df_diff = (
        df_nonan
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["diff"] * 100))
        .reset_index()
    )
    df_diff.columns = [
        "ticker",
        "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff"
    ]

    # Panel B: ln(RV / SW_0_30)
    df_ln = (
        df_nonan
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["lnratio"]))
        .reset_index()
    )
    df_ln.columns = [
        "ticker",
        "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln"
    ]

    # Merge the two panels on ticker
    out = pd.merge(df_diff, df_ln, on="ticker", how="inner")

    out = CarrWu_order(out)

    if print_latex:
        CarrWu2009_table_3_latex(out)

    return out



def CarrWu2009_table_3_latex_stat(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: (stat) x 100

    Each panel displays:
      Mean, Std. dev., Auto, Skew, Kurt, and t (Newey–West adjusted).
    """
    # Reorder columns for clarity
    table_df = table_df[
        [
            "ticker",
            "Mean", "Std", "Auto", "Skew", "Kurt", "t",
        ]
    ]

    # Create multi-level column headers
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "ticker"),
        ("Panel A: (RV - SW) x 100", "Mean"),
        ("Panel A: (RV - SW) x 100", "Std. dev."),
        ("Panel A: (RV - SW) x 100", "Auto"),
        ("Panel A: (RV - SW) x 100", "Skew"),
        ("Panel A: (RV - SW) x 100", "Kurt"),
        ("Panel A: (RV - SW) x 100", "t"),
    ])

    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True
    )
    print(latex_code)



def CarrWu2009_table_3_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: (RV - SW_0_30) x 100
      - Panel B: ln(RV / SW_0_30)

    Each panel displays:
      Mean, Std. dev., Auto, Skew, Kurt, and t (Newey–West adjusted).
    """
    # Reorder columns for clarity
    table_df = table_df[
        [
            "ticker",
            "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff",
            "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln"
        ]
    ]

    # Create multi-level column headers
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "ticker"),
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
    ])

    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True
    )
    print(latex_code)





def CarrWu2009_table_3_choose_stat(summary_dly_df, print_latex=False, stat = "SW_0_30-RV_0_30"):
    # Ensure ticker is a column
    df = summary_dly_df

    # Keep only rows where both RV and SW_0_30 are present
    df_nonan = df.dropna(subset=[stat]).copy()

    # Helper function to compute statistics including Newey-West adjusted t-stat
    def compute_stats(series):
        mean_ = series.mean()
        std_ = series.std()
        auto_ = series.autocorr(lag=1)
        skew_ = series.skew()
        kurt_ = series.kurt()
        t_ = newey_west_t_stat(series, lag=30)
        return pd.Series({
            "Mean": mean_,
            "Std. dev.": std_,
            "Auto": auto_,
            "Skew": skew_,
            "Kurt": kurt_,
            "t": t_
        })

    # Panel A: (RV - SW_0_30) x 100 (scale diff by 100)
    df_stat = (
        df_nonan
        .groupby("ticker", group_keys=False)
        .apply(lambda g: compute_stats(g[stat] * 100))
        .reset_index()
    )
    df_stat.columns = [
        "ticker",
        "Mean", "Std", "Auto", "Skew", "Kurt", "t"
    ]

    out = CarrWu_order(df_stat)

    if print_latex:
        CarrWu2009_table_3_latex_stat(out)

    return out


def CarrWu2009_table_3_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: (RV - SW_0_30) x 100
      - Panel B: ln(RV / SW_0_30)

    Each panel displays:
      Mean, Std. dev., Auto, Skew, Kurt, and t (Newey–West adjusted).
    """
    # Reorder columns for clarity
    table_df = table_df[
        [
            "ticker",
            "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff",
            "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln"
        ]
    ]

    # Create multi-level column headers
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "ticker"),
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
    ])

    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True
    )
    print(latex_code)


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
        r"  \caption{ETFs/Indices with Cboe-Branded 30-Day Volatility Indices. Numbers are in percent. To save space we note the scaled error as $\varepsilon^* = \displaystyle\frac{\varepsilon}{\mathrm{IV}^{\mathrm{Cboe}}}$.}",
        r"  \label{tab:Robustness analysis:summary_table_volatility_index_comparison}",
        r"\end{table}"
    ])

    # 4) Write to file
    output_dir = "figures/vix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vix_table.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_table)




def generate_latex_for_pairs(df):
    """
    Generate LaTeX code for figure pairs given a DataFrame with a 'ticker' column,
    and save it to 'figures/vix/vix_figure_latex_code.txt'.
    """
    output_dir = "figures/vix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vix_figure_latex_code.txt")

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
                    f"      \\includegraphics[width=\\linewidth]{{text/Robustness Analysis/vix figures/{fname}}}")
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
