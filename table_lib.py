import pandas as pd
import numpy as np




def CarrWu_order(df):
    ticker_order = [
        "SPX", "OEX", "DJX", "NDX", "QQQ", "MSFT", "INTC", "IBM", "AMER",
        "DELL", "CSCO", "GE", "CPQ", "YHOO", "SUNW", "MU", "MO", "AMZN",
        "ORCL", "LU", "TRV", "WCOM", "TYC", "AMAT", "QCOM", "TXN", "PFE",
        "MOT", "EMC", "HWP", "AMGN", "BRCM", "MER", "NOK", "CHL", "UNPH",
        "EBAY", "JNPR", "CIEN", "BRCD"
    ]
    # Create a mapping from ticker to its order
    order_dict = {ticker: idx for idx, ticker in enumerate(ticker_order)}

    # Map each ticker in the DataFrame to its order; tickers not in ticker_order become NaN.
    df['sort_order'] = df['Ticker'].map(order_dict)

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
    # Now 'Date' and 'Ticker' are columns in df.

    # 2. Filter out rows where SW is NaN:
    df_nonan = df[df["SW"].notna()]

    # 3. Group by Ticker and aggregate:
    table_df = (
        df_nonan
        .groupby("Ticker")
        .agg(
            Starting_date=("Date", "min"),
            Ending_date=("Date", "max"),
            N=("SW", "count"),
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
        columns=["Ticker", "Starting_date", "Ending_date", "N", "NK"],
        index=False,
        header=["Ticker", "Starting date", "Ending date", "N", "NK"],
        float_format="%.2f"
    )

    if print_latex:
        print(latex_code)
    return table_df



def CarrWu2009_table_2(summary_dly_df, print_latex=False):
    """
    Creates a summary DataFrame with two panels:
      - Panel A: Realized variance, RV×100
      - Panel B: Variance swap rate, SW×100
    Columns calculated in each panel: Mean, Std. dev., Auto, Skew, Kurt.
    """
    # Convert index to columns if Ticker is in the index
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
    df_rv = df[df["RV"].notna()].groupby("Ticker") \
        .apply(lambda g: compute_stats(g, "RV")) \
        .reset_index()

    # Rename columns so we can distinguish them before merging
    df_rv.columns = ["Ticker", "Mean_RV", "Std_RV", "Auto_RV", "Skew_RV", "Kurt_RV"]

    # --- Panel B: Variance swap rate (SW) ---
    df_sw = df[df["SW"].notna()].groupby("Ticker") \
        .apply(lambda g: compute_stats(g, "SW")) \
        .reset_index()

    df_sw.columns = ["Ticker", "Mean_SW", "Std_SW", "Auto_SW", "Skew_SW", "Kurt_SW"]

    # Merge realized variance and swap rate stats on Ticker
    # Use how="inner" so we only keep tickers that have both RV and SW
    out = pd.merge(df_rv, df_sw, on="Ticker", how="inner")

    out = CarrWu_order(out)


    # Optionally print LaTeX
    if print_latex:
        CarrWu2009_table_2_latex(out)

    return out

def CarrWu2009_table_2_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: Realized variance, RV×100
      - Panel B: Variance swap rate, SW×100
    Each panel has Mean, Std. dev., Auto, Skew, Kurt.
    """
    # Reorder columns
    table_df = table_df[
        [
            "Ticker",
            "Mean_RV", "Std_RV", "Auto_RV", "Skew_RV", "Kurt_RV",
            "Mean_SW", "Std_SW", "Auto_SW", "Skew_SW", "Kurt_SW"
        ]
    ]

    # Create multi-level column headers
    table_df.columns = pd.MultiIndex.from_tuples([
        ("", "Ticker"),
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
    """
    Creates a summary DataFrame with two panels:
      - Panel A: (RV - SW) x 100
      - Panel B: ln(RV / SW)

    For each panel, the following statistics are computed per Ticker:
      - Mean, Std. dev., Auto (lag-1 autocorrelation), Skew, Kurt,
        and t, where t is the Newey–West adjusted t-statistic (lag=30).

    Parameters:
      summary_dly_df : pd.DataFrame
          The original DataFrame. It must include columns "RV", "SW", and "Ticker".
      print_latex : bool, default False
          If True, the function will print the LaTeX code for the table.

    Returns:
      out : pd.DataFrame
          The merged summary statistics DataFrame.
    """
    # Ensure Ticker is a column
    df = summary_dly_df

    # Keep only rows where both RV and SW are present
    df_nonan = df.dropna(subset=["RV", "SW"]).copy()

    # Create new columns:
    df_nonan["diff"] = df_nonan["RV"] - df_nonan["SW"]  # (RV - SW)
    df_nonan["lnratio"] = np.log(df_nonan["RV"]) - np.log(df_nonan["SW"])  # ln(RV/SW)

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

    # Panel A: (RV - SW) x 100 (scale diff by 100)
    df_diff = (
        df_nonan
        .groupby("Ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["diff"] * 100))
        .reset_index()
    )
    df_diff.columns = [
        "Ticker",
        "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff"
    ]

    # Panel B: ln(RV / SW)
    df_ln = (
        df_nonan
        .groupby("Ticker", group_keys=False)
        .apply(lambda g: compute_stats(g["lnratio"]))
        .reset_index()
    )
    df_ln.columns = [
        "Ticker",
        "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln"
    ]

    # Merge the two panels on Ticker
    out = pd.merge(df_diff, df_ln, on="Ticker", how="inner")

    out = CarrWu_order(out)

    if print_latex:
        CarrWu2009_table_3_latex(out)

    return out


def CarrWu2009_table_3_latex(table_df):
    """
    Prints a LaTeX-formatted table with multi-level headers:
      - Panel A: (RV - SW) x 100
      - Panel B: ln(RV / SW)

    Each panel displays:
      Mean, Std. dev., Auto, Skew, Kurt, and t (Newey–West adjusted).
    """
    # Reorder columns for clarity
    table_df = table_df[
        [
            "Ticker",
            "Mean_diff", "Std_diff", "Auto_diff", "Skew_diff", "Kurt_diff", "t_diff",
            "Mean_ln", "Std_ln", "Auto_ln", "Skew_ln", "Kurt_ln", "t_ln"
        ]
    ]

    # Create multi-level column headers
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
    ])

    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True
    )
    print(latex_code)

