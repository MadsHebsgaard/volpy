import pandas as pd
import numpy as np


def CarrWu2009_table_1(summary_dly_df, print_latex=False):
    # 1. Convert index to columns:
    df = summary_dly_df.reset_index()
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
    Creates a summary DataFrame of variance swap rate statistics (SW),
    grouped by Ticker.
    Columns calculated: Mean, Std. dev., Autocorrelation (lag=1), Skew, Kurtosis.
    """
    # Ensure Ticker is a regular column (if it's part of the index)
    df = summary_dly_df.reset_index()

    # Define a helper function to compute the desired statistics
    def stats_func(x):
        # Scale SW by 100 just for computing mean and std
        sw_scaled = x["SW"] * 100

        return pd.Series({
            "Mean": sw_scaled.mean(),
            "Std. dev.": sw_scaled.std(),
            "Auto": sw_scaled.autocorr(lag=1),  # dimensionless
            "Skew": sw_scaled.skew(),  # dimensionless
            "Kurt": sw_scaled.kurt()  # dimensionless
        })

    # Group by Ticker and apply the stats function
    out = df.groupby("Ticker").apply(stats_func).reset_index()

    if print_latex:
        CarrWu2009_table_2_latex(out)

    return out


def CarrWu2009_table_2_latex(table_df):
    """
    Prints a LaTeX-formatted table of the variance swap statistics DataFrame,
    with two decimals for numeric values.
    """
    # Use float_format to enforce 2 decimal places
    latex_code = table_df.to_latex(
        index=False,
        float_format="%.2f",
        columns=["Ticker", "Mean", "Std. dev.", "Auto", "Skew", "Kurt"],
        header=["Ticker", "Mean", "Std. dev.", "Auto", "Skew", "Kurt"]
    )
    print(latex_code)