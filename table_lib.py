import pandas as pd
import numpy as np
import os
import volpy_func_lib as vp
import global_settings
from load_clean_lib import ticker_to_vol
from volpy_func_lib import Cross_AM_tickers, ticker_list_to_ordered_map
import importlib
importlib.reload(global_settings)
from global_settings import days_type

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


# def table_dataset_list_strike_count_pages(df, name, width_scale=0.75):
#
#     df = df.dropna(subset=["RV", "SW_0_30"]).copy()
#
#     table_df = (
#         df
#         .groupby("ticker")
#         .agg(
#             Starting_date=("date", "min"),
#             Ending_date=("date", "max"),
#             N=("date", "count"),
#             NK=("#K", "mean"),
#             Q1_K=("#K", lambda x: x.quantile(0.01)),
#             Q5_K=("#K", lambda x: x.quantile(0.05)),
#             Q10_K=("#K", lambda x: x.quantile(0.10))
#         )
#         .reset_index()
#     )
#     table_df = sort_table_df(table_df, df, name)
#
#     table_df.insert(0, "No.", table_df.index + 1)
#
#     # Formatting
#     table_df["Starting_date"] = pd.to_datetime(table_df["Starting_date"]).dt.strftime("%d-%b-%Y")
#     table_df["Ending_date"] = pd.to_datetime(table_df["Ending_date"]).dt.strftime("%d-%b-%Y")
#     table_df["N"] = table_df["N"].astype(int)
#     table_df["NK"] = table_df["NK"].round(1)
#     table_df["Q1_K"] = table_df["Q1_K"].round(1)
#     table_df["Q5_K"] = table_df["Q5_K"].round(1)
#     table_df["Q10_K"] = table_df["Q10_K"].round(1)
#     table_df["Name"] = [name.replace("&", r"\&") for name in vp.ticker_list_to_ordered_map(table_df["ticker"])["name"]]
#     table_df["ticker"] = vp.ticker_list_to_ordered_map(table_df["ticker"])["ticker_out"]
#
#     # LaTeX generation
#     raw = table_df.to_latex(index=False, header=False, float_format="%.1f")
#     lines = raw.splitlines()
#     start = next(i for i, l in enumerate(lines) if l.strip() == r'\midrule') + 1
#     end = next(i for i, l in enumerate(lines) if l.strip() == r'\bottomrule')
#     body = "\n".join(lines[start:end])
#
#     num_rows = len(table_df)
#
#     # Font size mapping (approximate scaling)
#     font_sizes = {
#         0.6: r"\scriptsize",
#         0.7: r"\footnotesize",
#         0.75: r"\small",
#         0.9: r"\normalsize",
#         1.0: r"\normalsize"
#     }
#     font_size = font_sizes.get(width_scale, r"\small")
#     if num_rows > 50:
#         full_table = (
#             rf"{font_size}" + "\n"
#             r"\begin{longtable}{@{}rlrlrrrrrl@{}}" + "\n"
#             rf"\caption{{List of stocks and stock indexes in the {name} sample}} \\" + "\n"
#             r"\toprule" + "\n"
#             r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
#             r"\cmidrule(r){6-9}" + "\n"
#             r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
#             r"\midrule" + "\n"
#             r"\endfirsthead" + "\n\n"
#             r"\multicolumn{10}{c}{{\tablename\ \thetable{} -- Continued}} \\" + "\n"
#             r"\toprule" + "\n"
#             r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
#             r"\cmidrule(r){6-9}" + "\n"
#             r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
#             r"\midrule" + "\n"
#             r"\endhead" + "\n\n"
#             r"\midrule" + "\n"
#             r"\multicolumn{10}{r}{{Continued}} \\" + "\n"
#             r"\endfoot" + "\n\n"
#             r"\bottomrule" + "\n"
#             r"\endlastfoot" + "\n\n"
#             + body + "\n"
#             rf"\caption{{List of stocks and stock indexes in the {name} sample}}" + "\n"
#             rf"\label{{tab:data_summary_{name}}}" + "\n"
#             r"\end{longtable}" + "\n"
#             r"\normalsize"  # Reset font size
#         )
#     else:
#         full_table = (
#             r"\begin{table}[ht]" + "\n"
#             r"\centering" + "\n"
#             rf"{font_size}" + "\n"
#             r"\begin{tabular}{rlrlrrrrrl}" + "\n"
#             r"\toprule" + "\n"
#             r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{4}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
#             r"\cmidrule(r){6-9}" + "\n"
#             r" &  &  &  &  & Mean & 1\% & 5\% & 10\% & \\" + "\n"
#             r"\midrule" + "\n"
#             + body + "\n"
#             r"\bottomrule" + "\n"
#             r"\end{tabular}" + "\n"
#             rf"\caption{{List of stocks and stock indexes in the {name} sample}}" + "\n"
#             rf"\label{{tab:data_summary_{name}}}" + "\n"
#             r"\end{table}" + "\n"
#             r"\normalsize"
#         )
#
#     out_path = f'figures/summary/data_summary_table_{name}.tex'
#     with open(out_path, 'w') as f:
#         f.write(full_table)
#     return table_df
#
# sort_map = {
#     "VIX":      True,
#     "Cross-AM": True,
#     "OEX":      True,
#     "Liquid":   True,
#     "CarrWu":   True,
#     "DJX":      True,
# }


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
            Q01_K=("#K", lambda x: x.quantile(0.001)),
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
    table_df["Q01_K"] = table_df["Q01_K"].round(1)
    table_df["Q1_K"] = table_df["Q1_K"].round(1)
    table_df["Q5_K"] = table_df["Q5_K"].round(1)
    table_df["Q10_K"] = table_df["Q10_K"].round(1)
    table_df["Name"] = [name.replace("&", r"\&") for name in vp.ticker_list_to_ordered_map(table_df["ticker"])["name"]]
    table_df["ticker"] = vp.ticker_list_to_ordered_map(table_df["ticker"])["ticker_out"]

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
            r"\begin{longtable}{@{}rlrlrrrrrrl@{}}" + "\n"
            rf"\caption{{List of stocks and stock indexes in the {name} sample}} \\" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{5}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-10}" + "\n"
            r" &  &  &  &  & Mean & .1\% & 1\% & 5\% & 10\% & \\" + "\n"
            r"\midrule" + "\n"
            r"\endfirsthead" + "\n\n"
            r"\multicolumn{11}{c}{{\tablename\ \thetable{} -- Continued}} \\" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{5}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-10}" + "\n"
            r" &  &  &  &  & Mean & .1\% & 1\% & 5\% & 10\% & \\" + "\n"
            r"\midrule" + "\n"
            r"\endhead" + "\n\n"
            r"\midrule" + "\n"
            r"\multicolumn{11}{r}{{Continued}} \\" + "\n"
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
            r"\begin{tabular}{rlrlrrrrrrl}" + "\n"
            r"\toprule" + "\n"
            r"No. & Ticker & \multicolumn{1}{c}{Start Date} & \multicolumn{1}{c}{End Date} & \multicolumn{1}{c}{Days} & \multicolumn{5}{c}{Strike Count} & \multicolumn{1}{c}{Name} \\" + "\n"
            r"\cmidrule(r){6-10}" + "\n"
            r" &  &  &  &  & Mean & .1\% & 1\% & 5\% & 10\% & \\" + "\n"
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
    "VIX":      False,
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
        df = df[orig_cols]
    return df



import pandas as pd

def CarrWu2009_table_2(df, name, save_latex = True):
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
    if save_latex:
        latex = CarrWu2009_table_2_latex_v2(out, name)   # now returns a string
        with open(f"figures/Analysis/Profitability/{name}_table_2.tex", "w") as f:
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


def CarrWu2009_table_3(df, name, save_latex = True):
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

    if save_latex:
        # 3) Generate LaTeX and write file
        latex = CarrWu2009_table_3_latex_v2(out, name)
        with open(f"figures/Analysis/Profitability/{name}_table_3.tex", "w") as f:
            f.write(latex)

    return out


def CarrWu2009_table_3_latex_v2(table_df, name):
    """
    Generates a longtable with:
      - Line number
      - Panel A: (RV - SW) × 100 (Mean, Std. dev., Auto, Skew, Kurt, t)
      - Panel B: ln(RV / SW) (Mean, Std. dev., Auto, Skew, Kurt, t, SR)
    """

    panel_A = "Panel A: (RV - SW) × 100"
    panel_B = "Panel B: ln(RV / SW)"

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
        (panel_A,   "Mean"),
        (panel_A,   "Std. dev."),
        (panel_A,   "Auto"),
        (panel_A,   "Skew"),
        (panel_A,   "Kurt"),
        (panel_A,     "t"),
        (panel_B,        "Mean"),
        (panel_B,        "Std. dev."),
        (panel_B,        "Auto"),
        (panel_B,        "Skew"),
        (panel_B,        "Kurt"),
        (panel_B,        "t"),
        (panel_B,        "SR")
    ])

    # 2) Export as longtable
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            rf"Summary statistics for average monthly variance risk premium, (RV - SW) × 100, and average monthly log return, ln(RV / SW), for the {name} dataset using monthly values. "
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

    panel_A = "Panel A: (RV - SW) x 100"
    panel_B = "Panel B: ln(RV / SW)"

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
        (panel_A, "Mean"),
        (panel_A, "Std. dev."),
        (panel_A, "Auto"),
        (panel_A, "Skew"),
        (panel_A, "Kurt"),
        (panel_A, "t"),
        (panel_B, "Mean"),
        (panel_B, "Std. dev."),
        (panel_B, "Auto"),
        (panel_B, "Skew"),
        (panel_B, "Kurt"),
        (panel_B, "t"),
        (panel_B, "SR")
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

    for ticker, vol in load_clean_lib.ticker_to_vol.items():
        sw_col = f"SW_0_30"  # Adjust based on actual column names
        iv_col = vol
        df_ticker = df[df["ticker"] == ticker].copy()
        df_ticker[iv_col] = df_ticker[iv_col]**2
        # df_ticker[sw_col] = np.sqrt(df_ticker[sw_col])

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
            'ticker': ticker,
            'Vol Index': vol,
            'Underlying': vp.ticker_list_to_ordered_map([ticker])["name"].iloc[0].replace("&", r"\&"),
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
    avg_row['ticker'] = 'Average'
    avg_row['Vol Index'] = ''
    avg_row['Underlying'] = ''
    avg_row['start_date'] = ''
    avg_row['end_date'] = ''

    results = sort_table_df(results, df, "VIX")

    results = pd.concat([results, pd.DataFrame([avg_row])], ignore_index=True)
    results.rename(columns={"ticker": "ETF / Index"})

    return results



def latex_vix_table(df):

    from datetime import datetime
    # 1) format dates as strings
    df['start_date'] = df['start_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime)) else x
    )
    df['end_date'] = df['end_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime)) else x
    )

    # 2) rename columns to match your header (and inject LaTeX math)
    results_tex = df.rename(columns={
        'ETF/Index': 'Ticker',
        'Vol Index': 'Vol Index',
        'Underlying': 'Underlying',
        'start_date': 'Start date',
        'end_date': 'End date',
        'SW': r'SW',
        'IV': r'$\mathrm{V}^{\mathrm{Cboe}}$',
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
        rf"  \label{{tab:Robustness analysis:summary_table_volatility_index_comparison_{days_type()}days}}",
        r"\end{table}"
    ])

    # 4) Write to file
    output_dir = "figures/vix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"vix_table_{days_type()}days.tex")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_table)




import os
def table_3_daily_EWMA(df, name, alpha, save_latex = True):
    alpha = int(round(alpha*100))
    # ensure output dir
    os.makedirs("figures/Analysis", exist_ok=True)

    # 1) Prep the data
    df = df.dropna(subset=["CF_30_SW_day", f"r_30_SW_day .20"]).copy()

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
        .apply(lambda g: compute_stats(g["CF_30_SW_day"] * 100*21))
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
        .apply(lambda g: compute_stats(g[f"r_30_SW_day .20"] * 100))
        .reset_index()
    )
    df_r.columns = ["ticker", "Mean_r", "Std_r", "Auto_r", "Skew_r", "Kurt_r", "t_r"]

    # Compute Sharpe ratios for Panel B
    df_r["SR_ann"] = -df_r["Mean_r"] / df_r["Std_r"] * np.sqrt(252)

    # Merge & sort
    out = pd.merge(df_CF, df_r, on="ticker", how="inner")
    out = sort_table_df(out, df, name)
    out["ticker"] = vp.ticker_list_to_ordered_map(out["ticker"])["ticker_out"]

    if save_latex:
        # 3) Generate LaTeX and write file
        latex = table_3_daily_EWMA_latex(out, name, alpha)
        with open(f"figures/Analysis/Profitability/{name}_table_3_dailyEWMA_{alpha}.tex", "w") as f:
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

    panel_A = "Panel A: CF × 100"
    panel_B = r"Panel B: (CF / $\overline{\text{SW}}) \times 100$"

    table_df.columns = pd.MultiIndex.from_tuples([
        (panel_A,                           "No."),
        (panel_A,                       "Ticker"),
        (panel_A,   "Mean"),
        (panel_A,   "Std. dev."),
        (panel_A,   "Auto"),
        (panel_A,   "Skew"),
        (panel_A,   "Kurt"),
        (panel_A,     "t"),
        (panel_B,        "Mean"),
        (panel_B,        "Std. dev."),
        (panel_B,        "Auto"),
        (panel_B,        "Skew"),
        (panel_B,        "Kurt"),
        (panel_B,        "t"),
        (panel_B,        "SR")
    ])

    # 2) Export as longtable
    latex = table_df.to_latex(
        index=False,
        float_format="%.2f",
        multicolumn=True,
        multirow=True,
        longtable=True,
        caption=(
            rf"Summary statistics for daily cashflow of rolling a variance swap (CF $\times$ 100) and the cashflow on the strategy with consistent expected variance exposure (CF / $\overline{{\text{{SW}}}} \times 100$) for the {name} dataset. "
            rf"$\overline{{\text{{SW}}}}$ is the exponentially weighted moving average swaprate with trading day decay of {round(alpha)}\%. "
            "Each panel shows the Mean, Std. dev., Auto(1), Skew, Kurt, and t-statistic; "
            "Panel B also reports an annualized Sharpe ratio (SR) of going short the rolling variance swap strategy. "
            "There are 252 trading days per year."
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
    with open(f"figures/Analysis/Profitability/{name}_{y_var}_table_3_daily.tex", "w") as f:
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

    panel_A = "Panel A: CF × 100"
    panel_B = r"Panel B: (CF / $\overline{\text{SW}}) \times 100$"

    # insert line numbers
    table_df.insert(0, ("", "No."), range(1, len(table_df) + 1))
    # rename columns to a two‐level header
    table_df.columns = pd.MultiIndex.from_tuples([
        ("",                           "No."),
        ("",                            "Ticker"),
        (panel_A,   "Mean"),
        (panel_A,   "Std. dev."),
        (panel_A,   "Auto"),
        (panel_A,   "Skew"),
        (panel_A,   "Kurt"),
        (panel_A,     "t"),
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



import statsmodels.api as sm

def capm_table(returns_df, name, y_var="r_30_SW_day", max_lags=0):
    # --- same data prep as before ---
    if max_lags is not None:
        ann_faktor = 1
        returns_df = returns_df.copy()
        for col in ["SPX", "Mkt"]:
            returns_df[col] = np.log(returns_df[col] + 1)
            returns_df[col] = (
                returns_df[col]
                .rolling(window=max_lags)
                .sum()
                .shift(-max_lags)
            )
    else:
        ann_faktor = 252

    # build the records
    records = []
    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}

        # Panel A: SPX
        gA = g.dropna(subset=[y_var, "SPX"])
        if len(gA):
            X = sm.add_constant(gA["SPX"])
            fit = (sm.OLS(gA[y_var], X).fit()
                   if max_lags is None
                   else sm.OLS(gA[y_var], X)
                          .fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}))
            a, b = fit.params; ta, tb = fit.tvalues; r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan
        rec.update({
            ("Panel A: SPX", r"\(\alpha\)"): f"{a*ann_faktor:.3f}\n({ta:.3f})",
            ("Panel A: SPX", r"\(\beta\)"):  f"{b:.3f}\n({tb:.3f})",
            ("Panel A: SPX", r"\(R^2\)"):    f"{r2:.3f}",
        })

        # Panel B: Mkt
        gB = g.dropna(subset=[y_var, "Mkt"])
        if len(gB):
            X = sm.add_constant(gB["Mkt"])
            fit = (sm.OLS(gB[y_var], X).fit()
                   if max_lags is None
                   else sm.OLS(gB[y_var], X)
                          .fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}))
            a, b = fit.params; ta, tb = fit.tvalues; r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan
        rec.update({
            ("Panel B: Mkt", r"\(\alpha\)"): f"{a*ann_faktor:.3f}\n({ta:.3f})",
            ("Panel B: Mkt", r"\(\beta\)"):  f"{b:.3f}\n({tb:.3f})",
            ("Panel B: Mkt", r"\(R^2\)"):    f"{r2:.3f}",
        })

        records.append(rec)

    # assemble into DataFrame
    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)

    # turn the index into a column so we can MultiIndex it
    df = df.reset_index(drop=True)

    # insert the two left‐most MultiIndex columns
    df.insert(0, ("", "No."),     range(1, len(df) + 1))
    df.insert(1, ("", "Ticker"), df["Ticker"])

    # drop the old single‐level 'Ticker' column
    df = df.drop(columns=["Ticker"])

    # rebuild the MultiIndex from the tuple‐labels
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # now produce the LaTeX
    col_fmt = "l" + "c" + "ccc" * 2
    latex_lines = (
        df.to_latex(
            index=False,
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
            column_format=col_fmt,
            multicolumn_format="c",
            float_format="%.3f",
        )
        .splitlines()
    )

    # insert cmidrules underneath the top header
    for i, L in enumerate(latex_lines):
        if L.strip().startswith(r"\toprule"):
            # skip \toprule line itself + header row, then inject our cmidrules
            latex_lines.insert(i + 2, r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
            break

    # wrap in size & float barrier
    out = [r"\scriptsize"] + latex_lines + [r"\normalsize", r"\FloatBarrier"]
    return "\n".join(out), df




import numpy as np
import pandas as pd
import statsmodels.api as sm
import volpy_func_lib as vp
from table_lib import sort_table_df

# def  capm_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
#
#     if max_lags is not None:
#         ann_faktor = 1
#         returns_df = returns_df.copy()
#         for col in ["SPX", "Mkt"]:
#             returns_df[col] = np.log(returns_df[col]+1)
#             returns_df[col] = returns_df[col].rolling(window=max_lags).sum().shift(-max_lags)
#
#     else:
#         ann_faktor = 252
#
#     records = []
#     for ticker, g in returns_df.groupby("ticker"):
#         rec = {"Ticker": ticker}
#
#         # Panel A: SPX
#         gA = g.dropna(subset=[y_var, "SPX"])
#         if len(gA) > 0:
#             X = sm.add_constant(gA["SPX"])
#             if max_lags is None:
#                 fit = sm.OLS(gA[y_var], X).fit()
#             else:
#                 fit = sm.OLS(gA[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
#             a, b = fit.params
#             ta, tb = fit.tvalues
#             r2 = fit.rsquared
#         else:
#             a = b = ta = tb = r2 = np.nan
#
#         ann_a = a * ann_faktor
#
#         rec.update({
#             ("Panel A: SPX", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
#             ("Panel A: SPX", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
#             ("Panel A: SPX", r"\(R^2\)"): f"{r2:.3f}"
#         })
#
#         # Panel B: Mkt
#         gB = g.dropna(subset=[y_var, "Mkt"])
#         if len(gB) > 0:
#             X = sm.add_constant(gB["Mkt"])
#             if max_lags is None:
#                 fit = sm.OLS(gB[y_var], X).fit()
#             else:
#                 fit = sm.OLS(gB[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
#             a, b = fit.params
#             ta, tb = fit.tvalues
#             r2 = fit.rsquared
#         else:
#             a = b = ta = tb = r2 = np.nan
#
#         ann_a = a * ann_faktor
#
#         rec.update({
#             ("Panel B: Mkt", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
#             ("Panel B: Mkt", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
#             ("Panel B: Mkt", r"\(R^2\)"): f"{r2:.3f}"
#         })
#
#         records.append(rec)
#
#     df = pd.DataFrame(records)
#     df.rename(columns={'Ticker': 'ticker'}, inplace=True)
#     df = sort_table_df(df, returns_df, name)
#     df["ticker"] = vp.ticker_list_to_ordered_map(df["ticker"])["ticker"]
#     df.rename(columns={'ticker': 'Ticker'}, inplace=True)
#     df = df.set_index("Ticker")
#     # ensure proper MultiIndex on columns
#     df.columns = pd.MultiIndex.from_tuples(df.columns)
#
#     df.insert(0, "No.", range(1, len(df) + 1))
#     col_fmt = "l" + "c" + "ccc" * 2  # = "lc" + "cccccc" -> "lccccccc"
#
#     latex = df.to_latex(
#         index=True,
#         escape=False,
#         multicolumn=True,
#         multirow=True,
#         longtable=True,
#         caption=(
#             "Explaining variance risk premiums with CAPM beta. "
#             "Each panel reports the GMM estimates (and t-statistics in parentheses) of "
#             "$r_{30\\_SW\\_day}=\\alpha+\\beta\\times \\mathrm{(SPX\\ or\\ Mkt)} + e$, "
#             "and unadjusted $R^2$."
#         ),
#         label=f"tab:analysis:beta{name}",
#         column_format = col_fmt,
#         multicolumn_format="c",
#         float_format="%.3f",
#     ).splitlines()
#
#     # insert cmidrules under the top header
#     for i, L in enumerate(latex):
#         if L.strip().startswith(r"\toprule"):
#             # after \toprule and the next header line
#             latex.insert(i + 2, r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
#             break
#
#     # add sizing & float barrier
#     out = ["\scriptsize"] + latex + ["\\normalsize", r"\FloatBarrier"]
#     return "\n".join(out), df


def save_capm_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = capm_table(returns_df, name, y_var, max_lags = max_lags)
    with open(fr"figures/Analysis/Factor models/{name}_{y_var.replace("/", "_")}_f1.tex", "w") as f:
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
    # df.set_index("Ticker")

    caption = (
        "Explaining variance risk premiums with Fama–French risk factors. "
        "Entries report the GMM estimates (and t-statistics in parentheses) of "
        r"$\ln RV_{t,\tau}/SW_{t,\tau} = \alpha + \beta\,ER^m_{t,\tau} + s\,SMB_{t,\tau} + h\,HML_{t,\tau} + e$, "
        "and unadjusted $R^2$."
    )

    df.insert(0, "No.", range(1, len(df) + 1))
    latex = (
        df.to_latex(index=False, escape=False, longtable=True,
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
    with open(f"figures/Analysis/Factor models/{name}_{y_var}_table_ff3.tex", "w") as f:
        f.write(tex)
    return df


def save_ff3_table_dly_mly(returns_df, name):
    tex, df = ff3_factor_table_dly_mly(returns_df, name)
    with open(f"figures/Analysis/Factor models/{name}_dly_mly_table_ff3.tex", "w") as f:
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


    for col in ["Mkt", "SPX", "SMB", "HML"]:
        returns_df[f"{col} lag"] = np.log(returns_df[f"{col}"]+1)
        returns_df[f"{col} lag"] = returns_df[f"{col} lag"].rolling(window=21).sum().shift(-21)

    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}
        X_vars_lag = ["SPX lag", "SMB lag", "HML lag"]
        X_vars = ["SPX", "SMB", "HML"]

        # Panel A: y_var="RV-SW 30", HAC with max_lags=21
        y_var_A = "ln RV/SW 30"
        gA = g.dropna(subset=[y_var_A] + X_vars_lag)
        if len(gA) > 0:
            X = sm.add_constant(gA[X_vars_lag])
            fitA = sm.OLS(gA[y_var_A], X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
            pA, tA = fitA.params, fitA.tvalues
            r2A = fitA.rsquared
            αA = pA["const"]
        else:
            pA = {k: np.nan for k in ["const"] + X_vars_lag}
            tA = {k: np.nan for k in ["const"] + X_vars_lag}
            r2A = αA = np.nan

        rec.update({
            (panel_A, r"\(\alpha\)"):    f"{αA:.3f}\n({tA['const']:.3f})",
            (panel_A, r"\(\beta^{SPX}\)"): f"{pA['SPX lag']:.3f}\n({tA['SPX lag']:.3f})",
            (panel_A, r"\(\beta^{SMB}\)"): f"{pA['SMB lag']:.3f}\n({tA['SMB lag']:.3f})",
            (panel_A, r"\(\beta^{HML}\)"): f"{pA['HML lag']:.3f}\n({tA['HML lag']:.3f})",
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
            r"Explaining variance risk premiums with Fama–French risk factors. "
            r"Panel A reports estimates of $\ln r_{\text{SW}}^{\text{month}} = \ln RV_{-30,0} - \ln SW_{-30,0} = \alpha + \beta\,Mkt + s\,SMB + h\,HML + e$ using HAC standard errors (max lags=21) and log returns. "
            r"Panel B reports estimates of $r_{\text{SW}}^{\text{day}} = RV_{-1,29} - SW_{-0,29} - \text{RV}_{-1,0} = \alpha + \beta\,Mkt + s\,SMB + h\,HML + e$ using OLS and arithmetic returns. "
            r"Each panel presents GMM estimates, t-statistics (parentheses), and unadjusted $R^2$. "
            r"$r_{\text{SW}}^{\text{month}}$ and $r_{\text{SW}}^{\text{day}}$ are the returns on the strategies of holding the swap for the full month and a single day, respectively."
        ),
        label=f"tab:analysis:ff3_dly_mly_{name}",
            column_format="l" + "ccccc" * 2,
            multicolumn_format="c",
            float_format="%.3f"
    ).splitlines()


    # — now pull “Ticker” into the α–β header row —
    for i, L in enumerate(latex):
        if L.strip().startswith("& \\(\\alpha\\)"):
            # replace leading & with “Ticker &”
            latex[i] = L.replace("&", "Ticker &", 1)
            break

    # drop the leftover “Ticker &” row that has only blanks
    latex = [
        L for L in latex
        if not (L.strip().startswith("Ticker &") and "\\(" not in L)
    ]

    # insert your cmidrules
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

    if max_lags is not None:
        ann_faktor = 1
        returns_df = returns_df.copy()
        for col in ["Mkt", "SMB", "HML", "BAB", "UMD"]:
            returns_df[col] = np.log(returns_df[col]+1)
            returns_df[col] = returns_df[col].rolling(window=max_lags).sum().shift(-max_lags)
    else:
        ann_faktor = 252

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

            ps = fit.params * np.array([ann_faktor] + [1]*5)
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

    df.insert(0, "No.", range(1, len(df) + 1))
    latex = (
        df.to_latex(index=False, escape=False, longtable=True,
                    caption=caption,
                    label=f"tab:analysis:ff{name}_5f",
                    column_format="lcccccccc",
                    float_format="%.3f")
          .splitlines()
    )

    return "\n".join(["\\scriptsize"] + latex + ["\\normalsize"]), df



def save_ff5_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = ff5_factor_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/Factor models/{name}_{y_var.replace("/", "_")}_table_ff5.tex", "w") as f:
        f.write(tex)
    return df


# def save_ff5_table_dly_mly(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
#     tex, df = ff5_factor_table_dly_mly(returns_df, name)
#     with open(f"figures/Analysis/{name}_{y_var}_table_ff5.tex", "w") as f:
#         f.write(tex)
#     return df




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
        {"suffix": " - vix", "caption": r"Difference ($\varepsilon = \text{V}^{\text{Cboe}} - \text{SW}$)"},
        {"suffix": " - vix scaled", "caption": r"Scaled difference ($\frac{\varepsilon}{\text{V}^{\text{Cboe}}} = \frac{\text{V}^{\text{Cboe}} - \text{SW}}{\text{V}^{\text{Cboe}}}$)"}
    ]

    latex = []
    for t1, t2 in pairs:
        u1 = ticker_list_to_ordered_map([t1])["name"].loc[0]
        u2 = ticker_list_to_ordered_map([t2])["name"].loc[0]
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



def df_to_latex(df, caption, label, float_fmt: str = "{:.2f}") -> str:
    """
    Render a pandas DataFrame as a LaTeX table with booktabs-style rules,
    without requiring pandas' `booktabs` argument.

    Only returns the table (wraps in table environment if caption/label given).
    """
    df = df.copy()

    # get a plain LaTeX table (with \hline)
    latex = df.to_latex(
        index=False,
        index_names=True,
        header=True,
        float_format=lambda x: float_fmt.format(x),
        column_format="l" + "r" * df.shape[1],
        escape=False
    )

    # replace the leading "\hline\n\hline\n" with "\toprule\n"
    latex = re.sub(r'^(\\hline\n\\hline\n)', r'\\toprule\n', latex)

    # replace the next single "\hline\n" (header/data separator) with "\midrule\n"
    latex = re.sub(r'\\hline\n', r'\\midrule\n', latex, count=1)

    # replace the final "\hline\n" before \end{tabular} with "\bottomrule\n"
    last = latex.rfind(r'\hline\n')
    if last != -1:
        latex = latex[:last] + r'\bottomrule\n' + latex[last + len(r'\hline\n'):]

    # optionally wrap in a table environment
    parts = ["\\begin{table}[htbp]", "\\centering"]
    parts.append(f"\\caption{{{caption}}}")
    parts.append(f"\\label{{{label}}}")
    parts.append(latex)
    parts.append("\\end{table}")
    return "\n".join(parts)

def performance_factor_table(df, save_file_path):

    latex_code = df_to_latex(
        df,
        caption="Performance measures for the daily variance‐swap rolling strategy and the daily rolling straddle strategy, compared to the daily returns of the equity factors: the Fama–French five factors, momentum (UMD), and betting‐against‐beta (BAB).",
        label="tab:perf_by_strategy",
        float_fmt="{:.2f}"
    )
    # print(latex_code)

    with open(save_file_path, "w") as f:
        f.write(latex_code)
    return


# def df_to_latex(df,
#                 label: str = None,
#                 float_fmt: str = "{:.2f}") -> str:
#     """
#     Render a pandas DataFrame as a LaTeX table with booktabs-style rules,
#     without requiring pandas' `booktabs` argument.
#
#     Only returns the table (wraps in table environment if caption/label given).
#     """
#     df = df.copy()
#
#     # caption = "Performance measures for the daily variance‐swap rolling strategy and the daily rolling straddle strategy, compared to the daily returns of the equity factors: the Fama–French five factors, momentum (UMD), and betting‐against‐beta (BAB)."
#     caption = None
#
#     # get a plain LaTeX table (with \hline)
#     latex = df.to_latex(
#         index=False,
#         index_names=True,
#         header=True,
#         float_format=lambda x: float_fmt.format(x),
#         column_format="l" + "r" * df.shape[1],
#         escape=False
#     )
#
#     # replace the leading "\hline\n\hline\n" with "\toprule\n"
#     latex = re.sub(r'^(\\hline\n\\hline\n)', r'\\toprule\n', latex)
#
#     # replace the next single "\hline\n" (header/data separator) with "\midrule\n"
#     latex = re.sub(r'\\hline\n', r'\\midrule\n', latex, count=1)
#
#     # replace the final "\hline\n" before \end{tabular} with "\bottomrule\n"
#     last = latex.rfind(r'\hline\n')
#     if last != -1:
#         latex = latex[:last] + r'\bottomrule\n' + latex[last + len(r'\hline\n'):]
#
#     # optionally wrap in a table environment
#     if caption or label:
#         parts = ["\\begin{table}[htbp]", "\\centering"]
#         if caption:
#             parts.append(f"\\caption{{{caption}}}")
#         if label:
#             parts.append(f"\\label{{{label}}}")
#         parts.append(latex)
#         parts.append("\\end{table}")
#         return "\n".join(parts)
#
#
#
#     latex_code = df_to_latex(
#         df,
#         caption="sdfs",
#         label="tab:perf_by_strategy",
#         float_fmt="{:.3f}"
#     )
#
#
#     with open(f"figures/Analysis/Implementability/performance_relative_to_factors2.tex", "w") as f:
#         f.write(latex_code)
#
#     return








import pandas as pd
import re

#
# def df_to_panel_latex(df,
#                       captions: list[str],
#                       labels:   list[str],
#                       float_fmt: str = "{:.2f}") -> str:
#     """
#     Produce two stacked LaTeX tables (Panel A and Panel B),
#     with multicolumn groups:
#       • Panel A: Option Strategy + 5 Moments + 3 Risk Measures
#       • Panel B: Option Strategy + 6 Ratios
#     """
#
#     # 1) rename first column to "Option Strategy"
#     df_fmt = df.copy()
#     first = df_fmt.columns[0]
#     df_fmt = df_fmt.rename(columns={ first: "Option Strategy" })
#
#     # 2) format numeric columns only
#     for col in df_fmt.columns[1:]:
#         if pd.api.types.is_numeric_dtype(df_fmt[col]):
#             df_fmt[col] = df_fmt[col].map(lambda x: float_fmt.format(x))
#
#     # 3) split into panels
#     n_moments = 5
#     n_risk    = 3
#     # Panel A: first col + moments + risk
#     cols_a = df_fmt.columns[: 1 + n_moments + n_risk ]
#     # Panel B: first col + remaining ratios
#     cols_b = ["Option Strategy"] + list(df_fmt.columns[1 + n_moments + n_risk :])
#
#     df_a = df_fmt[cols_a]
#     df_b = df_fmt[cols_b]
#
#     def make_panel(df_panel, caption, label, groups):
#         # render basic LaTeX tabular
#         raw = df_panel.to_latex(
#             index=False,
#             header=True,
#             escape=False,
#             column_format="l" + "r"*(df_panel.shape[1]-1)
#         )
#         # apply booktabs replacements
#         raw = re.sub(r'^(\\hline\n\\hline\n)', r'\\toprule\n', raw)
#         raw = re.sub(r'\\hline\n',            r'\\midrule\n', raw, count=1)
#         raw = re.sub(r'\\hline\n(?=\\end\{tabular\})',
#                      r'\\bottomrule\n',    raw)
#
#         lines = raw.splitlines()
#         # find where \toprule is
#         top_i = next(i for i,l in enumerate(lines) if l.strip() == r"\toprule")
#         # build group header row
#         grp_cells = ["Option Strategy (blank not needed)"] if False else [""]
#         grp_cells += [f"\\multicolumn{{{span}}}{{c}}{{{name}}}" for name, span in groups]
#         group_line = " & ".join(grp_cells) + r" \\\n"
#         # build cmidrule line(s)
#         cm = []
#         start = 2
#         for name, span in groups:
#             end = start + span - 1
#             cm.append(f"\\cmidrule(lr){{{start:d}-{end:d}}}")
#             start = end + 1
#         cm_line = " ".join(cm)
#
#         # insert group header and cmidrule right after top rule
#         lines.insert(top_i+1, group_line)
#         lines.insert(top_i+2, cm_line)
#
#         # wrap in table environment
#         out = ["\\begin{table}[htbp]",
#                "\\centering",
#                f"\\caption{{{caption}}}",
#                f"\\label{{{label}}}"] + lines + ["\\end{table}"]
#         return "\n".join(out)
#
#     # define groups
#     groups_a = [("Moments", n_moments),
#                 ("Risk Measures", n_risk)]
#     groups_b = [("Ratios", df_b.shape[1]-1)]
#
#     # render panels
#     panel_a = make_panel(df_a, captions[0], labels[0], groups_a)
#     panel_b = make_panel(df_b, captions[1], labels[1], groups_b)
#
#     return panel_a + "\n\n" + panel_b
#


import pandas as pd
import re

def df_to_panel_latex(df,
                      caption: str,
                      label:   str,
                      float_fmt: str = "{:.2f}") -> str:
    """
    Produce a single LaTeX table environment containing two panels:
      - Panel A: Option Strategy + 5 Moments + 3 Risk Measures
      - Panel B: Option Strategy + remaining Ratios
    Both panels share the same table number.
    """
    # rename first column
    df_fmt = df.copy()
    first = df_fmt.columns[0]
    df_fmt = df_fmt.rename(columns={first: "Option Strategy"})

    # format numeric columns
    for col in df_fmt.columns[1:]:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda x: float_fmt.format(x))

    # define splits
    n_moments = 5
    n_risk = 3
    cols_a = df_fmt.columns[:1 + n_moments + n_risk]
    cols_b = ["Option Strategy"] + list(df_fmt.columns[1 + n_moments + n_risk:])
    df_a = df_fmt[cols_a]
    df_b = df_fmt[cols_b]

    def build_panel(df_panel, groups):
        # build LaTeX tabular manually
        col_names = list(df_panel.columns)
        col_fmt = 'l' + 'r'*(len(col_names)-1)
        lines = []
        lines.append(fr"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")
        # group header row
        grp_cells = [""] + [fr"\multicolumn{{{span}}}{{c}}{{{name}}}" for name, span in groups]
        lines.append(" & ".join(grp_cells) + r" \\")
        # cmidrules
        start = 2
        cm_rules = []
        for _, span in groups:
            end = start + span - 1
            cm_rules.append(fr"\cmidrule(lr){{{start}-{end}}}")
            start = end + 1
        lines.append(" ".join(cm_rules))
        # column names row
        lines.append(" & ".join(col_names) + r" \\")
        lines.append(r"\midrule")
        # data rows
        for _, row in df_panel.iterrows():
            entries = [str(row[col]) for col in col_names]
            lines.append(" & ".join(entries) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    # define groups
    groups_a = [("Moments", n_moments), ("Risk Measures", n_risk)]
    groups_b = [("Ratios", len(cols_b)-1)]

    # build panels
    tab_a = build_panel(df_a, groups_a)
    tab_b = build_panel(df_b, groups_b)

    # assemble final table
    parts = [
        r"\begin{table}[htbp]",
        r"\centering",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        r"\textbf{Panel A: Moments and Risk Measures}\\",
        r"\vspace{1em}\\",
        tab_a,
        r"\vspace{0.5cm}\\",
        r"\textbf{Panel B: Ratios}\\",
        r"\vspace{1em}\\",
        tab_b,
        r"\end{table}"
    ]

    return "\n".join(parts)



def df_to_panel_latex(df, variable_name,
                      caption: str,
                      label:   str,
                      float_fmt: str = "{:.2f}") -> str:
    """
    Produce a single LaTeX table environment containing two panels:
      - Panel A: variable_name + 5 Moments + 3 Risk Measures
      - Panel B: variable_name + remaining Ratios
    Both panels share the same table number and are scaled together.
    """
    # rename first column
    df_fmt = df.copy()
    first = df_fmt.columns[0]
    df_fmt = df_fmt.rename(columns={first: variable_name})

    # format numeric columns
    for col in df_fmt.columns[1:]:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda x: float_fmt.format(x))

    # define splits
    n_moments = 5
    n_risk = 3
    cols_a = df_fmt.columns[:1 + n_moments + n_risk]
    cols_b = [variable_name] + list(df_fmt.columns[1 + n_moments + n_risk:])
    df_a = df_fmt[cols_a]
    df_b = df_fmt[cols_b]

    def build_panel(df_panel, groups):
        # build LaTeX tabular manually
        col_names = list(df_panel.columns)
        col_fmt = 'l' + 'r'*(len(col_names)-1)
        lines = []
        lines.append(fr"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")
        # group header row
        grp_cells = [""] + [fr"\multicolumn{{{span}}}{{c}}{{{name}}}" for name, span in groups]
        lines.append(" & ".join(grp_cells) + r" \\")
        # cmidrules
        start = 2
        cm_rules = []
        for _, span in groups:
            end = start + span - 1
            cm_rules.append(fr"\cmidrule(lr){{{start}-{end}}}")
            start = end + 1
        lines.append(" ".join(cm_rules))
        # column names row
        lines.append(" & ".join(col_names) + r" \\")
        lines.append(r"\midrule")
        # data rows
        for _, row in df_panel.iterrows():
            entries = [str(row[col]) for col in col_names]
            lines.append(" & ".join(entries) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    # define groups
    groups_a = [("Moments", n_moments), ("Risk Measures", n_risk)]
    groups_b = [("Ratios", len(cols_b)-1)]

    # build panels
    tab_a = build_panel(df_a, groups_a)
    tab_b = build_panel(df_b, groups_b)

    # pack both panels into one box so they scale equally
    combined = "\n".join([
        r"\resizebox{\textwidth}{!}{%",
        r"  \begin{minipage}{1.2\textwidth}",
        r"    \centering",
        r"    \textbf{Panel A: Moments and Risk Measures}\\",
        r"    \vspace{1em}\\",
        tab_a,
        r"    \vspace{0.5cm}\\",
        r"    \textbf{Panel B: Ratios}\\",
        r"    \vspace{1em}\\",
        tab_b,
        r"  \end{minipage}%",
        r"}"
    ])

    # assemble final table
    parts = [
        r"\begin{table}[H]",
        r"\centering",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        combined,
        r"\end{table}"
    ]

    return "\n".join(parts)


import pandas as pd


def vol_period_analysis(sum_df, name, split_method):
    df = sum_df
    spx_mask = df['ticker'] == 'SPX'

    # Build a small DataFrame of SPX’s SW_0_30 by date
    spx_sw = (
        df[spx_mask]
        .loc[:, ['date', 'SW_0_30']]
        .drop_duplicates()
        .set_index('date')
    )

    # Assign periods to SPX dates, two modes:
    if split_method == 'quantile':
        # exactly as before
        q30, q70 = spx_sw['SW_0_30'].quantile([0.33333333, 0.66666666])
        spx_sw['period'] = pd.cut(
            spx_sw['SW_0_30'],
            bins=[-np.inf, q30, q70, np.inf],
            labels=[3, 2, 1]
        ).astype(int)

    elif split_method == 'time':
        # make sure the dates are in ascending order
        spx_sw = spx_sw.sort_index()

        start, end = spx_sw.index.min(), spx_sw.index.max()
        third = (end - start) / 3
        bins = [start, start + third, start + 2 * third, end]

        # include_lowest=True so the very first date lands in bin 1
        spx_sw['period'] = pd.cut(
            spx_sw.index.to_series(),
            bins=bins,
            labels=[1, 2, 3],
            include_lowest=True
        ).astype(int)


    else:
        raise ValueError("split_method must be 'quantile' or 'time'")

    # Merge that period label back onto the full df (matching on date)
    df = df.merge(
        spx_sw['period'].rename('period'),
        left_on='date',
        right_index=True,
        how='left'
    )



    # start with all unique tickers in the right order (using CarrWu2009_table_2 to sort for ease of use)
    tickers_out = CarrWu2009_table_2(df, "Liquid", save_latex=False)["ticker"].unique()
    tickers = ticker_list_to_ordered_map(tickers_out, "ticker_out")["ticker"]
    period_df = pd.DataFrame({'ticker': tickers})

    # loop over your three periods
    periods = [1, 2, 3]
    for p in periods:
        sub = df[df['period'] == p]

        # 2a) Carr & Wu Table 2 (Mean_RV, Mean_SW)
        tbl2 = CarrWu2009_table_2(sub, "Liquid", save_latex=False)
        # if ticker is the index, bring it back as a column
        # if tbl2.index.name == 'ticker' or isinstance(tbl2.index, pd.Index):
        #     tbl2 = tbl2.reset_index().rename(columns={'index':'ticker'})
        tmp2 = tbl2[['ticker', 'Mean_RV', 'Mean_SW']].rename(columns={
            'Mean_RV': f'Mean_RV_{p}',
            'Mean_SW': f'Mean_SW_{p}',
        })
        period_df = period_df.merge(tmp2, on='ticker', how='left')

        # 2b) Carr & Wu Table 3 (VRP, t-stats, LVRP, t-stats)
        tbl3 = CarrWu2009_table_3(sub, "Liquid", save_latex=False)
        # if tbl3.index.name == 'ticker' or isinstance(tbl3.index, pd.Index):
        #     tbl3 = tbl3.reset_index().rename(columns={'index':'ticker'})
        tmp3 = tbl3[['ticker', 'Mean_diff', 't_diff', 'Mean_ln', 't_ln']].rename(columns={
            'Mean_diff': f'VRP_{p}',
            't_diff': f't_VRP_{p}',
            'Mean_ln': f'LVRP_{p}',
            't_ln': f't_LVRP_{p}',
        })
        period_df = period_df.merge(tmp3, on='ticker', how='left')

    # 3) (optional) sort tickers
    period_df = period_df.sort_values('ticker').reset_index(drop=True)
    period_df = sort_table_df(period_df, df, name)
    period_df["ticker"] = vp.ticker_list_to_ordered_map(period_df["ticker"])["ticker_out"]

    vol_period_analysis_latex(period_df, name, split_method = split_method)
    return period_df


def vol_period_analysis_latex(out, name, split_method):
    # 1) Prepare a copy and format RV and SW to two‐decimal strings

    # 1) scale RV and SW by 100
    for p in [1, 2, 3]:
        out[f"RV_{p}"] = out[f"Mean_RV_{p}"]
        out[f"SW_{p}"] = out[f"Mean_SW_{p}"]

    # 2) format VRP and LVRP with their t‐stats in parentheses
    for p in [1, 2, 3]:
        out[f"VRP_{p}"] = out.apply(
            lambda r: f"{r[f'VRP_{p}']:.2f}\\,( {r[f't_VRP_{p}']:.2f} )",
            axis=1
        )
        out[f"LVRP_{p}"] = out.apply(
            lambda r: f"{r[f'LVRP_{p}']:.2f}\\,( {r[f't_LVRP_{p}']:.2f} )",
            axis=1
        )

    for p in [1, 2, 3]:
        out[f"RV_{p}"] = out[f"RV_{p}"].map(lambda x: f"{x:.2f}")
        out[f"SW_{p}"] = out[f"SW_{p}"].map(lambda x: f"{x:.2f}")

    # 2) Select columns in the desired order
    cols = (
            ['ticker'] +
            [f"RV_{p}" for p in [1, 2, 3]] +
            [f"SW_{p}" for p in [1, 2, 3]] +
            [f"VRP_{p}" for p in [1, 2, 3]] +
            [f"LVRP_{p}" for p in [1, 2, 3]]
    )
    out2 = out[cols]

    # 3) Build a MultiIndex for the two header rows
    first_row = [''] \
                + ['RV\\times100'] * 3 \
                + ['SW\\times100'] * 3 \
                + ['(RV - SW)\\times100'] * 3 \
                + ['\\(\\ln(RV / SW)\\)'] * 3

    # second_row = ['Ticker'] \
    #              + ['S1', 'S2', 'S3'] \
    #              + ['S1', 'S2', 'S3'] \
    #              + ['S1', 'S2', 'S3'] \
    #              + ['S1', 'S2', 'S3']

    wrap = lambda s: rf'\multicolumn{{1}}{{c}}{{{s}}}'
    second_row = (['Ticker'] +
                  [wrap(fr'$S_{p}$') for _ in range(4) for p in (1, 2, 3)])

    out2.columns = pd.MultiIndex.from_arrays([first_row, second_row])

    # 4) Export to LaTeX with no index, allowing multicolumn headers, and tight columns
    body = out2.to_latex(
        index=False,
        escape=False,
        multicolumn=True,
        multicolumn_format='c',
        column_format='l' + '|rrr' * int((len(out2.columns) - 1) / 3)
    )

    lines = body.splitlines()

    # build the cmidrule line (adjust column numbers if you change column_format)
    cm = r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}"

    # find where the second header line ends (the one with all the multicolumn specs)
    # it's usually line 2 (0-based), but you can search for '\multicolumn' if you like
    lines.insert(3, cm)

    body = "\n".join(lines)

    # 5) Wrap in table+scalebox
    latex = f"""
    \\begin{{table}}[htbp]
      \\centering
      % reduce horizontal padding
      \\setlength{{\\tabcolsep}}{{3pt}}
      % scale to fit width
      \\scalebox{{0.7}}{{%
    {body}
      }}
      \\caption{{Period‐by‐period RV, SW, VRP, and LVRP (with t‐stats)}}
      \\label{{tab:period_metrics_({name}, {split_method})}}
    \\end{{table}}
    """

    latex = latex.replace(r"nan\,( nan )", " ").replace(r"nan", " ")
    with open(f"figures/Analysis/Period/{name}_{split_method}_period_analysis.tex", "w") as f:
        f.write(latex)





def make_bid_ask_table(t2_bid, t3_bid, t2_ask, t3_ask, name,
                       caption="Period-by-period SW, VRP, and LVRP (bid vs. ask)",
                       label="tab:period_metrics_bid_ask"):
    """
    t2_* should have columns ['ticker', ..., 'Mean_SW', ...]
    t3_* should have columns ['ticker', ..., 'Mean_diff','t_diff','Mean_ln','t_ln', ...]
    """
    # align by ticker
    t2b = t2_bid.set_index('ticker')
    t3b = t3_bid.set_index('ticker')
    t2a = t2_ask.set_index('ticker')
    t3a = t3_ask.set_index('ticker')
    tickers = t2b.index.intersection(t2a.index).intersection(t3b.index).intersection(t3a.index)

    # header
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \scalebox{0.7}{%",
        r"    \begin{tabular}{l|rr|rr|rr}",
        r"      \toprule",
        r"      & \multicolumn{2}{c}{SW\times100}",
        r"      & \multicolumn{2}{c}{(RV - SW)\times100}",
        r"      & \multicolumn{2}{c}{$\ln(RV / SW)$} \\",
        r"      \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"      Ticker & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} \\",
        r"      \midrule",
    ]

    # rows
    for t in tickers:
        sw_b = t2b.at[t, 'Mean_SW']
        sw_a = t2a.at[t, 'Mean_SW']
        v_b = t3b.at[t, 'Mean_diff']
        v_a = t3a.at[t, 'Mean_diff']
        tv_b = t3b.at[t, 't_diff']
        tv_a = t3a.at[t, 't_diff']
        l_b = t3b.at[t, 'Mean_ln']
        l_a = t3a.at[t, 'Mean_ln']
        tl_b = t3b.at[t, 't_ln']
        tl_a = t3a.at[t, 't_ln']

        line = (
            f"{t} & "
            f"{sw_b:.2f} & {sw_a:.2f} & "
            f"{v_b:.2f}\\,({tv_b:.2f}) & {v_a:.2f}\\,({tv_a:.2f}) & "
            f"{l_b:.2f}\\,({tl_b:.2f}) & {l_a:.2f}\\,({tl_a:.2f}) \\\\"
        )
        lines.append("      " + line)

    # footer
    lines += [
        r"      \bottomrule",
        r"    \end{tabular}%",
        r"  }",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}"
    ]

    latex = "\n".join(lines)
    with open(f"figures/Analysis/Implementability/{name}_bid_ask_table.tex", "w") as f:
        f.write(latex)

    return latex


def make_bid_ask_table_ff5(
    t2_bid,
    t3_bid,
    t2_ask,
    t3_ask,
    tff5_bid,
    tff5_ask,
    name,
    caption="Period-by-period SW, VRP, LVRP, and alpha (F5) (bid vs. ask)",
    label="tab:period_metrics_bid_ask"
):
    r"""
    t2_* should have columns ['ticker', ..., 'Mean_SW', ...]
    t3_* should have columns ['ticker', ..., 'Mean_diff','t_diff','Mean_ln','t_ln', ...]
    tff5_* should have column ['\\(\\alpha\\)'] with pre-formatted LaTeX strings
    """
    # align by ticker
    t2b = t2_bid.set_index('ticker')
    t3b = t3_bid.set_index('ticker')
    t2a = t2_ask.set_index('ticker')
    t3a = t3_ask.set_index('ticker')
    t5b = tff5_bid.set_index('ticker')
    t5a = tff5_ask.set_index('ticker')

    tickers = (
        t2b.index
          .intersection(t2a.index)
          .intersection(t3b.index)
          .intersection(t3a.index)
          .intersection(t5b.index)
          .intersection(t5a.index)
    )

    # header
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \scalebox{0.7}{%",
        r"    \begin{tabular}{l|rr|rr|rr|rr}",
        r"      \toprule",
        r"      & \multicolumn{2}{c}{SW$\times$100}",
        r"      & \multicolumn{2}{c}{(RV - SW)$\times$100}",
        r"      & \multicolumn{2}{c}{$\ln(\text{RV} / \text{SW})$}",
        r"      & \multicolumn{2}{c}{alpha$^\text{F$_5$}$} \\",
        r"      \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"      Ticker & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} & \multicolumn{1}{c}{bid} & \multicolumn{1}{c}{ask} \\",
        r"      \midrule",
    ]

    # rows
    for t in tickers:
        sw_b = t2b.at[t, 'Mean_SW']
        sw_a = t2a.at[t, 'Mean_SW']
        v_b = t3b.at[t, 'Mean_diff']
        v_a = t3a.at[t, 'Mean_diff']
        tv_b = t3b.at[t, 't_diff']
        tv_a = t3a.at[t, 't_diff']
        l_b = t3b.at[t, 'Mean_ln']
        l_a = t3a.at[t, 'Mean_ln']
        tl_b = t3b.at[t, 't_ln']
        tl_a = t3a.at[t, 't_ln']
        alpha_b_raw = t5b.at[t, '\\(\\alpha\\)']
        alpha_a_raw = t5a.at[t, '\\(\\alpha\\)']

        # format alpha: replace newline with LaTeX spacing
        alpha_b = alpha_b_raw.replace("\n", "\\, ")
        alpha_a = alpha_a_raw.replace("\n", "\\, ")

        line = (
            fr"{t} & {sw_b:.2f} & {sw_a:.2f} & "
            fr"{v_b:.2f}\, ({tv_b:.2f}) & {v_a:.2f}\, ({tv_a:.2f}) & "
            fr"{l_b:.2f}\, ({tl_b:.2f}) & {l_a:.2f}\, ({tl_a:.2f}) & "
            fr"{alpha_b} & {alpha_a} \\"
        )
        lines.append("      " + line)

    # footer
    lines += [
        r"      \bottomrule",
        r"    \end{tabular}%",
        r"  }",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}"
    ]

    latex = "\n".join(lines)
    file_path = f"figures/Analysis/Implementability/{name}_bid_ask_table_ff5.tex"
    with open(file_path, "w") as f:
        f.write(latex)

    return latex