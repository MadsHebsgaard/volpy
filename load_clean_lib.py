import numpy as np
from pathlib import Path
import volpy_func_lib as vp
import pandas as pd

from global_settings import *

import importlib
import global_settings
importlib.reload(global_settings)
days_type = global_settings.days_type

importlib.reload(vp)


def dirs():
    profile = my_profile()
    if profile == "Mads":
        Option_metrics_path = Path(r"D:\Finance Data\OptionMetrics")
    elif profile == "Axel":
        Option_metrics_path = Path(r"C:\Users\axell\Desktop\CBS\data\OptionMetrics")
    elif profile == "Mads_Laptop":
        Option_metrics_path = Path(r"C:\Users\hebsg\Data\OptionMetrics")

    dir = {
        "OptionMetrics": Option_metrics_path,
        "i4s4_CW": Option_metrics_path / "i4s4_CW",
        "SPX_short": Option_metrics_path / "SPX_short",
        "SPX_full": Option_metrics_path / "SPX_full",
        "i2s1_full": Option_metrics_path / "i2s1_full",
        "s5_full": Option_metrics_path / "s5_full",
        "SPX_full_v2": Option_metrics_path / "SPX_full_v2",
        "vix_markets_full": Option_metrics_path / "vix_markets_full",
        "vix_assets": Option_metrics_path / "vix_assets",
        "Liquid_ETFs": Option_metrics_path / "Liquid_ETFs", # combine with distilled OEX to get Liquid
        "VIX": Option_metrics_path / "VIX", # Final
        "OEX": Option_metrics_path / "OEX", # Final
        "Cross-AM": Option_metrics_path / "Cross-AM", # Final
        # "Liquid": Option_metrics_path / "Liquid", # Final (Liquid = distilled OEX + Liquid_ETFs)
    }
    return dir

def Option_metrics_path_from_profile():
    profile = my_profile()
    if profile == "Mads":
        return Path(r"D:\Finance Data\OptionMetrics")
    elif profile == "Axel":
        return Path(r"C:\Users\axell\Desktop\CBS\data\OptionMetrics")
    elif profile == "Mads_Laptop":
        return Path(r"C:\Users\hebsg\Data\OptionMetrics")
    else:
        print("choose viable profile such as 'Axel' or 'Mads' in global settings")
    return

def volpy_output_dir(om_folder):
    om_dir = Option_metrics_path_from_profile()
    return om_dir / "volpy data output" / om_folder


def load_od_FW_ZCY(om_folder="i4s4", tickers=None):
    """Loader optionsdata, forward priser, returns og yield curves."""
    om_dir = Option_metrics_path_from_profile()
    dir = om_dir / om_folder

    if not dir.is_dir():
        print("The specified OptionMetrics folder 'om_folder' does not exist, add the folder to OptionMetrics directorary")

    volpy_output_dir(om_folder).mkdir(parents=True, exist_ok=True)

    # Load hver dataset direkte
    od = vp.load_option_data(dir / "option data.csv")

    FW = vp.load_forward_price(dir / "forward price.csv")
    ret = vp.load_returns_and_price(dir / "returns and stock price.csv")
    ZCY_curves = vp.load_ZC_yield_curve(dir / "ZC yield curve.csv")

    # Filtrér på tickers, hvis angivet
    if tickers:
        od = od[od["ticker"].isin(tickers)]
        FW = FW[FW["ticker"].isin(tickers)]
        ret = ret[ret["ticker"].isin(tickers)]

    return od, FW, ZCY_curves, ret


def load_od_FW_ZCY_chunks(od_i_file, om_folder="i4s4", tickers=None):
    """Loader optionsdata, forward priser, returns og yield curves."""
    om_dir = Option_metrics_path_from_profile()
    dir = om_dir / om_folder

    if not dir.is_dir():
        print("The specified OptionMetrics folder 'om_folder' does not exist, add the folder to OptionMetrics directorary")

    volpy_output_dir(om_folder).mkdir(parents=True, exist_ok=True)

    # Load hver dataset direkte
    od = vp.clean_option_data_chunks(od_i_file, om_dir) #vp.load_option_data_chunks(od_i = od_i)
    # FW = vp.load_forward_price(dir / "forward price.csv")
    # ret = vp.load_returns_and_price(dir / "returns and stock price.csv")
    # ZCY_curves = vp.load_ZC_yield_curve(dir / "ZC yield curve.csv")

    # Filtrér på tickers, hvis angivet
    if tickers:
        od = od[od["ticker"].isin(tickers)]
        # FW = FW[FW["ticker"].isin(tickers)]
        # ret = ret[ret["ticker"].isin(tickers)]

    return od


def load_option_metadata(om_folder="i4s4", tickers=None):

    om_dir = Option_metrics_path_from_profile()
    base = om_dir / om_folder
    if not base.is_dir():
        raise FileNotFoundError(f"OptionMetrics-mappen '{om_folder}' findes ikke under {om_dir}")

    # 1) Load metadata
    FW = vp.load_forward_price(base / "forward price.csv")
    ret = vp.load_returns_and_price(base / "returns and stock price.csv")
    ZCY = vp.load_ZC_yield_curve(base / "ZC yield curve.csv")

    # 2) Filtrér på tickers hvis ønsket
    if tickers:
        FW = FW[FW["ticker"].isin(tickers)]
        ret = ret[ret["ticker"].isin(tickers)]
        # (ZCY er typisk ikke ticker-specific, så vi lader den være)

    return FW, ZCY, ret



def clean_od(od, first_day = pd.to_datetime("1996-01-04"), last_day = pd.to_datetime("2003-02-28")):

    days_var = days_type() + "days"

    # od = od.dropna(subset=["K", "IV_om", "cp_flag"])
    # od = od[od["c_days"] <= 365]
    od["spread"] = od["ask"] - od["bid"]
    od["mid"] = od["bid"] + od["spread"] / 2
    od = (
        od
        .dropna(subset=["K", "IV_om", "cp_flag"])
        .query(f"{days_var} <= 365 and spread > 0 and bid > 0")
    )
    # od = od[od["volume"] > 5000]

    # od = od[(od["date"] >= first_day) & (od["date"] <= last_day)].sort_values(
    #     by="date").reset_index(drop=True)
    od = od[(od["date"] >= first_day) & (od["date"] <= last_day)].sort_values(
        by=["date", "ticker", "c_days"]
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
    summary_dly_df["#K_low_before_filter"] = np.nan
    summary_dly_df["#K_high_before_filter"] = np.nan

    summary_dly_df["Active"] = False
    summary_dly_df["Inactive reason"] = "missing"

    return summary_dly_df

def load_clean_and_prepare_od(om_folder, tickers=None, first_day=None, last_day=None, IV_type = "od", safe_slow_IV = False):
    # load data
    print(f"{days_type()} was selected in global_settings.py")
    od, FW, ZCY_curves, returns_and_prices = load_od_FW_ZCY(om_folder, tickers=tickers)
    returns_and_prices = returns_and_prices[(returns_and_prices["open"] > 0) & (returns_and_prices["close"] > 0)]

    if first_day is None:   first_day = od["date"].min()
    if last_day is None:    last_day = od["date"].max()

    # add forward (before cleaning such that od_raw has the forward rate, used in options trats for ATM options that might be slightly ITM)
    od = vp.add_FW_to_od(od, FW)

    # add IV to options (bid/ask/mid/om)
    od["IV"] = vp.add_bid_mid_ask_IV(od, IV_type, safe_slow_IV = safe_slow_IV)
    od[f"IV_{IV_type}"] = od["IV"]

    od_raw = od

    # clean data (should be looked upon)
    od = clean_od(od, first_day=first_day, last_day=last_day)
    # remove if ITM
    od = od.loc[((od["F"] < od["K"]) & (od["cp_flag"] == "C")) | ((od["F"] > od["K"]) & (od["cp_flag"] == "P"))]
    # print(od.shape, "sdsfd")

    # add r to options
    od = vp.add_r_to_od_parallel(od, ZCY_curves)
    print("Data loaded")

    return od, returns_and_prices, od_raw


def load_clean_and_prepare_od_chunks(od_i_file, om_folder, FW, ZCY_curves, tickers=None, first_day=None, last_day=None, IV_type = "od", safe_slow_IV = False):
    # load data (is loaded)    
    print(f"{days_type()} was selected in global_settings.py")
    od = load_od_FW_ZCY_chunks(od_i_file, om_folder, tickers=tickers)

    if first_day is None:   first_day = od["date"].min()
    if last_day is None:    last_day = od["date"].max()

    # add forward (before cleaning such that od_raw has the forward rate, used in options trats for ATM options that might be slightly ITM)
    od = vp.add_FW_to_od(od, FW)

    # add IV to options (bid/ask/mid/om)
    od["IV"] = vp.add_bid_mid_ask_IV(od, IV_type, safe_slow_IV = safe_slow_IV)
    od[f"IV_{IV_type}"] = od["IV"]

    od_raw = od

    # clean data (should be looked upon)
    od = clean_od(od, first_day=first_day, last_day=last_day)
    # remove if ITM
    od = od.loc[((od["F"] < od["K"]) & (od["cp_flag"] == "C")) | ((od["F"] > od["K"]) & (od["cp_flag"] == "P"))]
    # print(od.shape, "sdsfd")

    # add r to options
    od = vp.add_r_to_od_parallel(od, ZCY_curves)
    print("Data loaded")

    return od, od_raw


def create_summary_dly_df(od, returns_and_prices, first_day=None, last_day=None, n_grid=200):
    if first_day is None:   first_day = od["date"].min()
    if last_day is None:    last_day = od["date"].max()

    # Add low/high and create summary (Filters dataset for criteria such as min 3 strikes ... min 8 days...)
    od, summary_dly_df = vp.od_filter_and_summary_creater(od)
    summary_dly_df.reset_index(inplace=True)


    # Add risk-free rate of the given date
    RF = download_factor_df(Factor_list=["FF5"])[['date', 'RF']]
    summary_dly_df = summary_dly_df.merge(RF[['date', 'RF']], on='date', how='left')

    # Realized vol calculation
    real_vol = vp.calc_realized_var(returns_and_prices, first_day, last_day)

    # Merge realized vol with summary_dly_df
    summary_dly_df = vp.add_realized_vol_to_summary(summary_dly_df, real_vol)

    # only keep the lowest ("low") and the second lowest ("high") TTMs
    od_rdy = od[(od["low"] == True) | (od["high"] == True)]

    # Calculate high/low Swap Rates
    summary_dly_df = vp.high_low_swap_rates(summary_dly_df, od_rdy, n_points=n_grid)

    return summary_dly_df, od_rdy



# # Old Volatility index mapping
# old_ticker_to_vol = {
#     "SPX": "VIX",
#     "QQQ": "VXN",
#     "IWM": "RVX",
#     "DIA": "VXD",
#     "FXE": "EVZ",
#     "EWZ": "VXEWZ",
#     "EEM": "VXEEM",
#     "TLT": "VIXTLT",
#     "SLV": "VXSLV",
#     "USO": "OVX",
#     "AAPL":  "VXAPL",
#     "AMZN":  "VXAZN",
#     "GOOG": "VXGOG",
#     "GS":    "VXGS",
#     "IBM":   "VXIBM"
# }


# Mapping from underlying asset ticker to Cboe volatility index ticker
ticker_to_vol = {
    "SPX":  "VIX",      # S&P500 → Cboe VIX
    "GLD":  "GVZ",    # SPDR Gold Shares ETF → Cboe Gold ETF Volatility Index
    "USO":  "OVX",    # United States Oil Fund ETF → Cboe Crude Oil ETF Volatility Index
    "RUT":  "RVX",    # Russell 2000 Index → Cboe Russell 2000 Volatility Index
    "AAPL": "VXAPL",  # Apple Inc. → Cboe Apple VIX Index
    "AMZN": "VXAZN",  # Amazon.com Inc. → Cboe Amazon VIX Index
    "DJX":  "VXD",    # Dow Jones Industrial Average → Cboe Dow Jones Industrial Average Volatility Index
    "GOOG": "VXGOG",  # Alphabet Inc. Class C → Cboe Google VIX Index
    "GS":   "VXGS",   # Goldman Sachs Group Inc. → Cboe Goldman Sachs VIX Index
    "IBM":  "VXIBM",  # IBM Corp. → Cboe IBM VIX Index
    "NDX":  "VXN",    # Nasdaq-100® Index → Cboe Nasdaq-100 Volatility Index
    "TLT":  "VXTLT",  # iShares 20+ Year Treasury Bond ETF → Cboe 20+ Year Treasury Bond ETF Volatility Index
    "EWZ":  "VXEWZ",  # iShares Brazil ETF → Cboe Brazil ETF Volatility Index
    "EEM":  "VXEEM",  # iShares MSCI Emerging Markets ETF → Cboe Emerging Markets ETF Volatility Index
    "EFA":  "VXEFA",  # iShares MSCI EAFE ETF → Cboe EFA ETF Volatility Index
    "VIX":  "VVIX",   # Cboe Volatility Index → Cboe VVIX Index
}
vol_symbols = list(ticker_to_vol.values())

# mangler GLD, EFA og returns for RUT





def ticker_to_vol_symbol(s):
    for ticker, vol in ticker_to_vol.items():
        if ticker in s:
            return s.replace(ticker, vol)
    return s  # unchanged if no match



def download_factor_df(Factor_list=["FF5", "SPX", "UMD", "BAB", "QMJ", "vol_indexes"]):
    import pandas as pd
    import requests
    from io import BytesIO
    from zipfile import ZipFile

    # Urls
    url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    url_bab = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx"
    url_qmj = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Daily.xlsx"
    # url_vix = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=VIXCLS&scale=left&cosd=1990-01-02&coed=2025-04-16&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily%2C%20Close&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-21&revision_date=2025-04-21&nd=1990-01-02"

    # Load FF5 Daily Data
    # fast as zip file
    # https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    response_ff5 = requests.get(url_ff5)
    zip_file_ff5 = ZipFile(BytesIO(response_ff5.content))
    csv_filename_ff5 = zip_file_ff5.namelist()[0]

    df = pd.read_csv(zip_file_ff5.open(csv_filename_ff5), skiprows=3)
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    df.rename(columns={'Mkt-RF': 'Mkt'}, inplace=True)
    # Keep only rows where date is an 8-digit string (YYYYMMDD)
    df = df[df['date'].astype(str).str.match(r'^\d{8}$')]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df / 100

    if "SPX" in Factor_list:
        import volpy_func_ticker_lib as vtp
        from pathlib import Path

        # fix any Bloomberg‐OM quirks, then point to your SPX CSV folder
        vtp.fix_index_returns_bloomberg_OM(["SPX"])
        SPX_dir = Option_metrics_path_from_profile() / "Tickers" / "Input" / "SPX"

        # read in the SPX returns file
        SPX_df = pd.read_csv(SPX_dir / "returns and stock price.csv", usecols=["date", "return"])
        SPX_df['date'] = pd.to_datetime(SPX_df['date'])
        SPX_df.rename(columns={'return': 'SPX'}, inplace=True)
        SPX_df.set_index('date', inplace=True)

        # join the SPX series into your main factor df
        df = df.join(SPX_df['SPX'], how='inner')
        df["SPX+RF"] = df["SPX"]
        df["SPX"] = df["SPX+RF"] - df["RF"]

    if "UMD" in Factor_list:
        # Load Momentum (UMD) Daily Data
        # fast as zip file
        # https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

        response_mom = requests.get(url_mom)
        zip_file_mom = ZipFile(BytesIO(response_mom.content))
        csv_filename_mom = zip_file_mom.namelist()[0]

        df_mom = pd.read_csv(zip_file_mom.open(csv_filename_mom), skiprows=13)
        df_mom.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        df_mom.rename(columns={'Mom   ': 'UMD'}, inplace=True)
        df_mom = df_mom[df_mom['date'].astype(str).str.match(r'^\d{8}$')]
        df_mom['date'] = pd.to_datetime(df_mom['date'], format='%Y%m%d')
        df_mom.set_index('date', inplace=True)
        df_mom = df_mom.apply(pd.to_numeric, errors='coerce')
        df_mom = df_mom / 100

        # Merge the datasets on the date index
        # This will join the 5 factors with the momentum factor (UMD)
        df = df.join(df_mom, how='inner')

    if "BAB" in Factor_list:
        df_bab = (
            pd.read_excel(url_bab, skiprows=18, usecols="A:AD")
            .rename(columns={'DATE': 'date'})
        )
        df_bab['date'] = pd.to_datetime(df_bab['date'], format='%m/%d/%Y')
        df_bab.set_index('date', inplace=True)

        # join the 'USA' column from df_bab as your BAB series
        df['BAB'] = df_bab['USA']

    if "QMJ" in Factor_list:
        df_qmj = (
            pd.read_excel(url_qmj, skiprows=18, usecols="A:AD")
            .rename(columns={'DATE': 'date'})
        )
        df_qmj['date'] = pd.to_datetime(df_qmj['date'], format='%m/%d/%Y')
        df_qmj.set_index('date', inplace=True)

        df['QMJ'] = df_qmj['USA']

    if "vol_indexes" in Factor_list:
        # download additional Cboe vol indexes directly
        for sym in vol_symbols:
            url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
            # read full CSV so we can detect if the column name is 'sym' or 'CLOSE'
            df_tmp = pd.read_csv(url, parse_dates=["DATE"])
            # pick the right "close" field
            close_var = sym if sym in df_tmp.columns else "CLOSE"
            df_tmp = df_tmp[["DATE", close_var]].rename(
                columns={"DATE": "date", close_var: sym}
            )
            # scale to decimal
            df_tmp[sym] = df_tmp[sym] / 100
            df = df.merge(df_tmp[['date', sym]], on='date', how='left')

    df.reset_index(inplace=True)
    return df

def download_factors():
    factor_df = download_factor_df()

    import os
    os.makedirs("data", exist_ok=True)
    factor_df.to_csv('data/factor_df.csv', index=False)

def Create_Security_map():
    '''
    Dependencies:
        - bloomberg map.xlsx     (bloomberg data for de forskellige (bloomberg) tickers)
        - Wrds-OM map.xlsx       (forskellige id og tickers for hvert aktiv)
    '''

    base_dir = Option_metrics_path_from_profile()
    bloomberg_dir = base_dir / "Tickers" / "index data"

    # Load Excel files
    bloomberg_df = pd.read_excel(bloomberg_dir / "bloomberg map.xlsx")
    id_map_df = pd.read_excel(bloomberg_dir / "Wrds-OM map.xlsx")

    # Truncate the last two chars of ID_CUSIP and normalize
    bloomberg_df['cusip_key'] = (
        bloomberg_df['ID_CUSIP']
        .astype(str)
        .str[:-1]  # drop last two characters
        .str.strip()
        .str.upper()
    )

    # also normalize the id_map_df key
    id_map_df['cusip_key'] = (
        id_map_df['cusip']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # build the lookup (dropping any duplicate keys)
    name_map = (
        bloomberg_df
        .drop_duplicates(subset='cusip_key', keep='first')
        .set_index('cusip_key')['LONG_COMP_name']
    )

    # map into a new column 'name'
    id_map_df['name'] = id_map_df['cusip_key'].map(name_map)
    id_map_df['name'] = id_map_df['name'].str.replace('/The', '', regex=False)

    # apply lookup, defaulting to the original ticker if not in the dict
    id_map_df['ticker_out'] = (
        id_map_df['ticker']
        .map(vp.ticker_to_ticker_out)
        .fillna(id_map_df['ticker'])
    )

    # apply overrides, defaulting to the original name if ticker not in the dict
    id_map_df['name'] = (
        id_map_df['ticker']
        .map(vp.name_overrides)
        .fillna(id_map_df['name'])
    )

    id_map_df["Cboe vol ticker"] = id_map_df["ticker"].map(ticker_to_vol)

    id_map_df["Synthetic"] = False
    EWU_combined = {
        "ticker": "EWU_combined",
        "name": "United Kingdom ETF",
        "ticker_out": "EWU",
        "cusip": "46435G334/46428669",
        "permno": "14907/83216",
        "secid": "106420",
        "cusip_key": "46435G334/46428669",
        "Synthetic": True,
    }
    id_map_df = pd.concat([id_map_df, pd.DataFrame([EWU_combined])], ignore_index=True)
    id_map_df.to_excel(bloomberg_dir / "Security_map.xlsx", index=False)


def fix_EWU():
    base_dir = Option_metrics_path_from_profile()
    bloomberg_dir = base_dir / "Tickers" / "index data"
    Input_dir = base_dir / "Tickers" / "Input"

    EWU_old_dir = Input_dir / "EWU_OLD"
    EWU_new_dir = Input_dir / "EWU"

    EWU_combined_dir = Input_dir / "EWU_combined"
    EWU_combined_dir.mkdir(parents=True, exist_ok=True)

    for file in EWU_old_dir.iterdir():
        if not file.is_file():
            continue

        df_old_path = EWU_old_dir / file.name
        df_new_path = EWU_new_dir / file.name
        df_combined_path = EWU_combined_dir / file.name

        df_old = pd.read_csv(df_old_path)
        df_new = pd.read_csv(df_new_path)

        # merge: simple concat + dedupe
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined["ticker"] = "EWU_combined"
        df_combined = df_combined.drop_duplicates()

        # write
        df_combined.to_csv(df_combined_path, index=False)
        print(df_combined_path)


def create_log_with_enty(message):
    base_dir = Option_metrics_path_from_profile()

    # add this:
    log_path = base_dir / "Tickers" / "log.txt"
    log_path.write_text(message)
