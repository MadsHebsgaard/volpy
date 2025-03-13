import wrds
import pandas as pd
import load_clean_lib as load

def fetch_options_data(db, begdate, enddate, tickers, csv_path):
    # db = wrds.Connection(wrds_username=username)

    secid_list = None  # Default: Hent hele databasen

    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]

        secid_df = db.raw_sql(f"""
            SELECT secid, ticker, cusip, issuer
            FROM optionm.securd1
            WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
        """)

        secid_list = secid_df["secid"].tolist()

        # Konverter til SQL-venlig streng
        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    df_list = []

    for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
        query = f"""
            SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
                   o.date, o.exdate, o.cp_flag, o.strike_price, 
                   o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
                   o.cfadj, ss_flag,
                   (o.exdate - o.date) AS days_diff  -- Beregn dage direkte i SQL
            FROM optionm.opprcd{year} o
            LEFT JOIN optionm.securd1 s ON o.secid = s.secid
            WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
              AND (o.exdate - o.date) > 8  
              AND (o.exdate - o.date) <= 365 
        """

        if secid_list:
            query += f" AND o.secid IN {secid_str}"

        df_list.append(db.raw_sql(query, date_cols=["date", "exdate"]))

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)
    
    # db.close()
    return df


def fetch_forward_prices(db, begdate, enddate, tickers, csv_path):
    # db = wrds.Connection(wrds_username=username)

    secid_list = None

    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]

        secid_df = db.raw_sql(f"""
            SELECT secid, ticker, cusip, issuer
            FROM optionm.securd1
            WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
        """)

        secid_list = secid_df["secid"].tolist()

        # Konverter til SQL-venlig streng
        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"  # Ingen komma, hvis kun én værdi
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"  # Normal liste
        else:
            secid_str = "(NULL)"  # Ingen secid'er fundet

    df_list = []

    for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
        query = f"""
            SELECT f.secid, s.ticker, s.cusip, s.issuer, 
                   f.date, f.expiration, f.amsettlement, f.forwardprice
            FROM optionm.fwdprd{year} f
            LEFT JOIN optionm.securd1 s ON f.secid = s.secid
            WHERE f.date BETWEEN '{begdate}' AND '{enddate}'
        """

        if secid_list:
            query += f" AND f.secid IN {secid_str}"

        df_list.append(db.raw_sql(query, date_cols=["date", "expiration"]))

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)
    
    # db.close()
    return df


def fetch_stock_returns(db, begdate, enddate, tickers, csv_path):
    # db = wrds.Connection(wrds_username=username)

    secid_list = None

    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]

        secid_df = db.raw_sql(f"""
            SELECT secid, ticker, cusip, issuer
            FROM optionm.securd1
            WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
        """)

        secid_list = secid_df["secid"].tolist()

        # Konverter til SQL-venlig streng
        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"  # Ingen komma, hvis kun én værdi
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"  # Normal liste
        else:
            secid_str = "(NULL)"  # Ingen secid'er fundet

    df_list = []

    for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
        query = f"""
            SELECT p.secid, s.ticker, s.cusip, s.issuer, 
                   p.date, p.open, p.close, p.return
            FROM optionm.secprd{year} p
            LEFT JOIN optionm.securd1 s ON p.secid = s.secid
            WHERE p.date BETWEEN '{begdate}' AND '{enddate}'
        """

        if secid_list:
            query += f" AND p.secid IN {secid_str}"

        df_list.append(db.raw_sql(query, date_cols=["date"]))

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)

    # db.close()
    return df


# def fetch_wrds_data(begdate, enddate, tickers, data_type, csv_path):

#     function_map = {
#         "O": fetch_options_data,
#         "F": fetch_forward_prices,
#         "S": fetch_stock_returns
#     }

#     if data_type not in function_map:
#         raise ValueError("Ugyldig data_type. Brug 'O', 'F', eller 'R'.")

#     return function_map[data_type](begdate, enddate, tickers, csv_path)


# def dirs(profile):
#     if profile == "Mads":
#         Option_metrics_path = Path(r"D:\Finance Data\OptionMetrics")
#     elif profile == "Axel":
#         Option_metrics_path = Path(r"C:\Users\axell\Desktop\CBS\data\OptionMetrics")
#     else:
#         raise ValueError("Ugyldigt profile navn. Brug 'Mads' eller 'Axel'.")

#     return Option_metrics_path

# Hovedfunktionen
def fetch_wrds_data(db, profile, folder_name, begdate, enddate, tickers=None, data_types=["O","F","S"], return_df=False):


    # Find base directory for brugerprofil
    base_dir = load.dirs(profile)["OptionMetrics"] / folder_name
    base_dir.mkdir(parents=True, exist_ok=True)  # Opret mappen hvis den ikke findes

    # Hvis ingen specifikke data_types er angivet, hentes alle tre
    if data_types is None:
        data_types = ["O", "F", "S"]

    # Funktioner og filnavne
    function_map = {
        "O": (fetch_options_data, "option data.csv"),
        "F": (fetch_forward_prices, "forward price.csv"),
        "S": (fetch_stock_returns, "returns and stock price.csv")
    }

    # WRDS-forbindelse åbnes én gang
    # db = wrds.Connection(wrds_username="din_bruger")

    # Dictionary til returnering af dataframes (kun hvis return_df=True)
    data_results = {} if return_df else None

    for data_type in data_types:
        if data_type not in function_map:
            raise ValueError(f"Ugyldig data_type: {data_type}. Brug kun 'O', 'F', eller 'S'.")

        fetch_function, filename = function_map[data_type]
        csv_path = base_dir / filename  # Generer den rigtige filsti

        # Hent data og gem i CSV
        print(f"Henter {data_type}-data og gemmer til: {csv_path}")
        df = fetch_function(db, begdate, enddate, tickers, csv_path)

        if return_df:
            data_results[data_type] = df

    # Luk WRDS-forbindelsen efter alt er hentet
    # db.close()

    return data_results if return_df else None
