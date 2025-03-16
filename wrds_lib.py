import wrds
import os
import pandas as pd
import load_clean_lib as load
from tqdm import tqdm  

# def fetch_options_data_progress(db, begdate, enddate, tickers, csv_path):
#     # db = wrds.Connection(wrds_username=username)

#     secid_list = None  # Default: Hent hele databasen

#     if tickers:
#         if isinstance(tickers, str):
#             tickers = [tickers]

#         secid_df = db.raw_sql(f"""
#             SELECT secid, ticker, cusip, issuer
#             FROM optionm.securd1
#             WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
#         """)

#         secid_list = secid_df["secid"].tolist()

#         # Konverter til SQL-venlig streng
#         if len(secid_list) == 1:
#             secid_str = f"({secid_list[0]})"
#         elif secid_list:
#             secid_str = f"({', '.join(map(str, secid_list))})"
#         else:
#             secid_str = "(NULL)"

#     df_list = []

#     for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
#         print(f"Henter data for år {year}...")

#         query = f"""
#             SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
#                    o.date, o.exdate, o.cp_flag, o.strike_price, 
#                    o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
#                    o.cfadj, ss_flag,
#                    (o.exdate - o.date) AS days_diff  -- Beregn dage direkte i SQL
#             FROM optionm.opprcd{year} o
#             LEFT JOIN optionm.securd1 s ON o.secid = s.secid
#             WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
#               AND (o.exdate - o.date) > 8  
#               AND (o.exdate - o.date) <= 365 
#         """

#         if secid_list:
#             query += f" AND o.secid IN {secid_str}"

#         df_list.append(db.raw_sql(query, date_cols=["date", "exdate"]))

#     df = pd.concat(df_list, ignore_index=True)
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)
    
#     # db.close()
#     return df


def fetch_options_data_progress(db, begdate, enddate, tickers, csv_path):
    """
    Optimeret fetch_options_data_progress med en dynamisk progress bar (tqdm).
    
    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
    """

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    df_list = []

    # Opret progress bar
    with tqdm(total=end_year - start_year, desc="Collecting data", unit="years") as pbar:
        for year in range(start_year, end_year):
            query = f"""
                SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
                       o.date, o.exdate, o.cp_flag, o.strike_price, 
                       o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
                       o.cfadj, ss_flag,
                       (o.exdate - o.date) AS days_diff
                FROM optionm.opprcd{year} o
                LEFT JOIN optionm.securd1 s ON o.secid = s.secid
                WHERE o.date BETWEEN {begdate_sql} AND {enddate_sql}
                  AND (o.exdate - o.date) > 8  
                  AND (o.exdate - o.date) <= 365
            """

            if secid_list:
                query += f" AND o.secid IN {secid_str}"

            df_list.append(db.raw_sql(query, date_cols=["date", "exdate"]))

            # Opdater progress bar
            pbar.update(1)

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)

    print(f"Data collected and saved: {len(df)} rows.")  
    # print(f"Data gemt til {csv_path}")
    return df

def fetch_options_data(db, begdate, enddate, tickers, csv_path):
    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Opret en dynamisk UNION ALL-query direkte i SQL
    table_union_query = " UNION ALL ".join(
        [f"SELECT * FROM optionm.opprcd{year}" for year in range(start_year, end_year)]
    )

    final_query = f"""
        SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
               o.date, o.exdate, o.cp_flag, o.strike_price, 
               o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
               o.cfadj, ss_flag,
               (o.exdate - o.date) AS days_diff
        FROM (
            {table_union_query}  -- Dynamisk UNION ALL af alle relevante årstabeller
        ) AS o
        LEFT JOIN optionm.securd1 AS s
        ON o.secid = s.secid
        WHERE o.date BETWEEN {begdate_sql} AND {enddate_sql}
          AND (o.exdate - o.date) > 8  
          AND (o.exdate - o.date) <= 365
    """

    if secid_list:
        final_query += f" AND o.secid IN {secid_str}"

    # Kør den samlede SQL-query
    try:
        df = db.raw_sql(final_query, date_cols=["date", "exdate"])
        print(f"Data collected and saved: {len(df)} rows.")
    except Exception as e:
        print(f"Error in data import: {e}")
        return None

    # Sortér og gem data
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)

    return df




# def fetch_forward_prices(db, begdate, enddate, tickers, csv_path):
#     # db = wrds.Connection(wrds_username=username)

#     secid_list = None

#     if tickers:
#         if isinstance(tickers, str):
#             tickers = [tickers]

#         secid_df = db.raw_sql(f"""
#             SELECT secid, ticker, cusip, issuer
#             FROM optionm.securd1
#             WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
#         """)

#         secid_list = secid_df["secid"].tolist()

#         # Konverter til SQL-venlig streng
#         if len(secid_list) == 1:
#             secid_str = f"({secid_list[0]})"  # Ingen komma, hvis kun én værdi
#         elif secid_list:
#             secid_str = f"({', '.join(map(str, secid_list))})"  # Normal liste
#         else:
#             secid_str = "(NULL)"  # Ingen secid'er fundet

#     df_list = []

#     for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
#         print(f"Henter data for år {year}...")

#         query = f"""
#             SELECT f.secid, s.ticker, s.cusip, s.issuer, 
#                    f.date, f.expiration, f.amsettlement, f.forwardprice
#             FROM optionm.fwdprd{year} f
#             LEFT JOIN optionm.securd1 s ON f.secid = s.secid
#             WHERE f.date BETWEEN '{begdate}' AND '{enddate}'
#         """

#         if secid_list:
#             query += f" AND f.secid IN {secid_str}"

#         df_list.append(db.raw_sql(query, date_cols=["date", "expiration"]))

#     df = pd.concat(df_list, ignore_index=True)
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)
    
#     # db.close()
#     return df


def fetch_forward_prices_progress(db, begdate, enddate, tickers, csv_path):
    """
    Optimeret fetch_forward_prices med en dynamisk progress bar (tqdm).
    
    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
    """

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    df_list = []

    # Opret progress bar
    with tqdm(total=end_year - start_year, desc="Collecting data", unit="years") as pbar:
        for year in range(start_year, end_year):
            query = f"""
                SELECT f.secid, s.ticker, s.cusip, s.issuer, 
                       f.date, f.expiration, f.amsettlement, f.forwardprice
                FROM optionm.fwdprd{year} f
                LEFT JOIN optionm.securd1 s ON f.secid = s.secid
                WHERE f.date BETWEEN {begdate_sql} AND {enddate_sql}
            """

            if secid_list:
                query += f" AND f.secid IN {secid_str}"

            df_list.append(db.raw_sql(query, date_cols=["date", "expiration"]))

            # Opdater progress bar
            pbar.update(1)

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Data collected and saved: {len(df)} rows.")    
    return df


def fetch_forward_prices(db, begdate, enddate, tickers, csv_path):
    """
    Optimeret fetch_forward_prices-funktion, der henter forward prices-data fra WRDS
    med en stor SQL-forespørgsel ved hjælp af UNION ALL.

    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
    """

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Opret en dynamisk UNION ALL-query direkte i SQL
    table_union_query = " UNION ALL ".join(
        [f"SELECT * FROM optionm.fwdprd{year}" for year in range(start_year, end_year)]
    )

    final_query = f"""
        SELECT f.secid, s.ticker, s.cusip, s.issuer, 
               f.date, f.expiration, f.amsettlement, f.forwardprice
        FROM (
            {table_union_query}  -- Dynamisk UNION ALL af alle relevante årstabeller
        ) AS f
        LEFT JOIN optionm.securd1 AS s
        ON f.secid = s.secid
        WHERE f.date BETWEEN {begdate_sql} AND {enddate_sql}
    """

    if secid_list:
        final_query += f" AND f.secid IN {secid_str}"

    # Kør den samlede SQL-query
    try:
        df = db.raw_sql(final_query, date_cols=["date", "expiration"])
        print(f"Data collected and saved: {len(df)} rows.")
    except Exception as e:
        print(f"Error in data import: {e}")
        return None

    # Sortér og gem data
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)

    return df



# def fetch_stock_returns(db, begdate, enddate, tickers, csv_path):
#     # db = wrds.Connection(wrds_username=username)

#     secid_list = None

#     if tickers:
#         if isinstance(tickers, str):
#             tickers = [tickers]

#         secid_df = db.raw_sql(f"""
#             SELECT secid, ticker, cusip, issuer
#             FROM optionm.securd1
#             WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
#         """)

#         secid_list = secid_df["secid"].tolist()

#         # Konverter til SQL-venlig streng
#         if len(secid_list) == 1:
#             secid_str = f"({secid_list[0]})"  # Ingen komma, hvis kun én værdi
#         elif secid_list:
#             secid_str = f"({', '.join(map(str, secid_list))})"  # Normal liste
#         else:
#             secid_str = "(NULL)"  # Ingen secid'er fundet

#     df_list = []

#     for year in range(max(int(begdate[:4]), 1996), min(int(enddate[:4]) + 1, 2024)):
#         print(f"Henter data for år {year}...")
#         query = f"""
#             SELECT p.secid, s.ticker, s.cusip, s.issuer, 
#                    p.date, p.open, p.close, p.return
#             FROM optionm.secprd{year} p
#             LEFT JOIN optionm.securd1 s ON p.secid = s.secid
#             WHERE p.date BETWEEN '{begdate}' AND '{enddate}'
#         """

#         if secid_list:
#             query += f" AND p.secid IN {secid_str}"

#         df_list.append(db.raw_sql(query, date_cols=["date"]))

#     df = pd.concat(df_list, ignore_index=True)
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)

#     # db.close()
#     return df



def fetch_stock_returns_progress(db, begdate, enddate, tickers, csv_path):
    """
    Optimeret fetch_stock_returns med en dynamisk progress bar (tqdm).
    
    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
    """

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    df_list = []

    # Opret progress bar
    with tqdm(total=end_year - start_year, desc="Collecting data", unit="years") as pbar:
        for year in range(start_year, end_year):
            query = f"""
                SELECT p.secid, s.ticker, s.cusip, s.issuer, 
                       p.date, p.open, p.close, p.return
                FROM optionm.secprd{year} p
                LEFT JOIN optionm.securd1 s ON p.secid = s.secid
                WHERE p.date BETWEEN {begdate_sql} AND {enddate_sql}
            """

            if secid_list:
                query += f" AND p.secid IN {secid_str}"

            df_list.append(db.raw_sql(query, date_cols=["date"]))

            # Opdater progress bar
            pbar.update(1)

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Data collected and saved: {len(df)} rows.") 
    return df



def fetch_stock_returns(db, begdate, enddate, tickers, csv_path):
    """
    Optimeret fetch_stock_returns, der bruger en samlet SQL-forespørgsel
    med UNION ALL for maksimal hastighed.

    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
    """

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # Hent SECID'er for de valgte tickers
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

        if len(secid_list) == 1:
            secid_str = f"({secid_list[0]})"
        elif secid_list:
            secid_str = f"({', '.join(map(str, secid_list))})"
        else:
            secid_str = "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Opret en dynamisk UNION ALL-query direkte i SQL
    table_union_query = " UNION ALL ".join(
        [f"SELECT * FROM optionm.secprd{year}" for year in range(start_year, end_year)]
    )

    final_query = f"""
        SELECT p.secid, s.ticker, s.cusip, s.issuer, 
               p.date, p.open, p.close, p.return
        FROM (
            {table_union_query}  -- Dynamisk UNION ALL af alle relevante årstabeller
        ) AS p
        LEFT JOIN optionm.securd1 AS s
        ON p.secid = s.secid
        WHERE p.date BETWEEN {begdate_sql} AND {enddate_sql}
    """

    if secid_list:
        final_query += f" AND p.secid IN {secid_str}"

    # Kør den samlede SQL-query
    try:
        df = db.raw_sql(final_query, date_cols=["date"])
        print(f"Data collected and saved: {len(df)} rows.")   
    except Exception as e:
        print(f"Error in data import: {e}")
        return None

    # Sortér og gem data
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(csv_path, index=False)

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


def fetch_wrds_data(db, profile, folder_name, begdate, enddate, tickers=None, data_types=["O", "F", "S"], return_df=False, progress=False):
    """
    Henter WRDS data (options, forward prices, stock returns) og gemmer i en mappe.
    
    Args:
        db: WRDS databaseforbindelse
        profile (str): Brugerprofil til mappehåndtering
        folder_name (str): Navn på mappen til at gemme data
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list, optional): Liste over tickers
        data_types (list, optional): Liste af datatyper ("O", "F", "S")
        return_df (bool, optional): Returnerer dataframes hvis True
        progress (bool, optional): Bruger progress-bar version af funktionerne hvis True

    Returns:
        dict: Dataframes som dictionary med keys "O", "F", "S" hvis return_df=True, ellers None
    """

    # Find base directory for brugerprofil
    base_dir = load.dirs(profile)["OptionMetrics"] / folder_name

    # Tjek om mappen allerede eksisterer og indeholder filer
    if base_dir.exists() and any(base_dir.iterdir()):
        print(f"Folder '{folder_name}' already exists. Aborting.")
        return None

    # Opret mappen, hvis den ikke findes
    base_dir.mkdir(parents=True, exist_ok=True)

    # Mapping af funktioner og filnavne
    function_map = {
        "O": (fetch_options_data_progress if progress else fetch_options_data, "option data.csv"),
        "F": (fetch_forward_prices_progress if progress else fetch_forward_prices, "forward price.csv"),
        "S": (fetch_stock_returns_progress if progress else fetch_stock_returns, "returns and stock price.csv")
    }

    # Initialiser dictionary til dataframes
    data_results = {}

    # Hent de valgte datatyper
    for data_type in data_types:
        if data_type not in function_map:
            raise ValueError(f"data_type: {data_type} is not accepted. Use only 'O', 'F', 'S'.")

        fetch_function, filename = function_map[data_type]
        csv_path = base_dir / filename  # Generer den rigtige filsti

        print(f"Importing {data_type}-data and saving to: {csv_path}")
        df = fetch_function(db, begdate, enddate, tickers, csv_path)

        # Hvis return_df=True, gem dataframe i dictionary
        if return_df:
            data_results[data_type] = df

    return data_results if return_df else None



