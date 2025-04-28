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


# def fetch_options_data_progress(db, begdate, enddate, tickers, csv_path):
#     """
#     Optimeret fetch_options_data_progress med en dynamisk progress bar (tqdm).
    
#     Args:
#         db: WRDS databaseforbindelse
#         begdate (str): Startdato i 'YYYY-MM-DD' format
#         enddate (str): Slutdato i 'YYYY-MM-DD' format
#         tickers (list): Liste af tickers
#         csv_path (str): Sti til output CSV-fil
#     """

#     # Konverter datoer til SQL-format
#     begdate_sql = f"'{begdate}'"
#     enddate_sql = f"'{enddate}'"

#     # Hent SECID'er for de valgte tickers
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

#         if len(secid_list) == 1:
#             secid_str = f"({secid_list[0]})"
#         elif secid_list:
#             secid_str = f"({', '.join(map(str, secid_list))})"
#         else:
#             secid_str = "(NULL)"

#     # Definer start- og slutår baseret på begdate og enddate
#     start_year = max(int(begdate[:4]), 1996)
#     end_year = min(int(enddate[:4]) + 1, 2024)

#     df_list = []

#     # Opret progress bar
#     with tqdm(total=end_year - start_year, desc="Collecting data", unit="years") as pbar:
#         for year in range(start_year, end_year):
#             query = f"""
#                 SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
#                        o.date, o.exdate, o.cp_flag, o.strike_price, 
#                        o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
#                        o.cfadj, ss_flag,
#                        (o.exdate - o.date) AS days_diff
#                 FROM optionm.opprcd{year} o
#                 LEFT JOIN optionm.securd1 s ON o.secid = s.secid
#                 WHERE o.date BETWEEN {begdate_sql} AND {enddate_sql}
#                   AND (o.exdate - o.date) > 8  
#                   AND (o.exdate - o.date) <= 365
#             """

#             if secid_list:
#                 query += f" AND o.secid IN {secid_str}"

#             df_list.append(db.raw_sql(query, date_cols=["date", "exdate"]))

#             # Opdater progress bar
#             pbar.update(1)

#     df = pd.concat(df_list, ignore_index=True)
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)

#     print(f"Data collected and saved: {len(df)} rows.")  
#     # print(f"Data gemt til {csv_path}")
#     return df



def fetch_options_data_progress(db, begdate, enddate, tickers, csv_path, chunk_size, return_df):
    """
    Optimeret fetch_options_data_progress med chunking og en dynamisk progress bar (tqdm).
    
    Args:
        db: WRDS databaseforbindelse
        begdate (str): Startdato i 'YYYY-MM-DD' format
        enddate (str): Slutdato i 'YYYY-MM-DD' format
        tickers (list): Liste af tickers
        csv_path (str): Sti til output CSV-fil
        chunk_size (int): Størrelse af hvert chunk, der skal skrives til CSV-filen
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

    # Opret progress bar
    with tqdm(total=end_year - start_year, desc="Collecting data", unit="years") as pbar:
        first_chunk = True
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
                  AND (o.exdate - o.date) <= 365
            """ # AND (o.exdate - o.date) > 8  

            if secid_list:
                query += f" AND o.secid IN {secid_str}"

            # Hent data i chunks ved at bruge LIMIT og OFFSET
            offset = 0
            while True:
                chunk_query = query + f" LIMIT {chunk_size} OFFSET {offset}"
                chunk = db.raw_sql(chunk_query, date_cols=["date", "exdate"])

                if chunk.empty:
                    break  # Stop, hvis der ikke er flere data
                
                # chunk["best_bid"] = pd.to_numeric(chunk["best_bid"], errors="coerce")
                # chunk["best_offer"] = pd.to_numeric(chunk["best_offer"], errors="coerce")
                # chunk["ticker"] = chunk["ticker"].astype(str)  # Sikrer at ticker altid er string

                if first_chunk:
                    # Skriv header kun for den første chunk
                    chunk.to_csv(csv_path, index=False, mode='w', sep=',', decimal='.')
                    first_chunk = False
                else:
                    # Undlad at skrive header for efterfølgende chunks
                    chunk.to_csv(csv_path, index=False, mode='a', header=False, sep=',', decimal='.')

                offset += chunk_size

            # Opdater progress bar
            pbar.update(1)
    
    if return_df == False: print("Data collected and saved")
    if return_df: 
        df = pd.read_csv(csv_path)
        print(f"Data collected and saved: {len(df)} rows.")
        return df
    return


# def fetch_options_data_multiplefiles(db, begdate, enddate, tickers, output_dir,
#                                      data_frq="Y", return_df=False):
#     """
#     Hent optionsdata for hvert år separat og gem månedlige eller årlige filer
#     uden at akkumulere hele datasættet i hukommelsen.
#     """
#     import pandas as pd
#     from pathlib import Path

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Forbered ticker-liste til SQL
#     if isinstance(tickers, str):
#         tickers = [tickers]
#     tickers_sql = ", ".join(f"'{t}'" for t in tickers)

#     # Hent SECIDs
#     secid_df = db.raw_sql(f"""
#         SELECT secid
#         FROM optionm.securd1
#         WHERE ticker IN ({tickers_sql})
#     """)
#     secids = secid_df["secid"].tolist()
#     secid_sql = f"({','.join(map(str, secids))})" if secids else "(NULL)"

#     # Definer start- og slutår baseret på begdate og enddate
#     start_year = max(int(begdate[:4]), 1996)
#     end_year = min(int(enddate[:4]) + 1, 2024)

#     all_years = [] if return_df else None

#     # Brug tqdm over år‐intervallet
#     for year in tqdm(range(start_year, end_year), desc="Collecting data", unit="years"):
#         # Hent data for dette år
#         sql = f"""
#             SELECT o.secid, s.ticker, o.optionid,
#                 o.date, o.exdate, o.cp_flag, o.strike_price,
#                 o.best_bid, o.best_offer, o.impl_volatility,
#                 o.volume, o.open_interest
#             FROM optionm.opprcd{year} o
#             JOIN optionm.securd1 s ON o.secid = s.secid
#             WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
#             AND o.secid IN {secid_sql}
#             AND (o.exdate - o.date) <= 365
#         """
#         df_year = db.raw_sql(sql, date_cols=["date", "exdate"])
#         if df_year.empty:
#             continue

#         # Split og gem pr. periode (måned eller år)
#         df_year["date"] = pd.to_datetime(df_year["date"])
#         df_year["period"] = df_year["date"].dt.to_period(data_frq).astype(str)

#         for period, grp in df_year.groupby("period"):
#             fn = output_dir / f"option data {period}.csv"
#             grp.to_csv(fn, index=False)

#         if return_df:
#             all_years.append(df_year)

#         # frigør hukommelse før næste år
#         del df_year

#     if return_df:
#         return pd.concat(all_years, ignore_index=True)
#     return None




def fetch_options_data_multiplefiles(db, begdate, enddate, tickers, output_dir,
                                     data_frq="Y", return_df=False, chunk_size=100_000):
    """
    Hent optionsdata for hvert år separat og gem månedlige eller årlige filer
    uden at akkumulere hele datasættet i hukommelsen. Nu med chunking for store år.
    """
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forbered ticker-liste til SQL
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers_sql = ", ".join(f"'{t}'" for t in tickers)

    # Hent SECIDs
    secid_df = db.raw_sql(f"""
        SELECT secid
        FROM optionm.securd1
        WHERE ticker IN ({tickers_sql})
    """)
    secids = secid_df["secid"].tolist()
    secid_sql = f"({','.join(map(str, secids))})" if secids else "(NULL)"

    # Definer start- og slutår baseret på begdate og enddate
    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    all_years = [] if return_df else None

    # Brug tqdm over år‐intervallet
    for year in tqdm(range(start_year, end_year), desc="Collecting data", unit="years"):
        offset = 0
        first_chunk = True
        df_year_chunks = []  # midlertidig liste til at samle chunks hvis return_df = True

        while True:
            # Hent data i chunks
            # sql_chunk = f"""
            #     SELECT o.secid, s.ticker, o.optionid,
            #         o.date, o.exdate, o.cp_flag, o.strike_price,
            #         o.best_bid, o.best_offer, o.impl_volatility,
            #         o.volume, o.open_interest
            #     FROM optionm.opprcd{year} o
            #     JOIN optionm.securd1 s ON o.secid = s.secid
            #     WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
            #     AND o.secid IN {secid_sql}
            #     AND (o.exdate - o.date) <= 365
            #     LIMIT {chunk_size} OFFSET {offset}
            # """
            # sql_chunk = f"""
            #     SELECT o.secid, s.ticker, o.optionid,
            #         o.date, o.exdate, o.cp_flag, o.strike_price,
            #         o.best_bid, o.best_offer, o.impl_volatility,
            #         o.volume, o.open_interest
            #     FROM optionm.opprcd{year} o
            #     JOIN optionm.securd1 s ON o.secid = s.secid
            #     WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
            #     AND o.secid IN {secid_sql}
            #     AND (o.exdate - o.date) <= 365
            #     AND NOT ( (s.issue_type IS NULL OR s.issue_type IN ('', '0')) 
            #                 AND s.exchange_d = 0 )
            #     LIMIT {chunk_size} OFFSET {offset}
            # """

            sql_chunk = f"""
                SELECT o.secid, s.ticker, o.optionid,
                    o.date, o.exdate, o.cp_flag, o.strike_price,
                    o.best_bid, o.best_offer, o.impl_volatility,
                    o.volume, o.open_interest, s.issue_type, s.exchange_d
                FROM optionm.opprcd{year} o
                JOIN optionm.securd1 s ON o.secid = s.secid
                WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
                AND o.secid IN {secid_sql}
                AND (o.exdate - o.date) <= 365
                AND s.issue_type IN ('0')
                AND s.exchange_d <> 0
                LIMIT {chunk_size} OFFSET {offset}
            """ 
         
            # AND (
            #     s.issue_type IN ('0', 'A', '7', 'F', 'S', 'U')
            #     OR s.issue_type = '%'
            # )

            chunk = db.raw_sql(sql_chunk, date_cols=["date", "exdate"])

            if chunk.empty:
                break  # ingen flere data i dette år

            # Forbered data
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["period"] = chunk["date"].dt.to_period(data_frq).astype(str)

            # Gem hver chunk direkte pr. periode
            for period, grp in chunk.groupby("period"):
                fn = output_dir / f"option data {period}.csv"
                if first_chunk and not fn.exists():
                    # Første gang: skriv med header
                    grp.to_csv(fn, index=False, mode='w')
                else:
                    # Senere: tilføj uden header
                    grp.to_csv(fn, index=False, mode='a', header=False)

            if return_df:
                df_year_chunks.append(chunk)

            first_chunk = False
            offset += chunk_size  # næste offset

        if return_df and df_year_chunks:
            all_years.append(pd.concat(df_year_chunks, ignore_index=True))

    if return_df and all_years:
        return pd.concat(all_years, ignore_index=True)
    return None




# def fetch_options_data(db, begdate, enddate, tickers, csv_path):
#     # Konverter datoer til SQL-format
#     begdate_sql = f"'{begdate}'"
#     enddate_sql = f"'{enddate}'"

#     # Hent SECID'er for de valgte tickers
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

#         if len(secid_list) == 1:
#             secid_str = f"({secid_list[0]})"
#         elif secid_list:
#             secid_str = f"({', '.join(map(str, secid_list))})"
#         else:
#             secid_str = "(NULL)"

#     # Definer start- og slutår baseret på begdate og enddate
#     start_year = max(int(begdate[:4]), 1996)
#     end_year = min(int(enddate[:4]) + 1, 2024)

#     # Opret en dynamisk UNION ALL-query direkte i SQL
#     table_union_query = " UNION ALL ".join(
#         [f"SELECT * FROM optionm.opprcd{year}" for year in range(start_year, end_year)]
#     )

#     final_query = f"""
#         SELECT o.secid, s.ticker, o.optionid, s.cusip, s.issuer, 
#                o.date, o.exdate, o.cp_flag, o.strike_price, 
#                o.best_bid, o.best_offer, o.impl_volatility, o.volume, o.open_interest,
#                o.cfadj, ss_flag,
#                (o.exdate - o.date) AS days_diff
#         FROM (
#             {table_union_query}  -- Dynamisk UNION ALL af alle relevante årstabeller
#         ) AS o
#         LEFT JOIN optionm.securd1 AS s
#         ON o.secid = s.secid
#         WHERE o.date BETWEEN {begdate_sql} AND {enddate_sql}
#           AND (o.exdate - o.date) > 8  
#           AND (o.exdate - o.date) <= 365
#     """

#     if secid_list:
#         final_query += f" AND o.secid IN {secid_str}"

#     # Kør den samlede SQL-query
#     try:
#         df = db.raw_sql(final_query, date_cols=["date", "exdate"])
#         print(f"Data collected and saved: {len(df)} rows.")
#     except Exception as e:
#         print(f"Error in data import: {e}")
#         return None

#     # Sortér og gem data
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)

#     return df



def fetch_options_data(db, begdate, enddate, tickers, csv_path, chunk_size, return_df):
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

    # Hent data i chunks
    offset = 0
    first_chunk = True
    while True:
        chunk_query = final_query + f" LIMIT {chunk_size} OFFSET {offset}"
        chunk = db.raw_sql(chunk_query, date_cols=["date", "exdate"])

        if chunk.empty:
            break  # Stop, hvis der ikke er flere data

        if first_chunk:
            # Skriv header kun for den første chunk
            chunk.to_csv(csv_path, index=False, mode='w')
            first_chunk = False
        else:
            # Undlad at skrive header for efterfølgende chunks
            chunk.to_csv(csv_path, index=False, mode='a', header=False)

        offset += chunk_size
    if return_df == False: print("Data collected and saved")
    if return_df: 
        df = pd.read_csv(csv_path)
        print(f"Data collected and saved: {len(df)} rows.")
        return df
    return



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

    df_list = [d for d in df_list if not d.empty]

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



def fetch_stock_returns_progress(db, begdate, enddate, tickers, csv_path): ### OM VERSION
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
    with tqdm(total = end_year - start_year, desc="Collecting data", unit="years") as pbar:
        for year in range(start_year, end_year):
            query = f"""
                SELECT p.secid, s.ticker, s.cusip, s.issuer, 
                       p.date, p.open, p.close, p.return, s.issue_type, s.exchange_d
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




# def fetch_stock_returns_progress(db, begdate, enddate, tickers, csv_path): ### CRSP VERSION
#     """
#     Fetch stock returns fra CRSP via WRDS, med progress bar.

#     Args:
#         db: WRDS databaseforbindelse
#         begdate (str): Startdato i 'YYYY-MM-DD' format
#         enddate (str): Slutdato i 'YYYY-MM-DD' format
#         tickers (list): Liste af tickers
#         csv_path (str): Sti til output CSV-fil
#     """

#     # expand enddate for RV calculation 
#     enddate_extended = (pd.to_datetime(enddate) + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

#     # Konverter til SQL-format
#     begdate_sql = f"'{begdate}'"
#     enddate_sql = f"'{enddate_extended}'"

#     # Hent PERMNO for de valgte tickers
#     permno_list = None
#     if tickers:
#         if isinstance(tickers, str):
#             tickers = [tickers]

#         ticker_df = db.raw_sql(f"""
#             SELECT permno, ticker, cusip, comnam as issuer
#             FROM crsp.dsenames
#             WHERE ticker IN ({', '.join(f"'{t}'" for t in tickers)})
#         """)

#         permno_list = ticker_df["permno"].tolist()

#         if len(permno_list) == 1:
#             permno_str = f"({permno_list[0]})"
#         elif permno_list:
#             permno_str = f"({', '.join(map(str, permno_list))})"
#         else:
#             permno_str = "(NULL)"

#     # Hent al data på én gang fra CRSP
#     query = f"""
#         SELECT d.permno, n.ticker, n.cusip, n.comnam as issuer,
#                d.date, d.bidlo as open, d.prc as close, d.ret as return
#         FROM crsp.dsf d
#         LEFT JOIN crsp.dsenames n ON d.permno = n.permno
#         WHERE d.date BETWEEN {begdate_sql} AND {enddate_sql}
#           AND n.namedt <= d.date AND d.date <= n.nameendt
#     """

#     if permno_list:
#         query += f" AND d.permno IN {permno_str}"

#     df = db.raw_sql(query, date_cols=["date"])

#     #remove related tickers
#     df = df[df["ticker"].isin(tickers)]
#     # Sorter og gem
#     df.sort_values(by=["ticker", "date"], inplace=True)
#     df.to_csv(csv_path, index=False)

#     print(f"Data collected and saved: {len(df)} rows.")
#     return df







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




def fetch_zerocoupons(db, begdate, enddate, csv_path):

    # Konverter datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    final_query = f"""
        SELECT *
        FROM optionm.zerocd
        WHERE date BETWEEN {begdate_sql} AND {enddate_sql}
        AND days <= 500
    """

    # Kør den samlede SQL-query
    try:
        df = db.raw_sql(final_query, date_cols=["date"])
        print(f"Data collected and saved: {len(df)} rows.")   
    except Exception as e:
        print(f"Error in data import: {e}")
        return None

    # Sortér og gem data
    df.sort_values(by=["date", "days"], inplace=True)
    df.to_csv(csv_path, index=False)

    return df


def fetch_dividends_progress(db, begdate, enddate, tickers, csv_path):
    """
    Henter distributions-/udbyttedata fra OptionMetrics' distrproj-tabeller i en 
    periode mellem begdate og enddate for de givne tickers.

    Antagelser:
      - 'optionm.distrprojYYYY' er tabellerne med exdate + 3 andre kolonner.
      - 'exdate' er den dato-kolonne, vi vil filtrere på.
      - 'optionm.securd1' indeholder secid, ticker, cusip, issuer.
      - 'db' er en WRDS-forbindelse (f.eks. wrds.Connection()).

    Parametre:
      db (wrds.Connection): Databaseforbindelse til WRDS.
      begdate (str): Startdato i format 'YYYY-MM-DD'.
      enddate (str): Slutdato i format 'YYYY-MM-DD'.
      tickers (list eller str): En eller flere tickers at filtrere på.
      csv_path (str): Sti til output-CSV.
    
    Returnerer:
      pd.DataFrame med de hentede distributionsdata, inkl. ticker, cusip, issuer.
    """

    # Konverter input-datoer til SQL-format
    begdate_sql = f"'{begdate}'"
    enddate_sql = f"'{enddate}'"

    # 1) Hent de relevante secid'er ud fra tickers
    secid_list = None
    if tickers:
        # Håndtér, hvis tickers er en enkelt streng
        if isinstance(tickers, str):
            tickers = [tickers]

        # Hent secid, ticker, cusip, issuer fra securd1
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
    else:
        # Hvis ingen tickers, kan du selv vælge logik:
        secid_str = "(NULL)"  # eller hent alt, men det kan være kæmpestort

    # 2) Definer start- og slutår (ligesom i fetch_stock_returns_progress)
    start_year = max(int(begdate[:4]), 1996)  # OptionMetrics startår
    end_year = min(int(enddate[:4]) + 1, 2024)  # Sæt selv grænse for slutår

    df_list = []

    # 3) Loop over hvert år og hent data
    with tqdm(total=end_year - start_year, desc="Collecting dividend data", unit="years") as pbar:
        for year in range(start_year, end_year):
            # Byg query. Vi antager her, at tabellen hedder "optionm.distrprojYYYY"
            # og at kolonnen med dato hedder "exdate"
            query = f"""
                SELECT p.*, s.ticker, s.cusip, s.issuer
                FROM optionm.distrproj{year} p
                LEFT JOIN optionm.securd1 s ON p.secid = s.secid
                WHERE p.exdate BETWEEN {begdate_sql} AND {enddate_sql}
            """

            if secid_list:
                query += f" AND p.secid IN {secid_str}"

            # Hent data via WRDS
            # Vi specificerer date_cols=["exdate"], så WRDS parser exdate som en dato
            df_year = db.raw_sql(query, date_cols=["exdate"])
            df_list.append(df_year)

            pbar.update(1)

    # 4) Saml alt i én DataFrame
    df = pd.concat(df_list, ignore_index=True)
    # Sortér for god ordens skyld
    df.sort_values(by=["ticker", "exdate"], inplace=True)

    # 5) Gem til CSV
    df.to_csv(csv_path, index=False)

    print(f"Data collected and saved: {len(df)} rows to {csv_path}.")
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


# def fetch_wrds_data(db, folder_name, begdate, enddate, save_per = None, tickers=None, data_types=["O", "F", "S", "Z"], return_df=False, progress=False, chunk_size = 1500000):
#     """
#     Henter WRDS data (options, forward prices, stock returns) og gemmer i en mappe.
    
#     Args:
#         db: WRDS databaseforbindelse
#         profile (str): Brugerprofil til mappehåndtering
#         folder_name (str): Navn på mappen til at gemme data
#         begdate (str): Startdato i 'YYYY-MM-DD' format
#         enddate (str): Slutdato i 'YYYY-MM-DD' format
#         tickers (list, optional): Liste over tickers
#         data_types (list, optional): Liste af datatyper ("O", "F", "S")
#         return_df (bool, optional): Returnerer dataframes hvis True
#         progress (bool, optional): Bruger progress-bar version af funktionerne hvis True

#     Returns:
#         dict: Dataframes som dictionary med keys "O", "F", "S" hvis return_df=True, ellers None
#     """

#     # Find base directory for brugerprofil
#     base_dir = load.dirs()["OptionMetrics"] / folder_name
#     # base_dir = load.Option_metrics_path_from_profile(profile) / folder_name # todo: use this function instead, haven't tested


#     # Tjek om mappen allerede eksisterer og indeholder filer
#     if base_dir.exists() and any(base_dir.iterdir()):
#         print(f"Folder '{folder_name}' already exists. Aborting.")
#         return None

#     # Opret mappen, hvis den ikke findes
#     base_dir.mkdir(parents=True, exist_ok=True)

#     # Mapping af funktioner og filnavne
#     function_map = {
#         "O": (fetch_options_data_progress if progress else fetch_options_data, "option data.csv"),
#         "F": (fetch_forward_prices_progress if progress else fetch_forward_prices, "forward price.csv"),
#         "S": (fetch_stock_returns_progress if progress else fetch_stock_returns, "returns and stock price.csv"),
#         "Z": (fetch_zerocoupons, "ZC yield curve.csv"),
#         "D": (fetch_dividends_progress, "Dividends.csv")
#     }

#     # Initialiser dictionary til dataframes
#     data_results = {}

#     # Hent de valgte datatyper
#     for data_type in data_types:
#         if data_type not in function_map:
#             raise ValueError(f"data_type: {data_type} is not accepted. Use only 'O', 'F', 'S'.")

#         fetch_function, filename = function_map[data_type]
#         csv_path = base_dir / filename  # Generer den rigtige filsti

#         print(f"Importing {data_type}-data and saving to: {csv_path}")
#         if data_type == "O":
#             df = fetch_function(db, begdate, enddate, tickers, csv_path, chunk_size=chunk_size, return_df=return_df)
#         elif data_type == "Z":
#             df = fetch_function(db, begdate, enddate, csv_path)
#         else:
#             df = fetch_function(db, begdate, enddate, tickers, csv_path)
#         # Hvis return_df=True, gem dataframe i dictionary
#         if return_df:
#             data_results[data_type] = df

#     return data_results if return_df else None


def fetch_wrds_data(db, folder_name, begdate, enddate,
                    save_per=None,
                    tickers=None,
                    data_types=["O", "F", "S", "Z"],
                    return_df=False,
                    progress=False,
                    chunk_size=1_500_000):
    """
    Henter WRDS-data og gemmer i en mappe. 
    Hvis save_per="M" eller "Y", splittes O-data per måned/år i output-mappen.
    """

    from pathlib import Path

    base_dir = load.dirs()["OptionMetrics"] / folder_name
    if base_dir.exists() and any(base_dir.iterdir()):
        print(f"Folder '{folder_name}' already exists. Aborting.")
        return None
    base_dir.mkdir(parents=True, exist_ok=True)

    # Gå datatyper igennem
    results = {}

    for data_type in data_types:
        if data_type == "O":
            if save_per in ("M", "Y"):
                print(f"Importing O-data split per '{save_per}' into folder: {base_dir}")
                df = fetch_options_data_multiplefiles(
                    db=db,
                    begdate=begdate,
                    enddate=enddate,
                    tickers=tickers,
                    output_dir=base_dir / "option data",
                    data_frq=save_per,
                    return_df=return_df,
                    chunk_size=chunk_size
                )
            else:
                csv_path = base_dir / "option data.csv"
                print(f"Importing O-data into single file: {csv_path}")
                df = (fetch_options_data_progress if progress else fetch_options_data)(
                    db=db,
                    begdate=begdate,
                    enddate=enddate,
                    tickers=tickers,
                    csv_path=csv_path,
                    chunk_size=chunk_size,
                    return_df=return_df
                )

        elif data_type == "F":
            csv_path = base_dir / "forward price.csv"
            print(f"Importing F-data into: {csv_path}")
            df = (fetch_forward_prices_progress if progress else fetch_forward_prices)(
                db=db, begdate=begdate, enddate=enddate,
                tickers=tickers, csv_path=csv_path
            )

        elif data_type == "S":
            csv_path = base_dir / "returns and stock price.csv"
            print(f"Importing S-data into: {csv_path}")
            df = (fetch_stock_returns_progress if progress else fetch_stock_returns)(
                db=db, begdate=begdate, enddate=enddate,
                tickers=tickers, csv_path=csv_path
            )

        elif data_type == "Z":
            csv_path = base_dir / "ZC yield curve.csv"
            print(f"Importing Z-data into: {csv_path}")
            df = fetch_zerocoupons(db=db, begdate=begdate, enddate=enddate, csv_path=csv_path)

        elif data_type == "D":
            csv_path = base_dir / "Dividends.csv"
            print(f"Importing D-data into: {csv_path}")
            df = fetch_dividends_progress(db=db, begdate=begdate, enddate=enddate, tickers=tickers, csv_path=csv_path)

        else:
            raise ValueError(f"data_type '{data_type}' ikke understøttet.")

        if return_df:
            results[data_type] = df

    return results if return_df else None
