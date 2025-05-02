import wrds
import os
import pandas as pd
import load_clean_lib as load
from tqdm import tqdm  
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta  # ← vigtigt!



def fetch_options_data_per_ticker(db, begdate, enddate, tickers, base_dir, chunk_size=100_000):

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Forbered samlet view af alle år
    union_tables = " \nUNION ALL\n ".join(
        f"SELECT * FROM optionm.opprcd{year}" for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS o"

    with tqdm(tickers, desc="Importing option data", unit="ticker") as pbar:
        for ticker in pbar:
            pbar.set_postfix({"ticker": ticker})

            # Find SECIDs for ticker
            secid_df = db.raw_sql(f"""
                SELECT secid, issuer
                FROM optionm.securd1
                WHERE ticker = '{ticker}'
            """)

            if secid_df.empty: # continue if no tickers found..
                continue  

            secids = secid_df["secid"].tolist()
            secid_sql = f"({','.join(map(str, secids))})"

            offset = 0
            first_chunk = True

            # Forbered output path
            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"option data.csv"

            while True:
                sql_chunk = f"""
                    SELECT o.secid, s.ticker, s.issuer, o.optionid,
                        o.date, o.exdate, o.cp_flag, o.strike_price,
                        o.best_bid, o.best_offer, o.impl_volatility,
                        o.volume, o.open_interest, s.issue_type, s.exchange_d, o.ss_flag
                    FROM {base_table}
                    JOIN optionm.securd1 s ON o.secid = s.secid
                    WHERE o.date BETWEEN '{begdate}' AND '{enddate}'
                    AND o.secid IN {secid_sql}
                    AND (o.exdate - o.date) <= 365
                    AND NOT (s.issue_type = '' OR s.exchange_d = 0)
                    LIMIT {chunk_size} OFFSET {offset}
                """

                chunk = db.raw_sql(sql_chunk, date_cols=["date", "exdate"])

                if chunk.empty:
                    break  # Færdig med denne ticker

                # Skriv chunk til fil
                if first_chunk:
                    chunk.to_csv(output_file, index=False, mode='w')
                else:
                    chunk.to_csv(output_file, index=False, mode='a', header=False)

                first_chunk = False
                offset += chunk_size


    return None



def fetch_options_data_per_ticker_days_iv_flag(
    db,
    begdate,
    enddate,
    tickers,
    base_dir,
    chunk_days=30,
    rel_diff_from_median=1
):

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    union_tables = " \nUNION ALL\n ".join(
        f"SELECT * FROM optionm.opprcd{year}" for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS o"

    begdate_dt = pd.to_datetime(begdate)
    enddate_dt = pd.to_datetime(enddate)

    with tqdm(tickers, desc="Importing option data", unit="ticker") as pbar:
        for ticker in pbar:
            pbar.set_postfix({"ticker": ticker})

            secid_df = db.raw_sql(f"""
                SELECT secid, issuer
                FROM optionm.securd1
                WHERE ticker = '{ticker}'
            """)

            if secid_df.empty:
                continue

            secids = secid_df["secid"].tolist()
            secid_sql = f"({','.join(map(str, secids))})"

            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"option data.csv"

            current_start = begdate_dt
            first_chunk = True

            while current_start < enddate_dt:
                current_end = min(current_start + timedelta(days=chunk_days), enddate_dt)
                date_start_str = current_start.strftime("%Y-%m-%d")
                date_end_str = current_end.strftime("%Y-%m-%d")

                # sql_chunk = f"""
                #     WITH base AS (
                #         SELECT o.secid, s.ticker, s.issuer, o.optionid,
                #             o.date, o.exdate, o.cp_flag, o.strike_price,
                #             o.best_bid, o.best_offer, o.impl_volatility,
                #             o.volume, o.open_interest, s.issue_type, s.exchange_d, o.ss_flag
                #         FROM {base_table}
                #         JOIN optionm.securd1 s ON o.secid = s.secid
                #         WHERE o.date >= '{date_start_str}'
                #         AND o.date <  '{date_end_str}'
                #         AND o.secid IN {secid_sql}
                #         AND (o.exdate - o.date) <= 365
                #         AND NOT (s.issue_type = '' OR s.exchange_d = 0)
                #     ),
                #     medians AS (
                #         SELECT date, exdate, cp_flag,
                #             PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY impl_volatility) AS median_iv
                #         FROM base
                #         GROUP BY date, exdate, cp_flag
                #     )
                #     SELECT b.*
                #     FROM base b
                #     JOIN medians m
                #     ON b.date = m.date
                #     AND b.exdate = m.exdate
                #     AND b.cp_flag = m.cp_flag
                #     WHERE (ABS(b.impl_volatility - m.median_iv) <= {abs_diff_from_median}
                #         OR b.impl_volatility IS NULL)
                # """

                sql_chunk = f"""
                    WITH base AS (
                        SELECT o.secid, s.ticker, s.issuer, o.optionid,
                            o.date, o.exdate, o.cp_flag, o.strike_price,
                            o.best_bid, o.best_offer, o.impl_volatility,
                            o.volume, o.open_interest, s.issue_type, s.exchange_d, o.ss_flag
                        FROM {base_table}
                        JOIN optionm.securd1 s ON o.secid = s.secid
                        WHERE o.date >= '{date_start_str}'
                        AND o.date <  '{date_end_str}'
                        AND o.secid IN {secid_sql}
                        AND (o.exdate - o.date) <= 365
                        AND NOT (s.issue_type = '' OR s.exchange_d = 0)
                    ),
                    medians AS (
                        SELECT date, exdate, cp_flag,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY impl_volatility) AS median_iv
                        FROM base
                        GROUP BY date, exdate, cp_flag
                    )
                    SELECT b.*
                    FROM base b
                    JOIN medians m
                    ON b.date = m.date
                    AND b.exdate = m.exdate
                    AND b.cp_flag = m.cp_flag
                    WHERE
                    (
                        (
                            m.median_iv > 0
                            AND ABS(b.impl_volatility - m.median_iv) / m.median_iv <= {rel_diff_from_median}
                        )
                        OR ABS(b.impl_volatility - m.median_iv) <= 0.25
                        OR b.impl_volatility IS NULL
                    )
                """


                # sql_chunk = f"""
                #     WITH base AS (
                #         SELECT o.secid, s.ticker, s.issuer, o.optionid,
                #             o.date, o.exdate, o.cp_flag, o.strike_price,
                #             o.best_bid, o.best_offer, o.impl_volatility,
                #             o.volume, o.open_interest, s.issue_type, s.exchange_d, o.ss_flag
                #         FROM {base_table}
                #         JOIN optionm.securd1 s ON o.secid = s.secid
                #         WHERE o.date >= '{date_start_str}'
                #         AND o.date <  '{date_end_str}'
                #         AND o.secid IN {secid_sql}
                #         AND (o.exdate - o.date) <= 365
                #         AND NOT (s.issue_type = '' OR s.exchange_d = 0)
                #     ),
                #     medians AS (
                #         SELECT date, exdate, cp_flag,
                #             PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY impl_volatility) AS median_iv
                #         FROM base
                #         GROUP BY date, exdate, cp_flag
                #     )
                #     SELECT b.*
                #     FROM base b
                #     JOIN medians m
                #     ON b.date = m.date
                #     AND b.exdate = m.exdate
                #     AND b.cp_flag = m.cp_flag
                #     WHERE (
                #         CASE 
                #             WHEN m.median_iv = 0 THEN FALSE
                #             ELSE ABS(b.impl_volatility - m.median_iv) / m.median_iv <= {rel_diff_from_median}
                #         END
                #         OR b.impl_volatility IS NULL
                #     )
                # """


                chunk = db.raw_sql(sql_chunk, date_cols=["date", "exdate"])

                if not chunk.empty:
                    if first_chunk:
                        chunk.to_csv(output_file, index=False, mode='w')
                        first_chunk = False
                    else:
                        chunk.to_csv(output_file, index=False, mode='a', header=False)

                current_start = current_end

    return None






def fetch_forward_prices_per_ticker(db, begdate, enddate, tickers, base_dir, chunk_size=100_000):

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Forbered samlet view af alle år
    union_tables = " \nUNION ALL\n ".join(
        f"SELECT * FROM optionm.fwdprd{year}" for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS f"

    with tqdm(tickers, desc="Importing forward prices", unit="ticker") as pbar:
        for ticker in pbar:
            pbar.set_postfix({"ticker": ticker})

            # Find SECIDs for ticker
            secid_df = db.raw_sql(f"""
                SELECT secid
                FROM optionm.securd1
                WHERE ticker = '{ticker}'
            """)

            if secid_df.empty:
                continue  

            secids = secid_df["secid"].tolist()
            secid_sql = f"({','.join(map(str, secids))})"

            offset = 0
            first_chunk = True

            # Forbered output path
            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"forward price.csv"

            while True:
                sql_chunk = f"""
                    SELECT f.secid, s.ticker, s.cusip, s.issuer, 
                        f.date, f.expiration, f.amsettlement, f.forwardprice
                    FROM {base_table}
                    LEFT JOIN optionm.securd1 s ON f.secid = s.secid
                    WHERE f.date BETWEEN '{begdate}' AND '{enddate}'
                    AND f.secid IN {secid_sql}
                    AND f.forwardprice >= 1
                    LIMIT {chunk_size} OFFSET {offset}
                """

                chunk = db.raw_sql(sql_chunk, date_cols=["date", "expiration"])

                if chunk.empty:
                    break  # Færdig med denne ticker

                # Skriv chunk til fil
                if first_chunk:
                    chunk.to_csv(output_file, index=False, mode='w')
                else:
                    chunk.to_csv(output_file, index=False, mode='a', header=False)

                first_chunk = False
                offset += chunk_size


    return None


def fetch_stock_returns_per_ticker(db, begdate, enddate, tickers, base_dir):

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    # Udvid enddate for evt. beregning af efterfølgende returns
    enddate_extended = (pd.to_datetime(enddate) + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

    with tqdm(tickers, desc="Importing stock returns", unit="ticker") as pbar:
        for ticker in pbar:
            pbar.set_postfix({"ticker": ticker})

            # Find PERMNO for ticker
            ticker_df = db.raw_sql(f"""
                SELECT permno, ticker, cusip, comnam as issuer
                FROM crsp.dsenames
                WHERE ticker = '{ticker}'
            """)

            if ticker_df.empty:
                continue  # spring ticker over hvis ikke fundet

            permnos = ticker_df["permno"].tolist()
            permno_sql = f"({','.join(map(str, permnos))})"

            # Forbered output path
            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"returns and stock price.csv"

            # Hent data for ticker
            query = f"""
                SELECT d.permno, n.ticker, n.cusip, n.comnam as issuer,
                    d.date, d.bidlo as open, d.prc as close, d.ret as return
                FROM crsp.dsf d
                LEFT JOIN crsp.dsenames n ON d.permno = n.permno
                WHERE d.date BETWEEN '{begdate}' AND '{enddate_extended}'
                AND n.namedt <= d.date AND d.date <= n.nameendt
                AND d.permno IN {permno_sql}
            """

            df = db.raw_sql(query, date_cols=["date"])

            # Fjern evt. dubletter hvis flere tickers deler PERMNO
            df = df[df["ticker"] == ticker]

            if not df.empty:
                df.sort_values(by=["ticker", "date"], inplace=True)
                df.to_csv(output_file, index=False)

    return None


def fetch_stock_returns_per_permno(db, begdate, enddate, permnos, base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(permnos, int):
        permnos = [permnos]

    # Udvid enddate for evt. efterfølgende returnberegning
    enddate_extended = (pd.to_datetime(enddate) + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

    with tqdm(permnos, desc="Importing stock returns", unit="permno") as pbar:
        for permno in pbar:
            pbar.set_postfix({"permno": permno})

            # Hent firmainfo for permno
            firm_df = db.raw_sql(f"""
                SELECT permno, ticker, cusip, comnam AS issuer
                FROM crsp.dsenames
                WHERE permno = {permno}
            """)

            if firm_df.empty:
                continue

            ticker = firm_df["ticker"].iloc[0]
            cusip  = firm_df["cusip"].iloc[0]
            issuer = firm_df["issuer"].iloc[0]

            # Forbered output path
            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"returns and stock price.csv"

            query = f"""
                SELECT d.permno, n.ticker, n.cusip, n.comnam AS issuer,
                       d.date, d.bidlo AS open, d.prc AS close, d.ret AS return
                FROM crsp.dsf d
                LEFT JOIN crsp.dsenames n ON d.permno = n.permno
                WHERE d.date BETWEEN '{begdate}' AND '{enddate_extended}'
                  AND d.permno = {permno}
                  AND n.namedt <= d.date AND d.date <= n.nameendt
            """

            df = db.raw_sql(query, date_cols=["date"])

            if not df.empty:
                df.sort_values(by=["date"], inplace=True)
                df.to_csv(output_file, index=False)

    return None




def fetch_stock_returns_per_ticker_om(db, begdate, enddate, tickers, base_dir, chunk_size=100_000):

    from pathlib import Path
    from tqdm import tqdm

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    # Forbered samlet view af alle år med kun nødvendige kolonner
    union_tables = " \nUNION ALL\n ".join(
        f"SELECT secid, date, open, close, return FROM optionm.secprd{year}"
        for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS p"

    with tqdm(tickers, desc="Importing stock returns from OM", unit="ticker") as pbar:
        for ticker in pbar:
            pbar.set_postfix({"ticker": ticker})

            # Find SECIDs for ticker
            secid_df = db.raw_sql(f"""
                SELECT secid, ticker, cusip, issuer, issue_type, exchange_d
                FROM optionm.securd1
                WHERE ticker = '{ticker}'
            """)

            if secid_df.empty:
                print(f"[!] No SECID found for ticker: {ticker}")
                continue

            secids = secid_df["secid"].tolist()
            secid_sql = f"({','.join(map(str, secids))})"

            offset = 0
            first_chunk = True

            # Forbered output path
            ticker_output_dir = base_dir / "Tickers" / "Returns OM" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"returns and stock price OM.csv"

            while True:
                sql_chunk = f"""
                    SELECT 
                        p.secid, s.ticker, s.cusip, s.issuer, 
                        p.date, p.open, p.close, p.return,
                        s.issue_type, s.exchange_d
                    FROM {base_table}
                    LEFT JOIN optionm.securd1 s ON p.secid = s.secid
                    WHERE p.date BETWEEN '{begdate}' AND '{enddate}'
                      AND p.secid IN {secid_sql}
                    LIMIT {chunk_size} OFFSET {offset}
                """

                chunk = db.raw_sql(sql_chunk, date_cols=["date"])

                if chunk.empty:
                    if offset == 0:
                        print(f"[!] No return data found for ticker: {ticker}")
                    break

                if first_chunk:
                    chunk.to_csv(output_file, index=False, mode='w')
                else:
                    chunk.to_csv(output_file, index=False, mode='a', header=False)

                first_chunk = False
                offset += chunk_size

    return None




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
    except Exception as e:
        print(f"Error in data import: {e}")
        return None

    # Sortér og gem data
    df.sort_values(by=["date", "days"], inplace=True)
    df.to_csv(csv_path, index=False)

    return df



def fetch_wrds_data_per_ticker(db, tickers, begdate="1996-01-01", enddate="2024-12-31", data_types=["O", "F", "S", "Z"], chunk_size=1000000, chunk_days = 30, chunk_on_days = True, rel_diff_from_median=1):

    #rmeoviing dublicates in case
    tickers = list(set(tickers))

    base_dir = load.dirs()["OptionMetrics"]
    base_dir = Path(base_dir)
    tickers_to_fetch = []
    skipped_tickers = []

    for ticker in tickers:
        ticker_path = base_dir / "Tickers" / "Input" / ticker
        if ticker_path.exists():
            skipped_tickers.append(ticker)
        else:
            tickers_to_fetch.append(ticker)
            
    if "S_OM" in data_types:
        fetch_stock_returns_per_ticker_om(db, begdate, enddate, tickers, base_dir) #run for all tickers it is fast anyway

    if skipped_tickers:
        print(f"Skipping already existing tickers: {', '.join(skipped_tickers)}")

    if not tickers_to_fetch:
        print("No new tickers to fetch. Exiting.")
    
        return None

    # Loop over datatyper
    for data_type in data_types:
        if data_type == "O":
            if not chunk_on_days: fetch_options_data_per_ticker(db, begdate, enddate, tickers_to_fetch, base_dir, chunk_size=chunk_size)
            if chunk_on_days: fetch_options_data_per_ticker_days_iv_flag(db, begdate, enddate, tickers_to_fetch, base_dir, chunk_days=chunk_days, rel_diff_from_median=rel_diff_from_median)

        elif data_type == "F":
            fetch_forward_prices_per_ticker(db, begdate, enddate, tickers_to_fetch, base_dir, chunk_size=chunk_size)

        elif data_type == "S":
            fetch_stock_returns_per_ticker(db, begdate, enddate, tickers_to_fetch, base_dir)

        elif data_type == "Z":
            csv_path = base_dir / "Tickers" / "ZC yield curve.csv"

            if csv_path.exists():
                print("Yield curve file already exists.")
            else:
                print(f"Importing yield curve data into: {csv_path}")
                fetch_zerocoupons(db=db, begdate=begdate, enddate=enddate, csv_path=csv_path)
        
        elif data_type == "S_OM":
            continue
        
        else:
            raise ValueError(f"data_type '{data_type}' not supported.")

    return None


