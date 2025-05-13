import wrds
import os
import pandas as pd
import load_clean_lib as load
from tqdm import tqdm  
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta  # ← vigtigt!



def fetch_options_data_per_ticker_days_iv_flag(
    db,
    begdate,
    enddate,
    tickers_secids,  # dict: {ticker: secid}
    base_dir,
    chunk_days=30
):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    begdate_dt = pd.to_datetime(begdate)
    enddate_dt = pd.to_datetime(enddate)

    start_year = max(begdate_dt.year, 1996)
    end_year = min(enddate_dt.year + 1, 2024)

    # Lav UNION af relevante opprcd-årstabeller
    union_tables = " \nUNION ALL\n ".join(
        f"SELECT * FROM optionm.opprcd{year}" for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS o"

    with tqdm(tickers_secids.items(), desc="Importing option data", unit="ticker") as pbar:
        for ticker, secid in pbar:
            pbar.set_postfix({"ticker": ticker})

            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"option data.csv"

            current_start = begdate_dt
            first_chunk = True

            while current_start < enddate_dt:
                current_end = min(current_start + timedelta(days=chunk_days), enddate_dt)
                date_start_str = current_start.strftime("%Y-%m-%d")
                date_end_str = current_end.strftime("%Y-%m-%d")

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
                          AND o.secid = {secid}
                          AND (o.exdate - o.date) <= 365
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
                                AND ABS(b.impl_volatility - m.median_iv) / m.median_iv <= 5
                            )
                            OR ABS(b.impl_volatility - m.median_iv) <= 0.5
                            OR b.impl_volatility IS NULL
                        )
                """

                # AND NOT (s.issue_type = '' OR s.exchange_d = 0)

                chunk = db.raw_sql(sql_chunk, date_cols=["date", "exdate"])

                if not chunk.empty:
                    chunk["ticker"] = ticker
                    mode = 'w' if first_chunk else 'a'
                    chunk.to_csv(output_file, index=False, mode=mode, header=first_chunk)
                    first_chunk = False

                current_start = current_end

    return None




def fetch_forward_prices_per_ticker(db, begdate, enddate, tickers_secids, base_dir, chunk_size=100_000):
    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    start_year = max(int(begdate[:4]), 1996)
    end_year = min(int(enddate[:4]) + 1, 2024)

    union_tables = " \nUNION ALL\n ".join(
        f"SELECT * FROM optionm.fwdprd{year}" for year in range(start_year, end_year)
    )
    base_table = f"({union_tables}) AS f"

    with tqdm(tickers_secids.items(), desc="Importing forward prices", unit="ticker") as pbar:
        for ticker, secid in pbar:
            pbar.set_postfix({"ticker": ticker})

            offset = 0
            first_chunk = True

            # Output path
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
                      AND f.secid = {secid}
                      AND f.forwardprice > 0 
                    LIMIT {chunk_size} OFFSET {offset}
                """

                chunk = db.raw_sql(sql_chunk, date_cols=["date", "expiration"])

                if chunk.empty:
                    break  # Done with this secid
                
                chunk["ticker"] = ticker

                if first_chunk:
                    chunk.to_csv(output_file, index=False, mode='w')
                else:
                    chunk.to_csv(output_file, index=False, mode='a', header=False)

                first_chunk = False
                offset += chunk_size

    return None




def fetch_stock_returns_per_ticker(db, begdate, enddate, tickers_permnos, base_dir):
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Udvid enddate for evt. beregning af efterfølgende returns
    enddate_extended = (pd.to_datetime(enddate) + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

    with tqdm(tickers_permnos.items(), desc="Importing stock returns", unit="ticker") as pbar:
        for ticker, permno in pbar:
            pbar.set_postfix({"ticker": ticker})

            # Output dir: navngiv med ticker
            ticker_output_dir = base_dir / "Tickers" / "Input" / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = ticker_output_dir / f"returns and stock price.csv"

            # Hent data for PERMNO
            query = f"""
                WITH firm_info AS (
                    SELECT d.permno, d.date, d.bidlo AS open, d.prc AS close, d.ret AS return,
                        FIRST_VALUE(n.ticker) OVER w AS ticker,
                        FIRST_VALUE(n.cusip) OVER w AS cusip,
                        FIRST_VALUE(n.comnam) OVER w AS issuer
                    FROM crsp.dsf d
                    LEFT JOIN crsp.dsenames n
                    ON d.permno = n.permno
                    AND n.namedt <= d.date AND d.date <= n.nameendt
                    WHERE d.permno = {permno}
                    AND d.date BETWEEN '{begdate}' AND '{enddate_extended}'
                    WINDOW w AS (
                        PARTITION BY d.permno, d.date
                        ORDER BY n.namedt DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    )
                )
                SELECT DISTINCT permno, date, open, close, return, ticker, cusip, issuer
                FROM firm_info
            """

            df = db.raw_sql(query, date_cols=["date"])

            if not df.empty:
                df["ticker"] = ticker
                df.sort_values(by=["ticker", "date"], inplace=True)
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



def fetch_wrds_data_per_ticker(db, tickers, begdate="1996-01-01", enddate="2024-12-31",
                                data_types=["O", "F", "S", "Z"], chunk_size=1000000,
                                chunk_days=30, overwrite = False):

    tickers = list(set(tickers))  # fjern duplikater

    # 1. Opsæt base-dir og indlæs mapping
    base_dir = Path(load.dirs()["OptionMetrics"])
    mapping_path = base_dir / "Tickers" / "index data" / "final map done_v2.xlsx"
    df_map = pd.read_excel(mapping_path, dtype=str)

    # Forventede kolonner: 'ticker', 'permno', 'secid'
    df_map["ticker"] = df_map["ticker"].str.strip().str.upper()

    # Opret direkte mappings uden check (vi ved, det er rent)
    ticker_to_permno = dict(zip(df_map["ticker"], df_map["permno"]))
    ticker_to_secid  = dict(zip(df_map["ticker"], df_map["secid"]))
    

    # 2. Tjek om alle inputtickers findes i mapping
    missing_tickers = [t for t in tickers if t.upper() not in ticker_to_permno]

    if missing_tickers:
        print(f"Tickers not found in mapping: {missing_tickers}")

    tickers = [t for t in tickers if t.upper() in ticker_to_permno]

    tickers_to_fetch = []
    skipped_tickers = []

    for ticker in tickers:
        t_upper = ticker.strip().upper()
        ticker_path = base_dir / "Tickers" / "Input" / t_upper
        if ticker_path.exists() and not overwrite:
            skipped_tickers.append(t_upper)
        else:
            tickers_to_fetch.append(t_upper)

    if "S_OM" in data_types:
        fetch_stock_returns_per_ticker_om(db, begdate, enddate, tickers, base_dir)  # denne bruger tickers direkte

    if skipped_tickers:
        print(f"Skipping already existing tickers: {', '.join(skipped_tickers)}")

    if not tickers_to_fetch:
        print("No new tickers to fetch. Exiting.")
        return None

    # 2. Loop over ønskede datatyper
    for data_type in data_types:
        if data_type == "O":
            tickers_secids = {t: ticker_to_secid[t] for t in tickers_to_fetch if t in ticker_to_secid}
            fetch_options_data_per_ticker_days_iv_flag(db, begdate, enddate, tickers_secids, base_dir, chunk_days)

        elif data_type == "F":
            tickers_secids = {t: ticker_to_secid[t] for t in tickers_to_fetch if t in ticker_to_secid}
            fetch_forward_prices_per_ticker(db, begdate, enddate, tickers_secids, base_dir, chunk_size=chunk_size)

        elif data_type == "S":
            tickers_permnos = {t: ticker_to_permno[t] for t in tickers_to_fetch if t in ticker_to_permno}
            fetch_stock_returns_per_ticker(db, begdate, enddate, tickers_permnos, base_dir)

        elif data_type == "Z":
            csv_path = base_dir / "Tickers" / "ZC yield curve.csv"
            if csv_path.exists():
                print("Yield curve file already exists.")
            else:
                print(f"Importing yield curve data into: {csv_path}")
                fetch_zerocoupons(db=db, begdate=begdate, enddate=enddate, csv_path=csv_path)

        elif data_type == "S_OM":
            continue  # allerede kørt ovenfor

        else:
            raise ValueError(f"data_type '{data_type}' not supported.")

    return None



