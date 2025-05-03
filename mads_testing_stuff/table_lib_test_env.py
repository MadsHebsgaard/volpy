import numpy as np
import pandas as pd
import statsmodels.api as sm

from table_lib import sort_table_df


def capm_beta_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    TRADING_DAYS = 252
    records = []
    for ticker, g in returns_df.groupby("ticker"):
        rec = {"Ticker": ticker}

        # Panel A: SPX
        gA = g.dropna(subset=[y_var, "SPX"])
        if len(gA) > 0:
            X = sm.add_constant(gA["SPX"])
            fit = sm.OLS(gA[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}) #
            a, b = fit.params
            ta, tb = fit.tvalues
            r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan

        ann_a = a * TRADING_DAYS

        rec.update({
            ("Panel A: SPX", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
            ("Panel A: SPX", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
            ("Panel A: SPX", r"\(R^2\)"): f"{r2:.3f}"
        })

        # Panel B: Mkt
        gB = g.dropna(subset=[y_var, "Mkt"])
        if len(gB) > 0:
            X = sm.add_constant(gB["Mkt"])
            fit = sm.OLS(gB[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}) #
            a, b = fit.params
            ta, tb = fit.tvalues
            r2 = fit.rsquared
        else:
            a = b = ta = tb = r2 = np.nan

        ann_a = a * TRADING_DAYS

        rec.update({
            ("Panel B: Mkt", r"\(\alpha\)"): f"{ann_a:.3f}\n({ta:.3f})",
            ("Panel B: Mkt", r"\(\beta\)"): f"{b:.3f}\n({tb:.3f})",
            ("Panel B: Mkt", r"\(R^2\)"): f"{r2:.3f}"
        })

        records.append(rec)

    df = pd.DataFrame(records)
    df.rename(columns={'Ticker': 'ticker'}, inplace=True)
    df = sort_table_df(df, returns_df, name)
    df.rename(columns={'ticker': 'Ticker'}, inplace=True)
    df = df.set_index("Ticker")
    # ensure proper MultiIndex on columns
    df.columns = pd.MultiIndex.from_tuples(df.columns)


    latex = df.to_latex(
        index=True,
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
        column_format="l" + "ccc" * 2,
        multicolumn_format="c",
        float_format="%.3f",
    ).splitlines()

    # insert cmidrules under the top header
    for i, L in enumerate(latex):
        if L.strip().startswith(r"\toprule"):
            # after \toprule and the next header line
            latex.insert(i + 2, r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
            break

    # add sizing & float barrier
    out = ["\scriptsize"] + latex + ["\\normalsize", r"\FloatBarrier"]
    return "\n".join(out), df


def save_capm_beta_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = capm_beta_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/table_beta_{name}.tex", "w") as f:
        f.write(tex)
    return df