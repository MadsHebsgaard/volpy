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
            if max_lags is None:
                fit = sm.OLS(gA[y_var], X).fit()
            else:
                fit = sm.OLS(gA[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
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
            if max_lags is None:
                fit = sm.OLS(gB[y_var], X).fit()
            else:
                fit = sm.OLS(gB[y_var], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
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




def ff_three_factor_table(returns_df, name,
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
            fit = (sm.OLS(g0[y_var], X)
                   .fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}))
            params = fit.params * TRADING_DAYS
            tvals  = fit.tvalues
            r2     = fit.rsquared
            # pull out each coefficient + t-stat
            alpha, βm, s, h = params["const"], params["Mkt"], params["SMB"], params["HML"]
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

    df = pd.DataFrame(records).set_index("Ticker")

    caption = (
        "Explaining variance risk premiums with Fama–French risk factors. "
        "Entries report the GMM estimates (and t-statistics in parentheses) of "
        r"$\ln RV_{t,\tau}/SW_{t,\tau} = \alpha + \beta\,ER^m_{t,\tau} + s\,SMB_{t,\tau} + h\,HML_{t,\tau} + e$, "
        "and unadjusted $R^2$."
    )

    latex = (
        df.to_latex(index=True, escape=False, longtable=True,
                    caption=caption,
                    label=f"tab:analysis:ff{name}",
                    column_format="lccccc",
                    float_format="%.3f")
          .splitlines()
    )

    # wrap in scriptsize + float barrier
    out = ["\\scriptsize"] + latex + ["\\normalsize", "\\FloatBarrier"]
    return "\n".join(out), df



def save_ff_table(returns_df, name, y_var = "r_30_SW_day", max_lags = 0):
    tex, df = ff_three_factor_table(returns_df, name, y_var, max_lags = max_lags)
    with open(f"figures/Analysis/table_ff_{name}.tex", "w") as f:
        f.write(tex)
    return df




