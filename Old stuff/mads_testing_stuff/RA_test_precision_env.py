import numpy as np
import math
from scipy.stats import norm
import pandas as pd

# --- Black‑76 vectorized pricer
def bs_price_vectorized(F, K, T, cp_flag, sigma, r=0.0):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put  = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return np.where(cp_flag=='C', call, put).astype(float)

# --- Merton Jump‑Diffusion (mixture) with compensator
def mjd_price_vectorized(F, K, T, cp_flag,
                         sigma, lam, mu_J, sigma_J,
                         r=0.0, n_max=50):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    # compensator
    muY   = mu_J + 0.5 * sigma_J**2
    gamma = np.exp(muY) - 1.0
    F_adj = F * np.exp(-lam * gamma * T)
    fact = np.array([math.factorial(n) for n in range(n_max+1)], dtype=float)
    prices = np.zeros_like(F, dtype=float)
    for n in range(n_max+1):
        p_n = np.exp(-lam * T) * (lam * T)**n / fact[n]
        F_n = F_adj * np.exp(n * muY)
        sigma_n = np.sqrt(sigma**2 + n * sigma_J**2 / T)
        prices += p_n * bs_price_vectorized(F_n, K, T, cp_flag, sigma_n, r=0.0)
    return np.exp(-r * T) * prices

# --- Heston characteristic function (trap form)
def _heston_cf_trap(u, F, T, kappa, theta, xi, rho, v0):
    a = kappa * theta
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b*b + xi**2 * (u*u + 1j*u))
    g = (b - d)/(b + d)
    expd = np.exp(-d*T)
    C = (a/xi**2)*((b-d)*T - 2*np.log((1-g*expd)/(1-g)))
    D = ((b-d)/xi**2)*((1-expd)/(1-g*expd))
    return np.exp(C + D*v0 + 1j*u*np.log(F))

# --- Bates characteristic function (trap + jump)
def _bates_cf_trap(u, F, T, kappa, theta, xi, rho, v0,
                   lam, mu_J, sigma_J):
    # jump compensator
    muY   = mu_J + 0.5*sigma_J**2
    gamma = np.exp(muY) - 1.0
    F_adj = F * np.exp(-lam * gamma * T)
    # Heston part
    phi0 = _heston_cf_trap(u, F_adj, T, kappa, theta, xi, rho, v0)
    # jump part
    jump = np.exp(lam*T*(np.exp(1j*u*mu_J - 0.5*sigma_J**2*u**2)-1))
    return phi0 * jump

# --- Simpson rule integrator
def simpson_rule(fx, u):
    """Composite Simpson’s rule on grid u (evenly spaced)."""
    N = len(u)
    h = u[1] - u[0]
    w = np.ones(N)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return (h/3) * np.sum(w * fx)

_eps = 1e-8

# --- Heston pricer via Simpson
def price_heston_simpson(F, K, T, cp_flag,
                         kappa, theta, xi, rho, v0,
                         r=0.0,
                         U_max=800, N=8000):
    # avoid u=0
    u = np.linspace(_eps, U_max, N)
    lnK = np.log(K)
    phi_u   = _heston_cf_trap(u,    F, T, kappa, theta, xi, rho, v0)
    phi_u1  = _heston_cf_trap(u-1j, F, T, kappa, theta, xi, rho, v0)
    phi_m1  = _heston_cf_trap(-1j,  F, T, kappa, theta, xi, rho, v0)
    int1 = np.real(np.exp(-1j*u*lnK) * phi_u1/(1j*u*phi_m1))
    int2 = np.real(np.exp(-1j*u*lnK) * phi_u   /(1j*u))
    P1 = 0.5 + simpson_rule(int1, u)/np.pi
    P2 = 0.5 + simpson_rule(int2, u)/np.pi
    call_fwd = F*P1 - K*P2
    put_fwd  = call_fwd + K - F
    price = np.exp(-r*T) * (call_fwd if cp_flag=='C' else put_fwd)
    return price

# --- Bates pricer via Simpson
def price_bates_simpson(F, K, T, cp_flag,
                        kappa, theta, xi, rho, v0,
                        lam, mu_J, sigma_J,
                        r=0.0,
                        U_max=800, N=8000):
    u = np.linspace(_eps, U_max, N)
    lnK = np.log(K)
    phi_u   = _bates_cf_trap(u,    F, T, kappa, theta, xi, rho, v0,
                              lam, mu_J, sigma_J)
    phi_u1  = _bates_cf_trap(u-1j, F, T, kappa, theta, xi, rho, v0,
                              lam, mu_J, sigma_J)
    phi_m1  = _bates_cf_trap(-1j,  F, T, kappa, theta, xi, rho, v0,
                              lam, mu_J, sigma_J)
    int1 = np.real(np.exp(-1j*u*lnK) * phi_u1/((1j*u)*phi_m1))
    int2 = np.real(np.exp(-1j*u*lnK) * phi_u   /(1j*u))
    P1 = 0.5 + simpson_rule(int1, u)/np.pi
    P2 = 0.5 + simpson_rule(int2, u)/np.pi
    call_fwd = F*P1 - K*P2
    put_fwd  = call_fwd + K - F
    price = np.exp(-r*T) * (call_fwd if cp_flag=='C' else put_fwd)
    return price

# --- master price selector
def price_model(F_arr, K_arr, T_arr, cp_flag_arr, ticker_arr, v0_arr,
                bs_params, mjd_params, heston_params, bates_params):
    prices = np.empty_like(F_arr, dtype=float)
    # BS
    mask_bs = (ticker_arr=='BS')
    if mask_bs.any():
        prices[mask_bs] = bs_price_vectorized(
            F_arr[mask_bs], K_arr[mask_bs], T_arr[mask_bs], cp_flag_arr[mask_bs],
            **bs_params
        )
    # MJD
    mask_mjd = (ticker_arr=='MJD')
    if mask_mjd.any():
        prices[mask_mjd] = mjd_price_vectorized(
            F_arr[mask_mjd], K_arr[mask_mjd], T_arr[mask_mjd], cp_flag_arr[mask_mjd],
            **mjd_params
        )
    # Heston
    mask_heston = (ticker_arr=='Heston')
    if mask_heston.any():
        p = heston_params
        for i in np.where(mask_heston)[0]:
            prices[i] = price_heston_simpson(
                F_arr[i], K_arr[i], T_arr[i], cp_flag_arr[i],
                p['kappa'], p['theta'], p['xi'], p['rho'], v0_arr[i],
                r=bs_params.get('r',0.0)
            )
    # Bates
    mask_bates = (ticker_arr=='SVMJD')
    if mask_bates.any():
        p = bates_params
        for i in np.where(mask_bates)[0]:
            prices[i] = price_bates_simpson(
                F_arr[i], K_arr[i], T_arr[i], cp_flag_arr[i],
                p['kappa'], p['theta'], p['xi'], p['rho'], v0_arr[i],
                p['lam'], p['mu_J'], p['sigma_J'],
                r=p.get('r',0.0)
            )
    return prices

# --- helper to build test DataFrame
def create_fake_option_df(TTM = 1/12.):
    K_low  = np.array([80, 90, 99.999999, 100.000001, 110, 120])
    K_high = np.arange(60,150,1)
    base = pd.DataFrame({'F':[100],'t_TTM':[TTM],'r':[0.056],'IV':[np.nan]})
    tickers = ['BS','MJD','Heston','SVMJD']
    frames = []
    for ticker in tickers:
        for label, Ks in [('low',K_low),('high',K_high)]:
            df = base.loc[base.index.repeat(len(Ks))].reset_index(drop=True).copy()
            df['K'] = np.tile(Ks,len(base))
            df['ticker'] = ticker
            df['cp_flag'] = np.where(df['F']<df['K'],'C','P')
            df['low'] = (label=='low')
            df['high'] = (label=='high')
            frames.append(df)
    return pd.concat(frames,ignore_index=True)[['ticker','cp_flag','K','t_days','t_TTM','r','F','IV','low','high']]

# EOF



