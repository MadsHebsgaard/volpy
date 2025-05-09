import numpy as np
import math
from scipy.stats import norm
from numpy.polynomial.laguerre import laggauss
import pandas as pd

def bs_price_vectorized(F, K, T, cp_flag, sigma, r=0.0):
    """Black‑76 vectorized pricer on forwards."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put = K * norm.cdf(-d2) - F * norm.cdf(-d1)

    mask = (cp_flag == 'C')
    return np.where(mask, call, put).astype(float)


def mjd_price_vectorized_old(F, K, T, cp_flag,
                             sigma, lam, mu_J, sigma_J,
                             r=0.0, n_max=50):
    """Merton jump‐diffusion via Poisson‐mixture of Black‑76."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    prices = np.zeros_like(F, dtype=float)
    fact = np.array([math.factorial(n) for n in range(n_max + 1)], dtype=float)

    for n in range(n_max + 1):
        p_n = np.exp(-lam * T) * (lam * T) ** n / fact[n]
        F_n = F * np.exp(n * mu_J)
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_J ** 2 / T)
        chunk = bs_price_vectorized(F_n, K, T, cp_flag, sigma_n, r)
        prices += p_n * chunk

    return prices


def mjd_price_vectorized(F, K, T, cp_flag,
                         sigma, lam, mu_J, sigma_J,
                         r=0.0, n_max=50):
    """Merton jump‑diffusion via Poisson‑mixture of Black‑76,
       with correct log‑normal jump compensator & discounting."""
    # array‐ify inputs
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    # 1) effective jump‐mean & compensator
    muY = mu_J + 0.5 * sigma_J ** 2
    gamma = np.exp(muY) - 1.0

    # 2) adjust base forward
    F_adj = F * np.exp(-lam * gamma * T)

    # 3) Poisson mixture
    prices = np.zeros_like(F_adj, dtype=float)
    fact = np.array([math.factorial(n) for n in range(n_max + 1)], dtype=float)

    for n in range(n_max + 1):
        p_n = np.exp(-lam * T) * (lam * T) ** n / fact[n]
        F_n = F_adj * np.exp(n * muY)
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_J ** 2 / T)

        # use r=0 here because we discount once below
        chunk = bs_price_vectorized(F_n, K, T, cp_flag, sigma_n, r=0.0)
        prices += p_n * chunk

    # 4) final spot‐measure discount
    return np.exp(-r * T) * prices


# Gauss–Laguerre nodes & weights for Heston
_GL_X, _GL_W = laggauss(32)


def _heston_cf_trap(u, F, T, kappa, theta, xi, rho, v0):
    a = kappa * theta
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + (u * u + 1j * u) * xi * xi)
    g = (b - d) / (b + d)
    exp_dT = np.exp(-d * T)
    C = (a / xi ** 2) * ((b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = ((b - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * v0 + 1j * u * np.log(F))


def _integrand_P(u, F, K, T, kappa, theta, xi, rho, v0, j):
    if j == 1:
        u_shift = u - 1j
        cf_num = _heston_cf_trap(u_shift, F, T, kappa, theta, xi, rho, v0)
        cf_den = _heston_cf_trap(-1j, F, T, kappa, theta, xi, rho, v0)
        phi_j = cf_num / cf_den
    else:
        phi_j = _heston_cf_trap(u, F, T, kappa, theta, xi, rho, v0)
    return np.real(np.exp(-1j * u * np.log(K)) * phi_j / (1j * u))


def heston_price_vectorized(F_arr, K_arr, T_arr, cp_flag_arr, v0_arr,
                            kappa, theta, xi, rho):
    prices = np.zeros_like(F_arr, dtype=float)
    # zip in v0_arr so each iteration sees one scalar v0
    for i, (F, K, T, flag, v0) in enumerate(zip(F_arr, K_arr, T_arr, cp_flag_arr, v0_arr)):
        # now v0 is a float, and _integrand_P will see a scalar v0
        P1 = 0.5 + (1 / np.pi) * np.sum(
            _GL_W * np.exp(_GL_X) *
            _integrand_P(_GL_X, F, K, T,
                         kappa, theta, xi, rho,
                         v0, j=1)
        )
        P2 = 0.5 + (1 / np.pi) * np.sum(
            _GL_W * np.exp(_GL_X) *
            _integrand_P(_GL_X, F, K, T,
                         kappa, theta, xi, rho,
                         v0, j=2)
        )
        call = F * P1 - K * P2
        put = call + K - F
        prices[i] = call if flag == 'C' else put

    return prices


def mjd_heston_price_vectorized(F_arr, K_arr, T_arr, cp_flag_arr, v0_arr,
                                sigma, lam, mu_J, sigma_J,
                                kappa, theta, xi, rho,
                                r=0.0, n_max=50):
    """Bates (MJD–Heston): Poisson mixture of Heston prices."""
    F_arr = np.asarray(F_arr, dtype=float)
    K_arr = np.asarray(K_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)

    prices = np.zeros_like(F_arr, dtype=float)
    fact = np.array([math.factorial(n) for n in range(n_max + 1)], dtype=float)

    # for each mixture component n
    for n in range(n_max + 1):
        p_n = np.exp(-lam * T_arr) * (lam * T_arr) ** n / fact[n]
        F_n = F_arr * np.exp(n * mu_J)
        # call our fixed heston pricer, still passing the same v0_arr
        chunk = heston_price_vectorized(
            F_n, K_arr, T_arr, cp_flag_arr, v0_arr,
            kappa, theta, xi, rho
        )
        prices += p_n * chunk

    return prices


def price_model(F_arr, K_arr, T_arr, cp_flag_arr, ticker_arr, v0_arr,
                bs_params, mjd_params, heston_params, mjd_heston_params):
    prices = np.empty_like(F_arr, dtype=float)
    mask = ticker_arr == 'BS'
    if mask.any():
        prices[mask] = bs_price_vectorized(
            F_arr[mask], K_arr[mask], T_arr[mask], cp_flag_arr[mask],
            **bs_params
        )
    mask = ticker_arr == 'MJD'
    if mask.any():
        prices[mask] = mjd_price_vectorized(
            F_arr[mask], K_arr[mask], T_arr[mask], cp_flag_arr[mask],
            **mjd_params
        )
    mask = ticker_arr == 'Heston'
    if mask.any():
        prices[mask] = heston_price_vectorized(
            F_arr[mask], K_arr[mask], T_arr[mask], cp_flag_arr[mask], v0_arr[mask],
            **heston_params
        )
    mask = ticker_arr == 'SVMJD'
    if mask.any():
        prices[mask] = mjd_heston_price_vectorized(
            F_arr[mask], K_arr[mask], T_arr[mask], cp_flag_arr[mask], v0_arr[mask],
            **mjd_heston_params
        )
    return prices




def create_fake_option_df(TTM):
    # 1) build the base DataFrame of common fields
    K_low = np.array([80, 90, 99.999999, 100.000001, 110, 120])  # np.arange(80, 130, 10)  # [80, 90, 100, 110, 120]
    K_high = np.arange(60, 150, 1)  # [80, 85, …, 120]
    base = pd.DataFrame({
        'F': [100],
        't_days': [21],
        't_TTM': [TTM],  # [1/12], [30/365]
        'r': [0.056],
        'IV': [np.nan],
    })

    # 2) replicate for each ticker & each strike‐grid
    tickers = ['BS', 'MJD', 'Heston', 'SVMJD']
    frames = []

    for ticker in tickers:
        for label, Ks in [('low', K_low), ('high', K_high)]:
            # repeat the one‐row base for as many strikes as we have
            df = base.loc[base.index.repeat(len(Ks))].reset_index(drop=True).copy()
            df['K'] = np.tile(Ks, len(base))
            df['ticker'] = ticker
            df['cp_flag'] = np.where(df['F'] < df['K'], 'C', 'P')
            df['low'] = (label == 'low')
            df['high'] = (label == 'high')
            frames.append(df)

    # 3) concatenate and reorder
    od_RA = pd.concat(frames, ignore_index=True)[
        ['ticker', 'cp_flag', 'K', 't_days', 't_TTM', 'r', 'F', 'IV', 'low', 'high']
    ]
    return od_RA