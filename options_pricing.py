"""
Black-Scholes Options Pricing Engine

Implements the Black-Scholes-Merton model for European options.
Includes first- and second-order Greeks, an IV solver via Newton-Raphson,
and multi-leg strategy construction.

The BSM model assumes:
  - Log-normal asset returns (GBM): dS = (r-q)S dt + σS dW
  - Constant volatility and risk-free rate
  - No arbitrage, frictionless markets, continuous trading
  - European-style exercise only

All Greeks are analytical (closed-form) derivatives of the BSM formula.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Compute d1 and d2 — the arguments to the normal CDF in BSM.

    d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T

    d1 can be interpreted as a risk-adjusted moneyness measure.
    N(d2) is the risk-neutral probability of the option expiring ITM.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton price for a European option.

    Parameters
    ----------
    S     : Current spot price
    K     : Strike price
    T     : Time to expiry in years
    r     : Continuously compounded risk-free rate
    sigma : Annualised implied volatility
    q     : Continuously compounded dividend yield (default 0)

    Formula
    -------
    Call: S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    Put:  K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

    The discount factors e^(-rT) and e^(-qT) reflect:
      - e^(-rT): present value of receiving K at expiry
      - e^(-qT): present value of holding a dividend-paying asset
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# First-order Greeks
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, option_type: OptionType = "call", q: float = 0.0) -> float:
    """
    Delta = ∂V/∂S — sensitivity to spot price.

    Call delta ∈ (0, 1),  Put delta ∈ (-1, 0).
    ATM options have |delta| ≈ 0.5.
    Delta also approximates the risk-neutral probability of expiring ITM
    (though strictly that is N(d2), not N(d1)).
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q: float = 0.0) -> float:
    """
    Gamma = ∂²V/∂S² = ∂Δ/∂S — rate of change of delta.

    Gamma is identical for calls and puts (put-call parity).
    Maximised when the option is ATM and near expiry.
    Long gamma → benefits from large moves (positive convexity).
    Short gamma → profits from low realised vol (theta collection).

    Formula: e^(-qT) · φ(d1) / (S · σ · √T)
    where φ is the standard normal PDF.
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q: float = 0.0) -> float:
    """
    Vega = ∂V/∂σ — sensitivity to volatility (per 1% change in vol).

    Vega is identical for calls and puts.
    Maximised ATM; decays as the option moves deep ITM or OTM.
    Long vega → benefits when realised vol exceeds implied vol.

    Formula: S · e^(-qT) · φ(d1) · √T  (divided by 100 for 1% convention)
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S, K, T, r, sigma, option_type: OptionType = "call", q: float = 0.0) -> float:
    """
    Theta = ∂V/∂t — time decay (per calendar day).

    Theta is almost always negative for long options — you pay for
    time value that erodes daily. The decay accelerates near expiry
    (the curve is convex), which is why short-dated ATM options have
    the most negative theta.

    The theta-gamma relationship: Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0
    (BSM PDE) — theta and gamma are two sides of the same coin.
    """
    if T <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        return (
            common
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365.0
    else:
        return (
            common
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365.0


def rho(S, K, T, r, sigma, option_type: OptionType = "call", q: float = 0.0) -> float:
    """
    Rho = ∂V/∂r — sensitivity to interest rates (per 1% change in r).

    Calls have positive rho (higher rates → higher call value via
    cost-of-carry), puts have negative rho.
    Rho matters most for long-dated options and in high-rate environments.

    Formula (call): K · T · e^(-rT) · N(d2) / 100
    """
    if T <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


# ---------------------------------------------------------------------------
# Second-order Greeks
# ---------------------------------------------------------------------------

def vanna(S, K, T, r, sigma, q: float = 0.0) -> float:
    """
    Vanna = ∂²V/(∂S·∂σ) = ∂Δ/∂σ = ∂Vega/∂S

    Measures how delta changes as vol moves, and how vega changes
    with spot. Critical for delta-hedging under vol uncertainty —
    a position with large vanna will see its delta shift significantly
    if implied vol spikes. Also used in skew trading strategies.

    Formula: -e^(-qT) · φ(d1) · d2 / σ
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, sigma, q: float = 0.0) -> float:
    """
    Volga (Vomma) = ∂²V/∂σ² = ∂Vega/∂σ

    Measures the convexity of the option price with respect to vol.
    Positive volga means the option benefits from vol-of-vol.
    OTM options typically have higher volga — they gain
    disproportionately from volatility spikes, explaining part
    of the vol smile/skew observed in practice.

    Formula: Vega · (d1 · d2) / σ
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    v = vega(S, K, T, r, sigma, q) * 100  # undo the /100 convention for internal use
    return v * (d1 * d2) / sigma / 100.0


def charm(S, K, T, r, sigma, option_type: OptionType = "call", q: float = 0.0) -> float:
    """
    Charm = ∂Δ/∂T = ∂²V/(∂S·∂T) — delta decay per day.

    Shows how delta drifts as time passes, holding everything else
    constant. Useful for understanding how a delta hedge will need
    to be rebalanced over time even if spot doesn't move.

    ITM calls have positive charm (delta drifts toward 1),
    OTM calls have negative charm (delta drifts toward 0).
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf_d1 = norm.pdf(d1)
    common = np.exp(-q * T) * pdf_d1 * (
        2 * (r - q) * T - d2 * sigma * np.sqrt(T)
    ) / (2 * T * sigma * np.sqrt(T))

    if option_type == "call":
        return (-q * np.exp(-q * T) * norm.cdf(d1) + common) / 365.0
    return (q * np.exp(-q * T) * norm.cdf(-d1) + common) / 365.0


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict:
    """Return all Greeks in a single dict."""
    return {
        "price": bs_price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
        "charm": charm(S, K, T, r, sigma, option_type, q),
    }


# ---------------------------------------------------------------------------
# Implied Volatility Solver (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Recover implied volatility by inverting the BSM formula via Newton-Raphson.

    The BSM price function is strictly increasing and convex in σ, so N-R
    converges quickly (quadratic convergence near the root).

    Update rule: σ_{n+1} = σ_n - (BSM(σ_n) - market_price) / Vega(σ_n)

    We seed with the Brenner-Subrahmanyam approximation:
        σ₀ ≈ √(2π/T) · (market_price / S)
    which is accurate for ATM options.

    Returns None if no solution is found (e.g. price violates no-arb bounds).
    """
    if T <= 0:
        return None

    # Check no-arbitrage bounds
    intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        if option_type == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    )
    if market_price <= intrinsic or market_price <= 0:
        return None

    # Brenner-Subrahmanyam seed
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = np.clip(sigma, 1e-4, 10.0)

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        v = vega(S, K, T, r, sigma, q) * 100  # full vega for N-R step

        if abs(v) < 1e-10:
            break

        diff = price - market_price
        sigma -= diff / v

        if sigma <= 0:
            sigma = 1e-4

        if abs(diff) < tol:
            return sigma

    return None


# ---------------------------------------------------------------------------
# Strategy Builder
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """A single option or stock position within a multi-leg strategy."""
    option_type: str          # "call", "put", or "stock"
    K: float                  # strike (ignored for stock legs)
    T: float                  # expiry in years (ignored for stock legs)
    direction: str            # "long" or "short"
    quantity: float = 1.0


STRATEGIES = {
    "Long Call":        [Leg("call",  100, 0.25, "long")],
    "Long Put":         [Leg("put",   100, 0.25, "long")],
    "Short Call":       [Leg("call",  100, 0.25, "short")],
    "Short Put":        [Leg("put",   100, 0.25, "short")],
    "Long Straddle":    [Leg("call",  100, 0.25, "long"),  Leg("put",  100, 0.25, "long")],
    "Short Straddle":   [Leg("call",  100, 0.25, "short"), Leg("put",  100, 0.25, "short")],
    "Long Strangle":    [Leg("call",  105, 0.25, "long"),  Leg("put",   95, 0.25, "long")],
    "Bull Call Spread": [Leg("call",   95, 0.25, "long"),  Leg("call", 105, 0.25, "short")],
    "Bear Put Spread":  [Leg("put",   105, 0.25, "long"),  Leg("put",   95, 0.25, "short")],
    "Iron Condor": [
        Leg("put",   90, 0.25, "long"),
        Leg("put",   95, 0.25, "short"),
        Leg("call", 105, 0.25, "short"),
        Leg("call", 110, 0.25, "long"),
    ],
}


def strategy_payoff(
    legs: list,
    S_range: np.ndarray,
    r: float,
    sigma: float,
    q: float = 0.0,
    current_S: float = 100.0,
) -> tuple:
    """
    Compute the net payoff at expiry and current P&L, plus aggregate Greeks.

    Returns
    -------
    payoff_at_expiry : array of net payoff values across S_range
    current_pnl     : array of mark-to-market P&L across S_range (T > 0)
    greeks          : aggregate Greeks of the full strategy at current_S
    """
    payoff = np.zeros_like(S_range, dtype=float)
    current_val = np.zeros_like(S_range, dtype=float)
    agg_greeks = {k: 0.0 for k in ["price", "delta", "gamma", "vega", "theta", "rho"]}

    for leg in legs:
        sign = 1.0 if leg.direction == "long" else -1.0

        if leg.option_type == "stock":
            payoff += sign * leg.quantity * S_range
            current_val += sign * leg.quantity * S_range
            agg_greeks["delta"] += sign * leg.quantity
            continue

        if leg.option_type == "call":
            leg_payoff = np.maximum(S_range - leg.K, 0)
        else:
            leg_payoff = np.maximum(leg.K - S_range, 0)

        # Premium priced at current_S
        premium = bs_price(current_S, leg.K, leg.T, r, sigma, leg.option_type, q)
        payoff += sign * leg.quantity * (leg_payoff - premium)

        # Mark-to-market across spot range
        for i, s in enumerate(S_range):
            p = bs_price(s, leg.K, leg.T, r, sigma, leg.option_type, q)
            current_val[i] += sign * leg.quantity * (p - premium)

        # Aggregate Greeks at current_S
        g = all_greeks(current_S, leg.K, leg.T, r, sigma, leg.option_type, q)
        for k in agg_greeks:
            agg_greeks[k] += sign * leg.quantity * g[k]

    return payoff, current_val, agg_greeks
