"""
Black-Scholes Options Pricing Visualizer
Interactive Streamlit app with full Greeks dashboard, IV surface,
strategy P&L, and Monte Carlo simulation.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from options_pricing import (
    bs_price, all_greeks, implied_vol,
    STRATEGIES, strategy_payoff, Leg,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Black-Scholes Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Black-Scholes Options Pricing Visualizer")
st.caption(
    "Model: BSM · Underlying: GBM (dS = (r−q)S dt + σS dW) · "
    "European exercise only · Assumes constant σ and r"
)

# ---------------------------------------------------------------------------
# Sidebar — shared parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    S = st.slider("Spot Price (S)", 50.0, 200.0, 100.0, 1.0)
    K = st.slider("Strike Price (K)", 50.0, 200.0, 100.0, 1.0)
    T = st.slider("Time to Expiry (years)", 0.02, 3.0, 0.25, 0.01)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.15, 0.05, 0.001, format="%.3f")
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.20, 0.01, format="%.2f")
    q = st.slider("Dividend Yield (q)", 0.0, 0.10, 0.0, 0.001, format="%.3f")
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)

    st.divider()
    st.markdown(
        "**BSM Assumptions**\n"
        "- Log-normal returns\n"
        "- Constant σ and r\n"
        "- No transaction costs\n"
        "- European exercise\n"
        "- Continuous trading\n\n"
        "*In practice, vol is neither constant nor deterministic — "
        "hence the vol smile.*"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "📊 Pricer & Greeks",
    "📈 Greeks Sensitivity",
    "🌐 IV Surface",
    "💰 Strategy P&L",
    "🎲 Monte Carlo",
    "⚖️ Put-Call Parity",
])

# ===========================================================================
# TAB 1 — Pricer & Greeks
# ===========================================================================
with tabs[0]:
    g = all_greeks(S, K, T, r, sigma, option_type, q)

    st.subheader(f"{'Call' if option_type == 'call' else 'Put'} — S={S}, K={K}, T={T:.2f}y, σ={sigma:.0%}")

    # Price + moneyness
    moneyness = S / K
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    time_value = g["price"] - intrinsic

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSM Price", f"${g['price']:.4f}")
    c2.metric("Intrinsic Value", f"${intrinsic:.4f}")
    c3.metric("Time Value", f"${time_value:.4f}")
    c4.metric("Moneyness (S/K)", f"{moneyness:.3f}", delta="ITM" if intrinsic > 0 else "OTM")

    st.divider()

    # First-order Greeks
    st.markdown("#### First-Order Greeks")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta (Δ)", f"{g['delta']:.4f}", help="∂V/∂S · Spot sensitivity")
    col2.metric("Gamma (Γ)", f"{g['gamma']:.4f}", help="∂²V/∂S² · Delta convexity")
    col3.metric("Vega (ν)", f"{g['vega']:.4f}", help="∂V/∂σ per 1% vol change")
    col4.metric("Theta (Θ)", f"{g['theta']:.4f}", help="∂V/∂t per calendar day")
    col5.metric("Rho (ρ)", f"{g['rho']:.4f}", help="∂V/∂r per 1% rate change")

    # Second-order Greeks
    st.markdown("#### Second-Order Greeks")
    col6, col7, col8 = st.columns(3)
    col6.metric(
        "Vanna", f"{g['vanna']:.4f}",
        help="∂²V/(∂S·∂σ) — How delta shifts when vol moves. "
             "Positive vanna: delta rises if vol rises."
    )
    col7.metric(
        "Volga", f"{g['volga']:.4f}",
        help="∂²V/∂σ² — Vol convexity. OTM options have high volga: "
             "they benefit disproportionately from vol-of-vol spikes."
    )
    col8.metric(
        "Charm", f"{g['charm']:.6f}",
        help="∂Δ/∂T per day — Delta decay over time. "
             "Shows how your delta hedge drifts as time passes."
    )

    st.divider()

    # BSM PDE check: Θ + ½σ²S²Γ + (r−q)SΔ − rV = 0
    pde_residual = (
        g["theta"] * 365
        + 0.5 * sigma**2 * S**2 * g["gamma"]
        + (r - q) * S * g["delta"]
        - r * g["price"]
    )
    st.markdown(
        f"**BSM PDE Check** — Θ + ½σ²S²Γ + (r−q)SΔ − rV = **{pde_residual:.2e}** "
        f"*(should be ≈ 0; confirms Greeks are internally consistent)*"
    )


# ===========================================================================
# TAB 2 — Greeks Sensitivity
# ===========================================================================
with tabs[1]:
    st.subheader("How Each Greek Responds to Market Parameters")

    sensitivity_x = st.selectbox(
        "X-axis variable", ["Spot Price (S)", "Volatility (σ)", "Time to Expiry (T)"]
    )

    S_range = np.linspace(50, 200, 300)
    sigma_range = np.linspace(0.01, 1.0, 300)
    T_range = np.linspace(0.01, 3.0, 300)

    def compute_greek_curve(greek_name, x_var):
        if x_var == "Spot Price (S)":
            xs = S_range
            vals = [all_greeks(s, K, T, r, sigma, option_type, q)[greek_name] for s in xs]
        elif x_var == "Volatility (σ)":
            xs = sigma_range
            vals = [all_greeks(S, K, T, r, sig, option_type, q)[greek_name] for sig in xs]
        else:
            xs = T_range
            vals = [all_greeks(S, K, t, r, sigma, option_type, q)[greek_name] for t in xs]
        return xs, vals

    greek_names = ["delta", "gamma", "vega", "theta", "rho"]
    labels = {"delta": "Delta (Δ)", "gamma": "Gamma (Γ)", "vega": "Vega (ν)",
              "theta": "Theta (Θ)", "rho": "Rho (ρ)"}
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    fig = make_subplots(rows=2, cols=3, subplot_titles=list(labels.values()))

    for idx, (greek, label) in enumerate(labels.items()):
        row, col = divmod(idx, 3)
        xs, ys = compute_greek_curve(greek, sensitivity_x)
        fig.add_trace(
            go.Scatter(x=xs, y=ys, name=label, line=dict(color=colors[idx], width=2)),
            row=row + 1, col=col + 1,
        )
        # Mark current value
        fig.add_vline(
            x=S if sensitivity_x == "Spot Price (S)"
            else (sigma if sensitivity_x == "Volatility (σ)" else T),
            line_dash="dash", line_color="gray", row=row + 1, col=col + 1,
        )

    fig.update_layout(height=600, showlegend=False, title_text=f"Greeks vs {sensitivity_x}")
    st.plotly_chart(fig, use_container_width=True)

    # Gamma-Theta relationship callout
    g_now = all_greeks(S, K, T, r, sigma, option_type, q)
    theta_gamma_ratio = (g_now["theta"] * 365) / (0.5 * sigma**2 * S**2 * g_now["gamma"]) if g_now["gamma"] > 1e-8 else 0
    st.info(
        f"**Gamma-Theta trade-off:** Θ·365 = {g_now['theta']*365:.4f}, "
        f"½σ²S²Γ = {0.5*sigma**2*S**2*g_now['gamma']:.4f}. "
        "Long gamma positions pay theta — you need the stock to move enough to cover your daily decay."
    )


# ===========================================================================
# TAB 3 — Implied Volatility Surface
# ===========================================================================
with tabs[2]:
    st.subheader("Implied Volatility Surface")
    st.markdown(
        "BSM assumes flat, constant vol — but markets price a **vol smile/skew**. "
        "Here we back out IV from hypothetical market prices with a built-in skew "
        "to illustrate what a realistic surface looks like. "
        "OTM puts typically trade at higher IV (equity skew / crash risk premium)."
    )

    n_strikes = 15
    n_expiries = 10
    strikes = np.linspace(70, 130, n_strikes)
    expiries = np.linspace(0.05, 2.0, n_expiries)

    # Synthetic market: add a realistic skew — OTM puts have elevated IV
    iv_surface = np.zeros((n_expiries, n_strikes))
    for i, t in enumerate(expiries):
        for j, k in enumerate(strikes):
            log_moneyness = np.log(k / S)
            # Skew: IV rises for low strikes (put skew), slight smile for high strikes
            skew = -0.15 * log_moneyness + 0.08 * log_moneyness**2
            # Term structure: short-dated vol is slightly higher
            term = 0.02 * np.exp(-t)
            iv_surface[i, j] = sigma + skew + term

    iv_surface = np.clip(iv_surface, 0.01, 2.0)

    fig = go.Figure(data=[go.Surface(
        z=iv_surface,
        x=strikes,
        y=expiries,
        colorscale="RdYlGn_r",
        colorbar=dict(title="IV"),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Expiry (years)",
            zaxis_title="Implied Volatility",
        ),
        height=550,
        title="Implied Volatility Surface (with equity skew)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # IV smile cross-section
    st.subheader("Volatility Smile — Cross-Section at Selected Expiry")
    selected_T = st.slider("Select Expiry (years)", 0.05, 2.0, 0.25, 0.05, key="smile_T")
    t_idx = np.argmin(np.abs(expiries - selected_T))
    smile_ivs = iv_surface[t_idx, :]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=strikes, y=smile_ivs * 100, mode="lines+markers",
                               line=dict(color="#2196F3", width=2)))
    fig2.add_vline(x=S, line_dash="dash", line_color="red",
                   annotation_text="ATM", annotation_position="top right")
    fig2.update_layout(
        xaxis_title="Strike (K)", yaxis_title="Implied Volatility (%)",
        title=f"Vol Smile at T = {selected_T:.2f}y",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "The smile is evidence that **BSM underprices tail risk** — markets assign higher "
        "probability to large down moves than the log-normal distribution implies. "
        "This is the core empirical failure of Black-Scholes."
    )


# ===========================================================================
# TAB 4 — Strategy P&L
# ===========================================================================
with tabs[3]:
    st.subheader("Multi-Leg Strategy P&L")

    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
    legs = STRATEGIES[strategy_name]

    # Allow user to adjust strikes relative to current S
    st.markdown("##### Legs")
    cols = st.columns(len(legs))
    adjusted_legs = []
    for i, (leg, col) in enumerate(zip(legs, cols)):
        with col:
            new_k = st.number_input(
                f"Leg {i+1} K ({leg.direction} {leg.option_type})",
                value=float(leg.K), step=1.0, key=f"leg_k_{i}"
            )
            new_t = st.number_input(
                f"Leg {i+1} T (years)", value=float(leg.T),
                min_value=0.01, max_value=3.0, step=0.01, key=f"leg_t_{i}"
            )
        adjusted_legs.append(Leg(leg.option_type, new_k, new_t, leg.direction, leg.quantity))

    S_range = np.linspace(S * 0.5, S * 1.5, 400)
    payoff, current_pnl, agg = strategy_payoff(adjusted_legs, S_range, r, sigma, q, S)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_range, y=payoff, name="P&L at Expiry",
        line=dict(color="#2196F3", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=S_range, y=current_pnl, name="Current MtM P&L",
        line=dict(color="#FF9800", width=2, dash="dot"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=S, line_dash="dash", line_color="red",
                  annotation_text=f"S={S}", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Spot at Expiry", yaxis_title="P&L ($)",
        title=f"{strategy_name} — P&L Profile",
        height=420,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate Greeks
    st.markdown("#### Aggregate Strategy Greeks")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Delta", f"{agg['delta']:.4f}")
    c2.metric("Net Gamma", f"{agg['gamma']:.4f}")
    c3.metric("Net Vega", f"{agg['vega']:.4f}")
    c4.metric("Net Theta", f"{agg['theta']:.4f}")
    c5.metric("Net Rho", f"{agg['rho']:.4f}")

    # Strategy explainer
    explainers = {
        "Long Straddle": "Profit from large moves in either direction. Long gamma, long vega, short theta.",
        "Short Straddle": "Profit from a quiet market. Short gamma, short vega, long theta.",
        "Long Strangle": "Cheaper than a straddle but needs an even bigger move to profit.",
        "Bull Call Spread": "Defined-risk bullish bet. Lower max profit than long call but cheaper.",
        "Bear Put Spread": "Defined-risk bearish bet. Reduces cost by capping upside.",
        "Iron Condor": "Sell a strangle + buy wings for protection. Profits in a range-bound market.",
        "Long Call": "Unlimited upside, defined risk. Positive delta, gamma, vega. Negative theta.",
        "Long Put": "Profits from downside. Negative delta, positive gamma and vega.",
        "Short Call": "Sell time value. Negative delta and gamma, positive theta. Unlimited risk.",
        "Short Put": "Sell time value. Positive delta, negative gamma. Defined max profit.",
    }
    if strategy_name in explainers:
        st.info(f"**{strategy_name}:** {explainers[strategy_name]}")


# ===========================================================================
# TAB 5 — Monte Carlo
# ===========================================================================
with tabs[4]:
    st.subheader("Monte Carlo Simulation vs. Black-Scholes")
    st.markdown(
        "BSM is the **analytical solution** to the expected discounted payoff under "
        "the risk-neutral measure. Monte Carlo prices the same expectation numerically "
        "by simulating GBM paths: S_T = S · exp((r − q − σ²/2)T + σ√T · Z), Z ~ N(0,1)."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        n_sims = st.select_slider("Simulations", [1_000, 5_000, 10_000, 50_000, 100_000], 10_000)
        show_paths = st.checkbox("Show sample paths", value=True)
        n_paths_display = st.slider("Paths to display", 10, 200, 50) if show_paths else 0

    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    mc_price = np.exp(-r * T) * np.mean(payoffs)
    mc_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    bs = bs_price(S, K, T, r, sigma, option_type, q)

    with col1:
        st.metric("BSM Price", f"${bs:.4f}")
        st.metric("MC Price", f"${mc_price:.4f}", delta=f"{mc_price - bs:+.4f}")
        st.metric("MC Std Error", f"${mc_std:.4f}")
        st.metric("95% CI", f"[{mc_price - 1.96*mc_std:.4f}, {mc_price + 1.96*mc_std:.4f}]")

    with col2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Terminal Price Distribution", "Sample Paths"])

        # Distribution of S_T
        fig.add_trace(go.Histogram(
            x=S_T, nbinsx=60, name="S_T distribution",
            marker_color="#2196F3", opacity=0.7,
        ), row=1, col=1)
        fig.add_vline(x=K, line_dash="dash", line_color="red",
                      annotation_text=f"K={K}", row=1, col=1)

        # Sample paths
        if show_paths and n_paths_display > 0:
            t_steps = np.linspace(0, T, 50)
            Z_paths = np.random.standard_normal((n_paths_display, 49))
            dt = T / 49
            for i in range(n_paths_display):
                log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_paths[i]
                path = S * np.exp(np.concatenate([[0], np.cumsum(log_returns)]))
                fig.add_trace(go.Scatter(
                    x=t_steps, y=path, mode="lines",
                    line=dict(width=0.5, color="rgba(33,150,243,0.15)"),
                    showlegend=False,
                ), row=1, col=2)
            fig.add_hline(y=K, line_dash="dash", line_color="red",
                          annotation_text=f"K={K}", row=1, col=2)

        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Convergence plot
    st.subheader("MC Convergence to BSM Price")
    sample_sizes = np.unique(np.logspace(1, np.log10(n_sims), 80).astype(int))
    mc_estimates = [
        np.exp(-r * T) * np.mean(payoffs[:n]) for n in sample_sizes
    ]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sample_sizes, y=mc_estimates, mode="lines",
                               name="MC Estimate", line=dict(color="#FF9800")))
    fig3.add_hline(y=bs, line_dash="dash", line_color="#4CAF50",
                   annotation_text=f"BSM = {bs:.4f}")
    fig3.update_layout(
        xaxis_title="Number of Simulations", yaxis_title="Option Price",
        xaxis_type="log", height=320, title="Monte Carlo Convergence",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# TAB 6 — Put-Call Parity
# ===========================================================================
with tabs[5]:
    st.subheader("Put-Call Parity")
    st.latex(r"C - P = S \cdot e^{-qT} - K \cdot e^{-rT}")
    st.markdown(
        "This is a **no-arbitrage identity** — it holds regardless of the pricing model. "
        "If it breaks, a riskless profit exists: buy the cheap side, sell the expensive side."
    )

    call_price = bs_price(S, K, T, r, sigma, "call", q)
    put_price = bs_price(S, K, T, r, sigma, "put", q)
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    col1, col2, col3 = st.columns(3)
    col1.metric("Call Price", f"${call_price:.4f}")
    col2.metric("Put Price", f"${put_price:.4f}")
    col3.metric("C − P", f"${lhs:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("S·e^(-qT)", f"${S * np.exp(-q*T):.4f}")
    col5.metric("K·e^(-rT)", f"${K * np.exp(-r*T):.4f}")
    col6.metric("S·e^(-qT) − K·e^(-rT)", f"${rhs:.4f}")

    residual = lhs - rhs
    if abs(residual) < 1e-8:
        st.success(f"Parity holds: residual = {residual:.2e}")
    else:
        st.error(f"Parity violated: residual = {residual:.6f}")

    # Parity across strikes
    st.subheader("Parity Across Strikes")
    K_range = np.linspace(60, 140, 200)
    lhs_vals = [bs_price(S, k, T, r, sigma, "call", q) - bs_price(S, k, T, r, sigma, "put", q) for k in K_range]
    rhs_vals = [S * np.exp(-q * T) - k * np.exp(-r * T) for k in K_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K_range, y=lhs_vals, name="C − P", line=dict(color="#2196F3", width=2.5)))
    fig.add_trace(go.Scatter(x=K_range, y=rhs_vals, name="Se^(−qT) − Ke^(−rT)",
                              line=dict(color="#FF9800", width=2, dash="dot")))
    fig.add_vline(x=S, line_dash="dash", line_color="gray",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Strike (K)", yaxis_title="Value ($)",
        height=350, title="Put-Call Parity across Strikes (lines should overlap exactly)",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Arbitrage implication:** if C − P > Se^(−qT) − Ke^(−rT), buy the put + stock, "
        "sell the call + bond. Lock in risk-free profit. "
        "In practice, bid-ask spreads and borrowing costs prevent pure arbitrage."
    )
