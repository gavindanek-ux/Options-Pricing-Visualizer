# Black-Scholes Options Pricing Visualizer

Interactive dashboard for pricing European options and visualizing Greeks using the Black-Scholes-Merton model.

## Features

- BSM option pricing with full first and second-order Greeks (Δ, Γ, ν, Θ, ρ, Vanna, Volga, Charm)
- Greeks sensitivity plots vs. spot, volatility, and time
- Implied volatility surface with equity skew
- Multi-leg strategy P&L profiles (straddle, iron condor, spreads, etc.)
- Monte Carlo simulation vs. analytical BSM price
- Put-call parity verifier

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · Streamlit · Plotly · NumPy · SciPy
