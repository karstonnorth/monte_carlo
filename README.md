# Monte Carlo 

Welcome to the **Monte Carlo Option Pricer** – a Python-based GUI application designed to simulate and price options using Monte Carlo methods. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Installation](#installation)

## Overview

The Monte Carlo Option Pricer uses simulation techniques to estimate the price of European-style options. It supports multiple models, including:

- **Geometric Brownian Motion (GBM):** The standard model for simulating stock prices.
- **Jump Diffusion:** Incorporates sudden jumps, capturing events like earnings surprises or market shocks.

The tool is built with:
- **Tkinter** for the GUI interface,
- **Matplotlib** for plotting simulations, histograms, and heatmaps,
- **NumPy** for efficient numerical calculations.

## Features

- **Interactive GUI:** User-friendly input with tooltips and themed styling.
- **Real-time Simulations:** Visualize simulated stock price paths.
- **Option Pricing:** Calculate call/put option prices with confidence intervals.
- **Distribution Analysis:** View histograms of the final stock prices.
- **Heatmap Visualization:** Examine the density of stock prices over time.
- **Customizable Parameters:** Adjust simulations, jump diffusion parameters, and more.
- **Save Plots:** Save your simulation plots for reporting or further analysis.

## Models Implemented

- **GBM (Geometric Brownian Motion):** Simulates continuous stock price changes.
- **Jump Diffusion:** Adding random jumps to mimic market discontinuities.

## Installation

Ensure you have Python 3.7 or above installed. Then, install the required packages using pip:

```bash
pip install numpy matplotlib tk
