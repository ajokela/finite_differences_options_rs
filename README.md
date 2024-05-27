# Finite Differences Options Pricing in Rust

This repository provides a Rust library for solving the option pricing problem using the Crank-Nicolson method. The library can be used as a Python module through the `pyo3` and `numpy` crates, making it easy to integrate with Python-based financial analysis workflows.

## Features

- Construct tridiagonal matrices for numerical solutions
- Solve option pricing problems using the Crank-Nicolson method
- Calculate option Greeks (delta, gamma, theta, vega, and rho)
- Extract option prices from the grid

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/finite_differences_options_rs.git
   cd finite_differences_options_rs
   ```

2. Build the Python module:

   ```bash
   cargo build
   ```

3. Test the installation:

   ```bash
   python -c "import libfinite_differences_options_rs; print('Module loaded successfully!')"
   ```

## Usage

### Constructing a Tridiagonal Matrix

```rust
use nalgebra as na;
use na::DMatrix;

fn tridiag(n: usize, a: f64, b: f64, c: f64) -> DMatrix<f64> {
    let mut mat = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if i > 0 {
            mat[(i, i - 1)] = a;
        }
        mat[(i, i)] = b;
        if i < n - 1 {
            mat[(i, i + 1)] = c;
        }
    }
    mat
}
```

### Solving the Option Pricing Problem

```python
import libfinite_differences_options_rs as fdor

s0 = 100.0  # Initial stock price
k = 100.0   # Strike price
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility
t = 1.0     # Time to maturity
n = 100     # Number of time steps
m = 100     # Number of stock price steps

option_prices = fdor.crank_nicolson(s0, k, r, sigma, t, n, m)
print(option_prices)
```

### Calculating Option Greeks

```python
greeks = fdor.calculate_greeks(s0, k, r, sigma, t, n, m)
delta, gamma, theta, vega, rho = greeks
print(f"Delta: {delta}, Gamma: {gamma}, Theta: {theta}, Vega: {vega}, Rho: {rho}")
```

### Extracting Option Price

```python
initial_price = fdor.extract_option_price(option_prices, s0, k, r, sigma, t, n, m)
print(f"Option Price at Initial Time Step: {initial_price}")
```
