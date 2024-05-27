extern crate nalgebra as na;
use na::{DMatrix};
use numpy::{PyArray2, PyArray};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;

/// Constructs a tridiagonal matrix with specified diagonals.
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

/// Solves the option pricing problem using the Crank-Nicolson method.
/// 
/// # Arguments
/// 
/// * `py` - The Python interpreter instance.
/// * `s0` - Initial stock price.
/// * `k` - Strike price.
/// * `r` - Risk-free rate.
/// * `sigma` - Volatility.
/// * `t` - Time to maturity.
/// * `n` - Number of time steps.
/// * `m` - Number of stock price steps.
/// 
/// # Returns
/// 
/// A 2D numpy array containing the option prices.
#[pyfunction]
fn crank_nicolson(
    py: Python,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n: usize,
    m: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    // Time step size
    let dt = t / n as f64;
    // Spatial step size
    let dx = sigma * (t / n as f64).sqrt();
    // Drift term
    let nu = r - 0.5 * sigma.powi(2);
    // Square of spatial step size
    let dx2 = dx * dx;

    // Initialize the grid for option prices
    let mut grid = DMatrix::<f64>::zeros(n + 1, m + 1);

    // Set terminal condition (payoff at maturity)
    for j in 0..=m {
        let stock_price = s0 * (-((m as isize / 2) as f64 - j as f64) * dx).exp();
        grid[(n, j)] = (k - stock_price).max(0.0);
    }

    // Coefficients for the tridiagonal matrices
    let alpha = 0.25 * dt * (sigma.powi(2) / dx2 - nu / dx);
    let beta = -0.5 * dt * (sigma.powi(2) / dx2 + r);
    let gamma = 0.25 * dt * (sigma.powi(2) / dx2 + nu / dx);

    // Construct tridiagonal matrices A and B
    let a = tridiag(m + 1, -alpha, 1.0 - beta, -gamma);
    let b = tridiag(m + 1, alpha, 1.0 + beta, gamma);

    // Perform LU decomposition of matrix A
    let lu = a.lu();

    // Backward time-stepping to solve the PDE
    for i in (0..n).rev() {
        let rhs = &b * grid.row(i + 1).transpose();
        let sol = lu.solve(&rhs).ok_or_else(|| PyValueError::new_err("Failed to solve LU"))?;
        grid.set_row(i, &sol.transpose());
    }

    // Convert DMatrix to Vec<Vec<f64>> for PyArray2
    let rows = grid.nrows();
    let cols = grid.ncols();
    let mut vec_2d = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(grid[(i, j)]);
        }
        vec_2d.push(row);
    }

    // Create a numpy array from the Vec<Vec<f64>>
    let array = PyArray::from_vec2_bound(py, &vec_2d).map_err(|_| PyValueError::new_err("Failed to create numpy array"))?;
    Ok(array.unbind())
}

/// Calculates the Greeks (delta, gamma, theta, vega, and rho) for the option.
/// 
/// # Arguments
/// 
/// * `py` - The Python interpreter instance.
/// * `s0` - Initial stock price.
/// * `k` - Strike price.
/// * `r` - Risk-free rate.
/// * `sigma` - Volatility.
/// * `t` - Time to maturity.
/// * `n` - Number of time steps.
/// * `m` - Number of stock price steps.
/// 
/// # Returns
/// 
/// A tuple containing the values of delta, gamma, theta, vega, and rho.
#[pyfunction]
fn calculate_greeks(
    py: Python,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n: usize,
    m: usize,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let grid = crank_nicolson(py, s0, k, r, sigma, t, n, m)?;
    let grid = grid.bind(py).readonly();
    let ds = s0 * (sigma * (t / n as f64).sqrt()).exp();
    let dt = t / n as f64;

    let price_idx = m / 2;
    let price = grid.as_slice().unwrap()[price_idx];
    let price_up = grid.as_slice().unwrap()[price_idx + 1];
    let price_down = grid.as_slice().unwrap()[price_idx - 1];

    // Calculate delta, gamma, and theta
    let delta = (price_up - price_down) / (2.0 * ds);
    let gamma = (price_up - 2.0 * price + price_down) / (ds * ds);
    let theta = (grid.as_slice().unwrap()[price_idx + 1] - price) / dt;

    // Calculate vega
    let eps = 0.01;
    let sigma_vega = sigma + eps;
    let grid_vega = crank_nicolson(py, s0, k, r, sigma_vega, t, n, m)?;
    let grid_vega = grid_vega.bind(py).readonly();
    let price_vega = grid_vega.as_slice().unwrap()[price_idx];
    let vega = (price_vega - price) / eps;

    // Calculate rho
    let r_rho = r + eps;
    let grid_rho = crank_nicolson(py, s0, k, r_rho, sigma, t, n, m)?;
    let grid_rho = grid_rho.bind(py).readonly();
    let price_rho = grid_rho.as_slice().unwrap()[price_idx];
    let rho = (price_rho - price) / eps;

    Ok((delta, gamma, theta, vega, rho))
}

/// Extracts the option price from the grid at the initial time step for the given parameters.
/// 
/// # Arguments
/// 
/// * `py` - The Python interpreter instance.
/// * `grid` - The option price grid.
/// * `_s0` - Initial stock price (not used).
/// * `_k` - Strike price (not used).
/// * `_r` - Risk-free rate (not used).
/// * `_sigma` - Volatility (not used).
/// * `_t` - Time to maturity (not used).
/// * `_n` - Number of time steps (not used).
/// * `m` - Number of stock price steps.
/// 
/// # Returns
/// 
/// The option price at the initial time step.
#[pyfunction]
fn extract_option_price(
    py: Python,
    grid: Py<PyArray2<f64>>,
    _s0: f64,
    _k: f64,
    _r: f64,
    _sigma: f64,
    _t: f64,
    _n: usize,
    m: usize,
) -> PyResult<f64> {
    let grid = grid.bind(py).readonly();
    // Assuming s0 is near the middle of the grid
    let price_idx = m / 2;
    Ok(grid.as_slice().unwrap()[price_idx])
}

/// Module definition for the finite_differences_options_rs Python module.
#[pymodule]
fn libfinite_differences_options_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crank_nicolson, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(extract_option_price, m)?)?;
    Ok(())
}
