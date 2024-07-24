//! ```cargo
//! [package]
//! edition = "2018"
//! [dependencies]
//! anyhow = "*"
//! ```

#![allow(clippy::collapsible_else_if)]
#![allow(clippy::double_parens)] // https://github.com/adsharma/py2many/issues/17
#![allow(clippy::map_identity)]
#![allow(clippy::needless_return)]
#![allow(clippy::print_literal)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::redundant_static_lifetimes)] // https://github.com/adsharma/py2many/issues/266
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::useless_vec)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_parens)]

use std::time::Instant;
use anyhow::Result;

pub fn initialize_parameters(n_features: usize) -> (Vec<f64>, f64) {
    let weights = vec![0.0; n_features];
    let bias: f64 = 0.0;
    (weights, bias)
}

pub fn predict(X: &Vec<Vec<f64>>, weights: &Vec<f64>, bias: f64) -> Vec<f64> {
    let mut predictions: Vec<f64> = Vec::new();
    for i in 0..X.len() {
        let prediction: f64 = X[i].iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() + bias;
        predictions.push(prediction);
    }
    predictions
}

pub fn compute_gradients(
    X: &Vec<Vec<f64>>, 
    y: &Vec<f64>, 
    y_pred: &Vec<f64>, 
    n_samples: usize
) -> (Vec<f64>, f64) {
    let n_features = X[0].len();
    let mut dw: Vec<f64> = vec![0.0; n_features];
    let mut db: f64 = 0.0;

    for i in 0..n_samples {
        let error = y_pred[i] - y[i];
        for j in 0..n_features {
            dw[j] += (2.0 / n_samples as f64) * X[i][j] * error;
        }
        db += (2.0 / n_samples as f64) * error;
    }
    (dw, db)
}

pub fn update_parameters(
    mut weights: Vec<f64>, 
    mut bias: f64, 
    dw: &Vec<f64>, 
    db: f64, 
    learning_rate: f64
) -> (Vec<f64>, f64) {
    for j in 0..weights.len() {
        weights[j] -= learning_rate * dw[j];
    }
    bias -= learning_rate * db;
    (weights, bias)
}

pub fn linear_regression(
    X: &Vec<Vec<f64>>, 
    y: &Vec<f64>, 
    learning_rate: f64, 
    epochs: usize
) -> (Vec<f64>, f64) {
    let n_samples = X.len();
    let n_features = X[0].len();
    let (mut weights, mut bias) = initialize_parameters(n_features);

    for _ in 0..epochs {
        let y_pred = predict(X, &weights, bias);
        let (dw, db) = compute_gradients(X, y, &y_pred, n_samples);
        let (updated_weights, updated_bias) = update_parameters(weights, bias, &dw, db, learning_rate);
        weights = updated_weights;
        bias = updated_bias;
    }
    (weights, bias)
}

pub fn main() -> Result<()> {
    let X: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let learning_rate: f64 = 0.01;
    let epochs: usize = 1000;
    let (weights, bias) = linear_regression(&X, &y, learning_rate, epochs);
    let predictions = predict(&X, &weights, bias);

    println!("Predictions: {:?}", predictions);

    Ok(())
}
