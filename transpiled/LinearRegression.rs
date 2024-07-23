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

extern crate anyhow;
use anyhow::Result;
use std::collections;

pub fn initialize_parameters(n_features: i32) {
    let weights = (vec![0.0] * n_features);
    let bias: f64 = 0.0;
    return (weights, bias);
}

pub fn predict(X: &Vec<Vec<f64>>, weights: &Vec<f64>, bias: f64) -> Vec<f64> {
    let mut predictions: List = vec![];
    for i in (0..X.len() as i32) {
        let prediction: f64 = (((0..X[0 as usize].len() as i32)
            .map(|j| ((X[i as usize][j] as f64) * weights[j as usize]))
            .collect::<Vec<_>>()
            .iter()
            .sum() as f64)
            + bias);
        predictions.push(prediction);
    }
    return predictions as Vec<f64>;
}

pub fn compute_gradients(X: &Vec<Vec<f64>>, y: &Vec<f64>, y_pred: &Vec<f64>, n_samples: i32) {
    let n_features = X[0 as usize].len() as i32;
    let mut dw: &mut Vec<f64> = &mut (vec![0.0] * n_features);
    let mut db: f64 = 0.0;
    for i in (0..n_samples) {
        let error: f64 = (y_pred[i as usize] - y[i as usize]);
        for j in (0..n_features) {
            dw[j as usize] +=
                ((((2 as f64) / (n_samples as f64)) * (X[i as usize][j] as f64)) * error);
        }
        db += (((2 as f64) / (n_samples as f64)) * error);
    }
    return (dw, db);
}

pub fn update_parameters(
    weights: &Vec<f64>,
    bias: f64,
    dw: &Vec<f64>,
    db: f64,
    learning_rate: f64,
) {
    for j in (0..weights.len() as i32) {
        weights[j as usize] -= (learning_rate * dw[j as usize]);
    }
    bias -= (learning_rate * db);
    return (weights, bias);
}

pub fn linear_regression(X: &Vec<Vec<f64>>, y: &Vec<f64>, learning_rate: f64, epochs: i32) {
    let (n_samples, n_features) = (X.len() as i32, X[0 as usize].len() as i32);
    let (weights, bias): &_ = &initialize_parameters(n_features);
    for _ in (0..epochs) {
        let y_pred: &Vec<f64> = &predict(X, &weights, bias);
        let (dw, db): &_ = &compute_gradients(X, y, y_pred, n_samples);
        let (weights, bias): &_ = &update_parameters(&weights, bias, &dw, db, learning_rate);
    }
    return (weights, bias);
}

pub fn main() -> Result<()> {
    let X: Vec<Vec<f64>> = vec![vec![1], vec![2], vec![3], vec![4], vec![5]];
    let y: Vec<f64> = vec![1, 2, 3, 4, 5];
    let learning_rate: f64 = 0.01;
    let epochs: i32 = 1000;
    let (weights, bias): &_ = &linear_regression(X, y, learning_rate, epochs);
    let predictions: &Vec<f64> = &predict(X, &weights, bias);
    println!("{} {}", "Predictions:", predictions);
    Ok(())
}
