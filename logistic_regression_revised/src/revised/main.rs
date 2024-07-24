//! ```cargo
//! [package]
//! edition = "2018"
//! [dependencies]
//! anyhow = "*"
//! ndarray = "0.15"
//! ndarray-rand = "0.14"
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
extern crate ndarray;
use anyhow::Result;
use ndarray::s;
use ndarray::{Array, Array1, Array2, Axis};
use std::collections;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::time::Instant;

pub fn load_data(file_path: &str) -> Result<Array2<f64>> {
    let file = OpenOptions::new().read(true).open(file_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<_> = reader.lines().collect::<Result<_, _>>()?;
    let header = lines[0].trim().split(',').collect::<Vec<_>>();
    let data: Array2<f64> = Array::from_shape_vec(
        (lines.len() - 1, header.len()),
        lines[1..]
            .iter()
            .flat_map(|line| {
                line.trim()
                    .split(',')
                    .map(|entry| entry.parse().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect(),
    )?;
    Ok(data)
}

pub fn convert_to_float(data: Array2<f64>) -> Array2<f64> {
    data
}

pub fn feature_scaling(X: Array2<f64>) -> Array2<f64> {
    let means = X.mean_axis(Axis(0)).unwrap();
    let stds = X.std_axis(Axis(0), 0.0);
    (X - &means) / &stds
}

pub fn split_data(
    data: Array2<f64>,
    test_ratio: f64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let num_samples = data.nrows();
    let num_test_samples = (num_samples as f64 * test_ratio) as usize;
    let X_train = data.slice(s![..-(num_test_samples as isize), ..-1]).to_owned();
    let y_train = data.slice(s![..-(num_test_samples as isize), -1]).to_owned();
    let X_test = data.slice(s![-(num_test_samples as isize).., ..-1]).to_owned();
    let y_test = data.slice(s![-(num_test_samples as isize).., -1]).to_owned();
    (X_train, y_train, X_test, y_test)
}

pub fn sigmoid(z: Array1<f64>) -> Array1<f64> {
    let z = z.mapv(|a| a.clamp(-500.0, 500.0));
    (1.0 / (1.0 + (-z).mapv(f64::exp)))
}

pub fn compute_cost(X: &Array2<f64>, y: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let m = y.len() as f64;
    let h = sigmoid(X.dot(theta));
    let h_clipped = h.mapv(|v| v.clamp(1e-10, 1.0 - 1e-10));
    (-1.0 / m) * (y.dot(&h_clipped.mapv(f64::ln)) + (1.0 - y).dot(&(1.0 - h_clipped).mapv(f64::ln)))
}

pub fn gradient(X: &Array2<f64>, y: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
    let m = y.len() as f64;
    let h = sigmoid(X.dot(theta));
    X.t().dot(&(h - y)) / m
}

pub fn gradient_descent(
    X: &Array2<f64>,
    y: &Array1<f64>,
    mut theta: Array1<f64>,
    alpha: f64,
    num_iters: usize,
) -> (Array1<f64>, Array1<f64>) {
    let m = y.len() as f64;
    let mut cost_history = Array::zeros(num_iters);
    for i in 0..num_iters {
        theta -= &(alpha * &gradient(X, y, &theta));
        cost_history[i] = compute_cost(X, y, &theta);
    }
    (theta, cost_history)
}

pub fn predict(X: &Array2<f64>, theta: &Array1<f64>) -> Array1<bool> {
        let sigmoid_values = sigmoid(X.dot(theta));
        sigmoid_values.mapv(|value| value >= 0.5)
    }
pub fn main() -> Result<()> {
        let data = load_data("data.csv")?;
        let data = convert_to_float(data);
        let (X_train, y_train, X_test, y_test) = split_data(data, 0.2);
        let X_train = feature_scaling(X_train);
        let X_test = feature_scaling(X_test);
        let X_train = ndarray::concatenate![Axis(1), Array::ones((X_train.nrows(), 1)), X_train];
        let X_test = ndarray::concatenate![Axis(1), Array::ones((X_test.nrows(), 1)), X_test];
        let theta = Array::zeros(X_train.ncols());
        let alpha = 0.01;
        let num_iters = 2000;
        let (theta, cost_history) = gradient_descent(&X_train, &y_train, theta, alpha, num_iters);
        println!("Final cost: {}", cost_history[num_iters - 1]);
        println!("Optimal parameters: {:?}", theta);
        let train_predictions = predict(&X_train, &theta);
        let test_predictions = predict(&X_test, &theta);
    
                let train_correct: Array1<f64> = train_predictions
                .iter()
                .zip(y_train.iter())
                .map(|(&pred, &actual)| (pred == (actual != 0.0)) as u8 as f64)
                .collect::<Array1<_>>();
        let train_accuracy = (train_correct.mean().unwrap() * 100.0) as i32;

        let test_correct: Array1<f64> = test_predictions
                .iter()
                .zip(y_test.iter())
                .map(|(&pred, &actual)| (pred == (actual != 0.0)) as u8 as f64)
                .collect::<Array1<_>>();
        let test_accuracy = (test_correct.mean().unwrap() * 100.0) as i32;
        
        println!("Training Accuracy: {}%", train_accuracy);
        println!("Test Accuracy: {}%", test_accuracy);
        Ok(())
    }