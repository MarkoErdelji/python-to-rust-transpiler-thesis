//! ```cargo
//! [package]
//! edition = "2018"
//! [dependencies]
//! anyhow = "*"
//! numpy = "*"
//! pylib = "*"
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
extern crate numpy;
extern crate pylib;
use anyhow::Result;
use pylib::FileReadString;
use std::collections;
use std::fs::File;
use std::fs::OpenOptions;

pub fn load_data<T0, RT>(file_path: T0) -> RT {
    ({
        let file = OpenOptions::new().read(true).open(file_path)?;
        let lines = file.readlines();
    });
    let header = lines[0].strip().split(",");
    let mut data = lines[1..]
        .iter()
        .map(|line| line.strip().split(","))
        .collect::<Vec<_>>();
    data = np.array(data, str);
    return data;
}

pub fn convert_to_float<T0, RT>(data: T0) -> RT {
    return data.astype(float);
}

pub fn feature_scaling<T0, RT>(X: T0) -> RT {
    let means = np.mean(X, 0);
    let stds = np.std(X, 0);
    return ((X - means) / stds);
}

pub fn split_data<T0, T1, RT>(data: T0, test_ratio: T1) -> RT {
    let num_samples = data.shape[0];
    let num_test_samples: i32 = (num_samples * test_ratio) as i32;
    let X_train = data[(..-(num_test_samples), ..-1)];
    let y_train = data[(..-(num_test_samples), -1)];
    let X_test = data[(-(num_test_samples).., ..-1)];
    let y_test = data[(-(num_test_samples).., -1)];
    return (X_train, y_train, X_test, y_test);
}

pub fn sigmoid<T0>(z: T0) -> f64 {
    z = np.clip(z, -500, 500);
    return ((1 as f64) / ((1 + (np.exp(-(z)) as i32)) as f64)) as f64;
}

pub fn compute_cost<T0, T1, T2>(X: T0, y: T1, theta: T2) -> i32 {
    let m = y.len() as i32;
    let mut h: f64 = sigmoid(X.dot(theta));
    h = np.clip(h, 1e-10, ((1 as f64) - 1e-10));
    let cost: i32 =
        ((-1 / (m as i32)) * ((y.dot(np.log(h)) + (1 - y).dot(np.log(((1 as f64) - h)))) as i32));
    return cost;
}

pub fn gradient<T0, T1, T2>(X: T0, y: T1, theta: T2) -> i32 {
    let m = y.len() as i32;
    let h: f64 = sigmoid(X.dot(theta));
    let grad: i32 = ((1 / (m as i32)) * (X.T.dot((h - (y as f64))) as i32));
    return grad;
}

pub fn gradient_descent<T0, T1, T2, T3, T4, RT>(
    X: T0,
    y: T1,
    theta: T2,
    alpha: T3,
    num_iters: T4,
) -> RT {
    let m = y.len() as i32;
    let mut cost_history = np.zeros(num_iters);
    for i in (0..num_iters) {
        theta -= ((alpha as i32) * gradient(X, y, theta));
        cost_history[i] = compute_cost(X, y, theta);
    }
    return (theta, cost_history);
}

pub fn predict<T0, T1>(X: T0, theta: T1) -> bool {
    return sigmoid(X.dot(theta)) >= 0.5;
}

pub fn main() -> Result<()> {
    let mut data = load_data("data.csv");
    data = convert_to_float(data);
    let (X_train, y_train, X_test, y_test) = split_data(data);
    let mut X_train = feature_scaling(X_train);
    let mut X_test = feature_scaling(X_test);
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train));
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test));
    let mut theta = np.zeros(X_train.shape[1]);
    let alpha: f64 = 0.01;
    let num_iters: i32 = 2000;
    let (theta, cost_history) = gradient_descent(X_train, y_train, theta, alpha, num_iters);
    println!(
        "{}",
        vec!["Final cost: ", &cost_history[-1].to_string()].join("")
    );
    println!(
        "{}",
        vec!["Optimal parameters: ", &theta.to_string()].join("")
    );
    let train_predictions: bool = predict(X_train, theta);
    let test_predictions: bool = predict(X_test, theta);
    let train_accuracy = ((train_predictions.mapv(|v| v as u8) == y_train.mapv(|v| v as u8))
    .mapv(|v| v as f64)
    .mean()
    .unwrap()
    * 100.0) as i32;

let test_accuracy = ((test_predictions.mapv(|v| v as u8) == y_test.mapv(|v| v as u8))
    .mapv(|v| v as f64)
    .mean()
    .unwrap()
    * 100.0) as i32;
    println!(
        "{}",
        vec!["Training Accuracy: ", &train_accuracy.to_string(), "%"].join("")
    );
    println!(
        "{}",
        vec!["Test Accuracy: ", &test_accuracy.to_string(), "%"].join("")
    );
    Ok(())
}
