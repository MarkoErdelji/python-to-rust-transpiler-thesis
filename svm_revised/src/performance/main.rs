//! ```cargo
//! [package]
//! edition = "2018"
//! [dependencies]
//! anyhow = "*"
//! ndarray = "*"
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
use ndarray::{Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

pub fn load_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let row_data: Vec<f64> = line?.split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let (feature_values, target_value) = row_data.split_at(row_data.len() - 1);
        features.push(feature_values.to_vec());
        targets.push(target_value[0]);
    }

    let num_features = features[0].len();
    let num_samples = features.len();
    let flattened_features: Vec<f64> = features.into_iter().flatten().collect();
    Ok((
        Array2::from_shape_vec((num_samples, num_features), flattened_features)?,
        Array1::from_vec(targets),
    ))
}

pub fn svm_soft_margin_train(
    X: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
    lambda_: f64,
    n_iterations: usize,
) -> (Array1<f64>, f64) {
    let (n_samples, n_features) = X.dim();
    let mut w = Array1::zeros(n_features);
    let mut b = 0.0;
    let mut epoch_times: Vec<f64> = Vec::new();

    for _ in 0..n_iterations {
        let epoch_start_time = Instant::now();
        
        for i in 0..n_samples {
            let Xi = X.row(i).to_owned();
            let yi = y[i];
            if yi * (Xi.dot(&w) - b) >= 1.0 {
                w = &w - alpha * (2.0 * lambda_ * &w);
            } else {
                w = &w - alpha * (2.0 * lambda_ * &w - yi * Xi);
                b = b - alpha * yi;
            }
        }

        let epoch_elapsed_time = epoch_start_time.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
        epoch_times.push(epoch_elapsed_time);
    }

    let file_path = "svm_epoch_times.csv";
    let mut file = File::create(file_path).expect("Could not create file");

    writeln!(file, "Epoch\tTime (ms)").expect("Could not write to file");
    for (epoch, time) in epoch_times.iter().enumerate() {
        writeln!(file, "{},{}", epoch + 1, time).expect("Could not write to file");
    }

    (w, b)
}

pub fn svm_soft_margin_predict(
    X: &Array2<f64>,
    w: &Array1<f64>,
    b: f64,
) -> Array1<i32> {
    let pred = X.dot(w) - b;
    pred.mapv(|val| if val > 0.0 { 1 } else { -1 })
}

pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<i32>) -> f64 {
    let correct_predictions = y_true.iter()
        .zip(y_pred.iter())
        .filter(|&(true_val, pred_val)| (*true_val == *pred_val as f64))
        .count();
    correct_predictions as f64 / y_true.len() as f64
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "extendedKNN.csv";
    let (X, y) = load_data(file_path)?;
    let y = y.mapv(|val| if val == 0.0 { -1.0 } else { 1.0 });

    let start_time = Instant::now();
    let (w, b) = svm_soft_margin_train(&X, &y, 0.001, 0.01, 1000);
    let predictions = svm_soft_margin_predict(&X, &w, b);
    let acc = accuracy(&y, &predictions);
    let elapsed_time = start_time.elapsed().as_secs_f64();
    
    println!("Predictions: {:?}", predictions);
    println!("Accuracy: {}", acc);
    println!("Training Time: {:.6} seconds", elapsed_time);

    Ok(())
}
