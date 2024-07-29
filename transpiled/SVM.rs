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
    let mut features: List = vec![];
    let mut targets: List = vec![];
    ({
        let file = OpenOptions::new().read(true).open(file_path)?;
        let lines = file.readlines();
        for line in lines[1..] {
            let row_data = line
                .strip()
                .split(",")
                .iter()
                .map(float)
                .collect::<Vec<_>>();
            let feature_values = row_data[..-1];
            let target_value = row_data[-1];
            features.push(feature_values);
            targets.push(target_value);
        }
    });
    return (np.array(features), np.array(targets));
}

pub fn svm_soft_margin_train<T0, T1, T2, T3, T4, RT>(
    X: T0,
    y: T1,
    alpha: T2,
    lambda_: T3,
    n_iterations: T4,
) -> RT {
    let (n_samples, n_features) = X.shape;
    let mut w = np.zeros(n_features);
    let mut b: i32 = 0;
    for iteration in (0..n_iterations) {
        for (i, Xi) in X.iter().enumerate() {
            if ((y[i] * (np.dot(Xi, w) - b)) as i32) >= 1 {
                w -= ((alpha as i32) * ((2 * (lambda_ as i32)) * w));
            } else {
                w -= ((alpha as i32) * (((2 * (lambda_ as i32)) * w) - (np.dot(Xi, y[i]) as i32)));
                b -= (alpha * y[i]);
            }
        }
    }
    return (w, b);
}

pub fn svm_soft_margin_predict<T0, T1, T2, RT>(X: T0, w: T1, b: T2) -> RT {
    let pred = (np.dot(X, w) - b);
    let result = pred
        .iter()
        .map(|val| if (val as i32) > 0 { 1 } else { -1 })
        .collect::<Vec<_>>();
    return result;
}

pub fn accuracy<T0, T1, RT>(y_true: T0, y_pred: T1) -> RT {
    return np.mean(np.array(y_true) == np.array(y_pred));
}

pub fn main() -> Result<()> {
    let file_path: &'static str = "KNN.csv";
    let (X, y) = load_data(file_path);
    let mut y = np.where_((y as i32) == 0, -1, 1);
    let (w, b) = svm_soft_margin_train(X, y);
    let predictions = svm_soft_margin_predict(X, w, b);
    let acc = accuracy(y, predictions);
    println!("{} {}", "Predictions:", predictions);
    println!("{} {}", "Accuracy:", acc);
    Ok(())
}
