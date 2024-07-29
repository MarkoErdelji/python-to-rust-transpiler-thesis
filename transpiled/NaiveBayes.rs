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
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;

pub fn fit_naive_bayes<T0, T1, RT>(X: T0, y: T1) -> RT {
    let class_counts = np.array(vec![np.sum((y as i32) == 0), np.sum((y as i32) == 1)]);
    let X_class_0 = X[(y as i32) == 0];
    let X_class_1 = X[(y as i32) == 1];
    let class_summaries: &HashMap<i32, _> = &[
        (0, (np.mean(X_class_0, 0), np.var(X_class_0, 0))),
        (1, (np.mean(X_class_1, 0), np.var(X_class_1, 0))),
    ]
    .iter()
    .cloned()
    .collect::<HashMap<_, _>>();
    return (class_summaries, class_counts);
}

pub fn calculate_probability<T0, T1, T2>(x: T0, mean: T1, var: T2) -> i32 {
    let exponent = np.exp(((-(pow((x - mean), 2)) as f64) / ((2 * (var as i32)) as f64)));
    return ((1 / (np.sqrt(((2 * (np.pi as i32)) * (var as i32))) as i32)) * (exponent as i32));
}

pub fn calculate_class_probabilities<T0, T1, T2, RT>(
    x: T0,
    class_summaries: T1,
    class_counts: T2,
) -> RT {
    let total_count = np.sum(class_counts);
    let prob_class_0 = (class_counts[0] / total_count);
    let prob_class_1 = (class_counts[1] / total_count);
    let (mean_0, var_0) = class_summaries[0];
    let (mean_1, var_1) = class_summaries[1];
    let probs_0: i32 = calculate_probability(x, mean_0, var_0);
    let probs_1: i32 = calculate_probability(x, mean_1, var_1);
    let prob_0 = (prob_class_0 * np.prod(probs_0));
    let prob_1 = (prob_class_1 * np.prod(probs_1));
    return np.array(vec![prob_0, prob_1]);
}

pub fn predict_naive_bayes<T0, T1, T2, RT>(X: T0, class_summaries: T1, class_counts: T2) -> RT {
    let mut predictions: List = vec![];
    for x in X {
        let probabilities = calculate_class_probabilities(x, class_summaries, class_counts);
        predictions.push(np.argmax(probabilities));
    }
    return np.array(predictions);
}

pub fn accuracy<T0, T1, RT>(y_true: T0, y_pred: T1) -> RT {
    return np.mean(y_true == y_pred);
}

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

pub fn main() -> Result<()> {
    let file_path: &'static str = "KNN.csv";
    let (X, y) = load_data(file_path);
    let (class_summaries, class_counts) = fit_naive_bayes(X, y);
    let predictions = predict_naive_bayes(X, class_summaries, class_counts);
    let acc = accuracy(y, predictions);
    println!("{} {}", "Predictions:", predictions);
    println!("{} {}", "Accuracy:", acc);
    Ok(())
}
