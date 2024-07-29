extern crate ndarray;

use ndarray::{Array2, Array1};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

pub fn fit_naive_bayes(X: &Array2<f64>, y: &Array1<i32>) -> (Vec<(Array1<f64>, Array1<f64>)>, Array1<f64>) {
    let class_counts = Array1::from_vec(vec![
        y.iter().filter(|&&label| label == 0).count() as f64,
        y.iter().filter(|&&label| label == 1).count() as f64,
    ]);

    let X_class_0 = X.select(ndarray::Axis(0), &y.iter().enumerate().filter_map(|(i, &label)| if label == 0 { Some(i) } else { None }).collect::<Vec<_>>());
    let X_class_1 = X.select(ndarray::Axis(0), &y.iter().enumerate().filter_map(|(i, &label)| if label == 1 { Some(i) } else { None }).collect::<Vec<_>>());

    let class_summaries = vec![
        (
            X_class_0.mean_axis(ndarray::Axis(0)).unwrap(),
            X_class_0.var_axis(ndarray::Axis(0), 0.0)
        ),
        (
            X_class_1.mean_axis(ndarray::Axis(0)).unwrap(),
            X_class_1.var_axis(ndarray::Axis(0), 0.0)
        ),
    ];

    (class_summaries, class_counts)
}

pub fn calculate_probability(x: &Array1<f64>, mean: &Array1<f64>, var: &Array1<f64>) -> Array1<f64> {
    let exponent = (-((x - mean) * (x - mean)) / (2.0 * var)).mapv(f64::exp);
    (1.0 / (2.0 * std::f64::consts::PI * var).mapv(f64::sqrt)) * exponent
}

pub fn calculate_class_probabilities(
    x: &Array1<f64>,
    class_summaries: &Vec<(Array1<f64>, Array1<f64>)>,
    class_counts: &Array1<f64>,
) -> Array1<f64> {
    let total_count = class_counts.sum();
    let prob_class_0 = class_counts[0] / total_count;
    let prob_class_1 = class_counts[1] / total_count;

    let (mean_0, var_0) = &class_summaries[0];
    let (mean_1, var_1) = &class_summaries[1];

    let probs_0 = calculate_probability(x, mean_0, var_0).product();
    let probs_1 = calculate_probability(x, mean_1, var_1).product();

    Array1::from_vec(vec![
        prob_class_0 * probs_0,
        prob_class_1 * probs_1,
    ])
}

pub fn predict_naive_bayes(
    X: &Array2<f64>,
    class_summaries: &Vec<(Array1<f64>, Array1<f64>)>,
    class_counts: &Array1<f64>,
) -> Array1<i32> {
    let mut predictions = Vec::with_capacity(X.nrows());

    for x in X.axis_iter(ndarray::Axis(0)) {
        let x = x.to_owned(); 
        let probabilities = calculate_class_probabilities(&x, class_summaries, class_counts);
        let max_index = probabilities.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap() as i32;

        predictions.push(max_index);
    }

    Array1::from_vec(predictions)
}

pub fn accuracy(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
    y_true.iter().zip(y_pred.iter())
        .filter(|&(true_label, pred_label)| true_label == pred_label)
        .count() as f64 / y_true.len() as f64
}

pub fn load_data(file_path: &str) -> Result<(Array2<f64>, Array1<i32>), Box<dyn Error>> {
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
        targets.push(target_value[0] as i32);
    }

    let num_features = features[0].len();
    let num_samples = features.len();
    let flattened_features: Vec<f64> = features.into_iter().flatten().collect();
    Ok((
        Array2::from_shape_vec((num_samples, num_features), flattened_features)?,
        Array1::from_vec(targets),
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    
    let file_path = "extendedKNN.csv";
    let (X, y) = load_data(file_path)?;
    
    let fit_start_time = Instant::now();
    let (class_summaries, class_counts) = fit_naive_bayes(&X, &y);
    let fit_time = fit_start_time.elapsed().as_secs_f64();
    
    let predict_start_time = Instant::now();
    let predictions = predict_naive_bayes(&X, &class_summaries, &class_counts);
    let predict_time = predict_start_time.elapsed().as_secs_f64();
    
    let acc = accuracy(&y, &predictions);
    let total_time = start_time.elapsed().as_secs_f64();

    println!("Predictions: {:?}", predictions);
    println!("Accuracy: {:.2}", acc);
    println!("Fit Time: {:.6} seconds", fit_time);
    println!("Prediction Time: {:.6} seconds", predict_time);
    println!("Total Execution Time: {:.6} seconds", total_time);

    Ok(())
}
