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

fn normalize_matrix(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_features = data[0].len();
    let mut mean: Vec<f64> = vec![0.0; num_features];
    let mut std_dev: Vec<f64> = vec![0.0; num_features];
    
    for row in data.iter() {
        for (j, &value) in row.iter().enumerate() {
            mean[j] += value;
        }
    }
    
    for m in mean.iter_mut() {
        *m /= data.len() as f64;
    }

    for row in data.iter() {
        for (j, &value) in row.iter().enumerate() {
            std_dev[j] += (value - mean[j]).powi(2);
        }
    }

    for s in std_dev.iter_mut() {
        *s = (*s / data.len() as f64).sqrt();
    }

    let mut normalized_data = vec![vec![0.0; num_features]; data.len()];
    for (i, row) in data.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            normalized_data[i][j] = (value - mean[j]) / std_dev[j];
        }
    }
    
    normalized_data
}

pub fn main() -> Result<()> {
    let num_samples = 10000;
    let num_features = 10;

    let X: Vec<Vec<f64>> = vec![vec![1.0; num_features]; num_samples];
    let y: Vec<f64> = vec![1.0; num_samples];

    let start_time = Instant::now();

    let learning_rate: f64 = 0.01;
    let epochs: usize = 1000;
    let (weights, bias) = linear_regression(&X, &y, learning_rate, epochs);
    let predictions = predict(&X, &weights, bias);

    let elapsed_time = start_time.elapsed().as_secs_f64();
    println!("Predictions: {:?}", predictions);
    println!("Execution Time: {:.6?} seconds", elapsed_time);

    Ok(())
}
