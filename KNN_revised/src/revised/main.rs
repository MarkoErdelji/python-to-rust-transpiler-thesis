extern crate ndarray;
use ndarray::{Array, Array1, Array2, Axis, s};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct KNN {
    pub k: usize,
    pub X_train: Array2<f64>,
    pub y_train: Array1<f64>,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        KNN {
            k,
            X_train: Array2::zeros((0, 0)),
            y_train: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, X: Array2<f64>, y: Array1<f64>) {
        self.X_train = X;
        self.y_train = y;
    }

    pub fn predict(&self, X: Array2<f64>) -> Array1<f64> {
        X.axis_iter(Axis(0))
            .map(|x| self._predict(x.to_owned()))
            .collect::<Vec<f64>>()
            .into()
    }

    fn _predict(&self, x: Array1<f64>) -> f64 {
        let distances = self._compute_distances(&x);
        let mut k_indices: Vec<usize> = (0..distances.len()).collect();
        k_indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());
        k_indices.truncate(self.k);

        let mut label_counts = HashMap::new();
        for &index in &k_indices {
            let label = self.y_train[index].to_string(); // Convert to String for HashMap
            *label_counts.entry(label).or_insert(0) += 1;
        }

        let (most_common_label, _) = label_counts.into_iter().max_by_key(|&(_, count)| count).unwrap();
        most_common_label.parse::<f64>().unwrap() // Convert back to f64
    }

    fn _compute_distances(&self, x: &Array1<f64>) -> Array1<f64> {
        self.X_train.axis_iter(Axis(0))
            .map(|xi| {
                xi.iter()
                    .zip(x.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    }
}

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

pub fn calculate_accuracy(predictions: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let correct = predictions.iter()
        .zip(actual.iter())
        .filter(|&(pred, act)| (pred - act).abs() < f64::EPSILON)
        .count();
    (correct as f64 / predictions.len() as f64) * 100.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "KNN.csv";
    let (X, y) = load_data(file_path)?;
    let (X_train, X_test) = (X.slice(s![..-50, ..]).to_owned(), X.slice(s![-50.., ..]).to_owned());
    let (y_train, y_test) = (y.slice(s![..-50]).to_owned(), y.slice(s![-50..]).to_owned());

    let mut knn = KNN::new(3);
    knn.fit(X_train, y_train);
    let predictions = knn.predict(X_test);
    let accuracy = calculate_accuracy(&predictions, &y_test);

    println!("Predictions: {:?}", predictions);
    println!("Actual: {:?}", y_test);
    println!("Accuracy: {:.2}%", accuracy);

    Ok(())
}
