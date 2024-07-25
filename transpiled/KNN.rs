//! ```cargo
//! [package]
//! edition = "2018"
//! [dependencies]
//! anyhow = "*"
//! collections = "*"
//! numpy = "*"
//! pylib = "*"
//! ```

#![allow(clippy::collapsible_else_if)]
#![allow(clippy::double_parens)]  // https://github.com/adsharma/py2many/issues/17
#![allow(clippy::map_identity)]
#![allow(clippy::needless_return)]
#![allow(clippy::print_literal)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::redundant_static_lifetimes)]  // https://github.com/adsharma/py2many/issues/266
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
extern crate collections;
extern crate numpy;
extern crate pylib;

use anyhow::Result;
use pylib::FileReadString;
use std::collections;
use std::fs::File;
use std::fs::OpenOptions;

pub struct KNN {
    pub k: _,
    pub X_train: _,
    pub y_train: _,
}

impl KNN {
    pub fn __init__<T0>(&self, k: T0) {
        self.k = k;
    }

    pub fn fit<T0, T1>(&self, X: T0, y: T1) {
        self.X_train = np.array(X);
        self.y_train = np.array(y);
    }

    pub fn predict<T0, RT>(&self, X: T0) -> RT {
        X = np.array(X);
        return np.array(X.iter().map(|x| self._predict(x)).collect::<Vec<_>>());
    }

    pub fn _predict<T0, RT>(&self, x: T0) -> RT {
        pub const distances = self._compute_distances(x);
        pub const k_indices = np.argsort(distances)[..self.k];
        pub const k_nearest_labels = self.y_train[k_indices];
        pub const label_counts = Counter(k_nearest_labels);
        return label_counts.most_common(1)[0][0];
    }

    pub fn _compute_distances<T0, RT>(&self, x: T0) -> RT {
        return np.sqrt(np.sum(pow((self.X_train - x), 2), 1));
    }
}

pub fn load_data<T0, RT>(file_path: T0) -> RT {
    let (features, targets) = (vec![], vec![]);
    ({
        let file = OpenOptions::new().read(true).open(file_path)?;
        let lines = file.readlines();
        for line in lines[1..] {
            let row_data = line.strip().split(",").iter().map(float).collect::<Vec<_>>();
            let feature_values = row_data[..-1];
            let target_value = row_data[-1];
            features.append(feature_values);
            targets.append(target_value);
        }
    });
    return (np.array(features), np.array(targets));
}

pub fn calculate_accuracy<T0, T1>(predictions: T0, actual: T1) -> i32 {
    return ((np.mean(predictions == actual) as i32) * 100);
}

pub fn main() -> Result<()> {
    let file_path: &'static str = "KNN.csv";
    let (X, y) = load_data(file_path);
    let (X_train, X_test) = (X[..-50], X[-50..]);
    let (y_train, y_test) = (y[..-50], y[-50..]);
    let knn: KNN = KNN { k: 3 };
    knn.fit(X_train, y_train);
    let predictions = knn.predict(X_test);
    let accuracy: i32 = calculate_accuracy(predictions, y_test);
    println!("{} {}", "Predictions:", predictions);
    println!("{} {}", "Actual:", y_test);
    println!("{}","Accuracy: {:.2f}%".format(accuracy));
    Ok(())
}
