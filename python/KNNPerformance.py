import numpy as np
from collections import Counter
import time
from memory_profiler import memory_usage


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        label_counts = Counter(k_nearest_labels)
        return label_counts.most_common(1)[0][0]

    def _compute_distances(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))


def load_data(file_path):
    features, targets = [], []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            row_data = list(map(float, line.strip().split(',')))
            feature_values = row_data[:-1]
            target_value = row_data[-1]
            features.append(feature_values)
            targets.append(target_value)
    return np.array(features), np.array(targets)


def calculate_accuracy(predictions, actual):
    return np.mean(predictions == actual) * 100


if __name__ == "__main__":
    file_path = 'extendedKNN.csv'

    start_time = time.time()
    mem_usage_before = memory_usage()[0]

    X, y = load_data(file_path)
    X_train, X_test = X[:-50], X[-50:]
    y_train, y_test = y[:-50], y[-50:]

    knn = KNN(k=3)

    fit_start_time = time.time()
    knn.fit(X_train, y_train)
    fit_end_time = time.time()
    fit_duration = fit_end_time - fit_start_time
    mem_usage_after_fit = memory_usage()[0]

    predict_start_time = time.time()
    predictions = knn.predict(X_test)
    predict_end_time = time.time()
    predict_duration = predict_end_time - predict_start_time
    mem_usage_after_predict = memory_usage()[0]

    accuracy = calculate_accuracy(predictions, y_test)

    end_time = time.time()
    mem_usage_after = memory_usage()[0]

    execution_time = end_time - start_time
    total_memory_usage = mem_usage_after - mem_usage_before
    fit_memory_usage = mem_usage_after_fit - mem_usage_before
    predict_memory_usage = mem_usage_after_predict - mem_usage_after_fit

    print("Predictions:", predictions)
    print("Actual:", y_test)
    print("Accuracy: {:.2f}%".format(accuracy))
    print(f"Total Execution Time: {execution_time:.6f} seconds")
    print(f"Total Memory Usage: {total_memory_usage:.2f} MB")
    print(f"Fit Time: {fit_duration:.6f} seconds")
    print(f"Predict Time: {predict_duration:.6f} seconds")
    print(f"Fit Memory Usage: {fit_memory_usage:.2f} MB")
    print(f"Predict Memory Usage: {predict_memory_usage:.2f} MB")
