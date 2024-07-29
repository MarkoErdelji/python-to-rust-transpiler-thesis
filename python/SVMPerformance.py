import numpy as np
import time
from memory_profiler import memory_usage
import csv

def load_data(file_path):
    features = []
    targets = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            row_data = list(map(float, line.strip().split(',')))
            feature_values = row_data[:-1]
            target_value = row_data[-1]
            features.append(feature_values)
            targets.append(target_value)
    return np.array(features), np.array(targets)


def svm_soft_margin_train(X, y, alpha=0.001, lambda_=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    epoch_times = []

    for iteration in range(n_iterations):
        epoch_start_time = time.time()

        for i, Xi in enumerate(X):
            if y[i] * (np.dot(Xi, w) - b) >= 1:
                w -= alpha * (2 * lambda_ * w)
            else:
                w -= alpha * (2 * lambda_ * w - np.dot(Xi, y[i]))
                b -= alpha * y[i]

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

    return w, b, epoch_times


def svm_soft_margin_predict(X, w, b):
    pred = np.dot(X, w) - b
    result = [1 if val > 0 else -1 for val in pred]
    return result


def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def write_times_to_csv(epoch_times: list, filename: str) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Time (seconds)"])
        for i, time in enumerate(epoch_times):
            writer.writerow([i + 1, time])


if __name__ == "__main__":
    start_time = time.time()
    mem_usage_before = memory_usage()[0]

    file_path = 'extendedKNN.csv'
    X, y = load_data(file_path)

    y = np.where(y == 0, -1, 1)

    w, b, epoch_times = svm_soft_margin_train(X, y)
    predictions = svm_soft_margin_predict(X, w, b)
    acc = accuracy(y, predictions)

    print("Predictions:", predictions)
    print("Accuracy:", acc)

    mem_usage_after = memory_usage()[0]

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    print(f"Memory Usage: {mem_usage_after - mem_usage_before:.2f} MB")

    write_times_to_csv(epoch_times, 'svm_epoch_times.csv')
