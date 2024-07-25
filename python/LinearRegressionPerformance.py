from typing import List, Tuple
import time
import csv
from memory_profiler import memory_usage

def initialize_parameters(n_features: int) -> Tuple[List[float], float]:
    weights = [0.0] * n_features
    bias = 0.0
    return weights, bias

def predict(X: List[List[float]], weights: List[float], bias: float) -> List[float]:
    predictions = []
    for i in range(len(X)):
        prediction = sum(X[i][j] * weights[j] for j in range(len(X[0]))) + bias
        predictions.append(prediction)
    return predictions

def compute_gradients(X: List[List[float]], y: List[float], y_pred: List[float], n_samples: int) -> Tuple[List[float], float]:
    n_features = len(X[0])
    dw = [0.0] * n_features
    db = 0.0

    for i in range(n_samples):
        error = y_pred[i] - y[i]
        for j in range(n_features):
            dw[j] += (2 / n_samples) * X[i][j] * error
        db += (2 / n_samples) * error

    return dw, db

def update_parameters(weights: List[float], bias: float, dw: List[float], db: float, learning_rate: float) -> Tuple[List[float], float]:
    for j in range(len(weights)):
        weights[j] -= learning_rate * dw[j]
    bias -= learning_rate * db
    return weights, bias

def linear_regression(X: List[List[float]], y: List[float], learning_rate: float = 0.01, epochs: int = 1000) -> Tuple[List[float], float, List[float]]:
    n_samples, n_features = len(X), len(X[0])
    weights, bias = initialize_parameters(n_features)

    epoch_times = []

    for _ in range(epochs):
        epoch_start_time = time.time()
        
        y_pred = predict(X, weights, bias)
        dw, db = compute_gradients(X, y, y_pred, n_samples)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

    return weights, bias, epoch_times

def write_times_to_csv(epoch_times: List[float], filename: str) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Time (seconds)"])
        for i, epoch_time in enumerate(epoch_times):
            writer.writerow([i + 1, epoch_time])

if __name__ == "__main__":
    num_samples = 10000 
    num_features = 10  
    mem_usage_before = memory_usage()[0]

    X: List[List[float]] = [[1.0 for _ in range(num_features)] for _ in range(num_samples)]
    y: List[float] = [1.0 for _ in range(num_samples)]

    start_time = time.time()

    learning_rate: float = 0.01
    epochs: int = 1000
    weights, bias, epoch_times = linear_regression(X, y, learning_rate, epochs)

    predictions = predict(X, weights, bias)

    mem_usage_after = memory_usage()[0]

    end_time = time.time()
    execution_time = end_time - start_time

    print("Predictions:", predictions)
    print(f"Execution Time: {execution_time:.6f} seconds")
    print(f"Memory Usage: {mem_usage_after - mem_usage_before:.2f} MB")

    # Write epoch times to CSV
    write_times_to_csv(epoch_times, 'epoch_times.csv')
