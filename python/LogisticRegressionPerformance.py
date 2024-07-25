import numpy as np
import time
import csv
from memory_profiler import memory_usage

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    data = np.array(data, dtype=str)
    return data

def convert_to_float(data):
    return data.astype(float)

def feature_scaling(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds

def split_data(data, test_ratio=0.2):
    num_samples = data.shape[0]
    num_test_samples = int(num_samples * test_ratio)
    X_train = data[:-num_test_samples, :-1]
    y_train = data[:-num_test_samples, -1]
    X_test = data[-num_test_samples:, :-1]
    y_test = data[-num_test_samples:, -1]
    return X_train, y_train, X_test, y_test

def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    h = np.clip(h, 1e-10, 1 - 1e-10)
    cost = (-1 / m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
    return cost

def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1 / m) * X.T.dot(h - y)
    return grad

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)
    epoch_times = []
    
    for i in range(num_iters):
        epoch_start_time = time.time()
        
        theta -= alpha * gradient(X, y, theta)
        cost_history[i] = compute_cost(X, y, theta)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
    
    return theta, cost_history, epoch_times

def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5
    
def write_times_and_costs_to_csv(cost_history: np.ndarray, epoch_times: list, filename: str) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Time (seconds)", "Cost"])
        for i in range(len(epoch_times)):
            writer.writerow([i + 1, epoch_times[i], cost_history[i]])

if __name__ == "__main__":
    start_time = time.time()
    mem_usage_before = memory_usage()[0]

    data = load_data('extendedData.csv')
    data = convert_to_float(data)

    X_train, y_train, X_test, y_test = split_data(data)

    X_train = feature_scaling(X_train)
    X_test = feature_scaling(X_test)

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    theta = np.zeros(X_train.shape[1])
    alpha = 0.01
    num_iters = 2000

    theta, cost_history, epoch_times = gradient_descent(X_train, y_train, theta, alpha, num_iters)

    print(f'Final cost: {cost_history[-1]}')
    print(f'Optimal parameters: {theta}')

    train_predictions = predict(X_train, theta)
    test_predictions = predict(X_test, theta)

    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100

    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    mem_usage_after = memory_usage()[0]

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    print(f"Memory Usage: {mem_usage_after - mem_usage_before:.2f} MB")

    write_times_and_costs_to_csv(cost_history, epoch_times, 'logistic_regression_epoch_times_and_costs.csv')
