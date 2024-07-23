from typing import List, Tuple

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

def linear_regression(X: List[List[float]], y: List[float], learning_rate: float = 0.01, epochs: int = 1000) -> Tuple[List[float], float]:
    n_samples, n_features = len(X), len(X[0])
    weights, bias = initialize_parameters(n_features)

    for _ in range(epochs):
        y_pred = predict(X, weights, bias)
        dw, db = compute_gradients(X, y, y_pred, n_samples)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

    return weights, bias

if __name__ == "__main__":
    X: List[List[ float ]] = [[1] , [2] , [3] , [4] , [5]]
    y: List[ float ] = [1, 2, 3, 4, 5]

    learning_rate: float = 0.01
    epochs: int = 1000
    weights, bias = linear_regression(X, y, learning_rate, epochs)

    predictions = predict(X, weights, bias)

    print("Predictions:", predictions)