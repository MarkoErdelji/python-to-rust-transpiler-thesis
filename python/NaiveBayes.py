import numpy as np

def fit_naive_bayes(X, y):
    class_counts = np.array([np.sum(y == 0), np.sum(y == 1)])
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]
    class_summaries = {
        0: (np.mean(X_class_0, axis=0), np.var(X_class_0, axis=0)),
        1: (np.mean(X_class_1, axis=0), np.var(X_class_1, axis=0))
    }
    return class_summaries, class_counts

def calculate_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def calculate_class_probabilities(x, class_summaries, class_counts):
    total_count = np.sum(class_counts)
    prob_class_0 = class_counts[0] / total_count
    prob_class_1 = class_counts[1] / total_count
    mean_0, var_0 = class_summaries[0]
    mean_1, var_1 = class_summaries[1]
    probs_0 = calculate_probability(x, mean_0, var_0)
    probs_1 = calculate_probability(x, mean_1, var_1)
    prob_0 = prob_class_0 * np.prod(probs_0)
    prob_1 = prob_class_1 * np.prod(probs_1)
    return np.array([prob_0, prob_1])

def predict_naive_bayes(X, class_summaries, class_counts):
    predictions = []
    for x in X:
        probabilities = calculate_class_probabilities(x, class_summaries, class_counts)
        predictions.append(np.argmax(probabilities))
    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

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

if __name__ == "__main__":
    file_path = 'KNN.csv'
    X, y = load_data(file_path)
    class_summaries, class_counts = fit_naive_bayes(X, y)
    predictions = predict_naive_bayes(X, class_summaries, class_counts)
    acc = accuracy(y, predictions)
    print("Predictions:", predictions)
    print("Accuracy:", acc)
