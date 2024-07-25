import numpy as np
from collections import Counter

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
    file_path = 'KNN.csv'
    X, y = load_data(file_path)
    X_train, X_test = X[:-50], X[-50:]
    y_train, y_test = y[:-50], y[-50:]

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = calculate_accuracy(predictions, y_test)

    print("Predictions:", predictions)
    print("Actual:", y_test)
    print("Accuracy: {:.2f}%".format(accuracy))
