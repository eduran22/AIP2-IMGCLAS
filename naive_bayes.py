# naive_bayes.py
# Implements a Bernoulli Naive Bayes classifier for image classification.

import math
from typing import List


class NaiveBayesClassifier:
    """
    Bernoulli Naive Bayes classifier.
    - Designed for binary feature vectors (0/1), but works with small integers.
    - Supports multi-class classification (digits: 10 classes, faces: 2 classes).
    """

    def __init__(self):
        self.num_classes = None
        self.num_features = None

        # P(class)
        self.class_priors = []

        # P(feature_i = 1 | class = c)
        # Shape: [num_classes][num_features]
        self.feature_probs = []

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------
    def fit(self, X: List[List[int]], y: List[int]):
        """
        Fit the Naive Bayes classifier.

        X : list of feature vectors (each a list of ints: 0/1 or small int)
        y : list of class labels (e.g., 0–9 for digits, 0/1 for faces)
        """
        n_samples = len(X)
        self.num_features = len(X[0])
        self.num_classes = len(set(y))

        # Count examples per class
        class_counts = [0] * self.num_classes

        # Count how often each feature is "on" for each class
        feature_counts = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]

        # -------------------------------
        # PASS 1: Count occurrences
        # -------------------------------
        for feats, label in zip(X, y):
            class_counts[label] += 1

            for i, value in enumerate(feats):
                if value > 0:  # treat any positive number as "on"
                    feature_counts[label][i] += 1

        # -------------------------------
        # Compute priors: P(class = c)
        # -------------------------------
        self.class_priors = [
            class_counts[c] / n_samples
            for c in range(self.num_classes)
        ]

        # -------------------------------
        # Compute conditional probabilities:
        # P(feature_i = 1 | class = c)
        #
        # Using Laplace smoothing:
        #   (count + 1) / (class_count + 2)
        # -------------------------------
        self.feature_probs = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]

        for c in range(self.num_classes):
            for i in range(self.num_features):
                count_on = feature_counts[c][i]
                total = class_counts[c]
                prob = (count_on + 1) / (total + 2)  # Laplace smoothing
                self.feature_probs[c][i] = prob

    # ------------------------------------------------------------
    # PREDICT A SINGLE EXAMPLE
    # ------------------------------------------------------------
    def predict_one(self, feats: List[int]) -> int:
        """
        Predict the class of a single feature vector.
        Uses log-probabilities to avoid underflow.
        """
        best_class = None
        best_score = -math.inf

        for c in range(self.num_classes):
            # Start with log prior
            score = math.log(self.class_priors[c])

            # Add log-conditional probabilities
            for i, value in enumerate(feats):
                p = self.feature_probs[c][i]

                if value > 0:       # feature ON
                    score += math.log(p)
                else:               # feature OFF
                    score += math.log(1 - p)

            # Track best-scoring class
            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    # ------------------------------------------------------------
    # PREDICT MANY SAMPLES
    # ------------------------------------------------------------
    def predict(self, X: List[List[int]]) -> List[int]:
        """
        Predict the class labels for a list of feature vectors.
        """
        return [self.predict_one(feats) for feats in X]
