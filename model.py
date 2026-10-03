"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):

    x = x.copy()

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = x - x_mean

    x_std[x_std == 0] = 1

    x = x / x_std

    return x

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w = np.zeros(n_features)
    b = 0.0

    return {
        'w': w,
        'b': b
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.

    w = params['w']

    b = params['b']
    
    

    scores = x @ w + b

    return scores

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.

    predictions = np.where(scores >= 0, 1 , -1)

    return predictions

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    loss = max(0, 1-y * score)

    return loss

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)

    w = params['w']

    scores = compute_scores(x, params)

    hinge = np.maximum(0,1 - y * scores)

    mean_hinge = np.mean(hinge)

    regularization = reg_lambda * np.dot( w , w)

    return mean_hinge + regularization

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""

    w = params['w']
    b = params['b']

    scores = x @ w + b

    margin_violations = y * scores < 1

    dw = np.zeros_like(w, dtype=float)
    db = 0.0

    if np.any(margin_violations):
        dw = np.sum(
            -y[margin_violations, None] * x[margin_violations],
            axis=0
        ) / len(y)

        db = np.sum(
            -y[margin_violations]
        ) / len(y)

    dw = dw + 2 * reg_lambda * w

    return {
        'dw': dw,
        'db': db
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params['w']
    b = params['b']

    new_w = w - learning_rate * grads['dw']
    new_b = b - learning_rate * grads['db']

    return {
        'w' : new_w,
        'b' : new_b
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):

    params = initialize_parameters(x.shape[1])

    for _ in range(n_epochs):

        grads = compute_gradients(
            x,
            y,
            params,
            reg_lambda
        )

        params = apply_update(
            params,
            grads,
            learning_rate
        )

    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].

    scores = compute_scores(x, params)

    predict = predict_from_scores(scores)

    return predict

# Step 11 - accuracy_score
import numpy as np

def accuracy_score(y_pred, y_true):
    # TODO: return the fraction of positions where y_pred equals y_true.

    scores = y_pred == y_true

    accuracy = np.mean(scores)
    



    return accuracy

