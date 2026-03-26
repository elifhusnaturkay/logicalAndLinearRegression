# Linear and Logistic Regression from Scratch

A hands-on implementation of linear and logistic regression built step by step, focusing on understanding the underlying mathematics rather than using high-level libraries.

## Purpose

This project is a learning exercise. The code is written incrementally to understand each concept before moving to the next. Comments and structure reflect the learning process.

## Project Structure

```
.
├── main.py                  # Entry point
├── linear_regression.py     # Linear regression model
├── logistic_regression.py   # Logistic regression model
├── loss.py                  # MSE and Binary Cross-Entropy loss functions
├── plotter.py               # Visualization with Matplotlib
├── data.py                  # Synthetic data generation
└── requirements.txt         # Dependencies
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

> Note: Run with the venv Python, not the system Python. Otherwise matplotlib will not be found.

## Dependencies

- numpy
- matplotlib

## Learning Notes

### Linear Regression

The model learns the equation `y = w*x + b` from data using gradient descent.

**Loss function — MSE (Mean Squared Error)**
- Measures how far the predictions are from the real values.
- Each error is squared to eliminate negatives and penalize large mistakes more heavily.
- Formula: `MSE = (1/n) * sum((y - y_hat)^2)`

**Gradient Descent**
- We minimize MSE by iteratively updating `w` and `b`.
- At each step, we compute the gradient (derivative of MSE with respect to w and b).
- The gradient tells us which direction increases the loss — so we move in the opposite direction.
- Update rule: `w = w - learning_rate * gradient`
- `learning_rate` controls step size. Too large: overshoots. Too small: converges slowly.

**Forward Pass**
- Given an input `x`, the model produces a prediction: `y_hat = w*x + b`
- `w` and `b` start at 0 and are updated each iteration.

**Result**
- Synthetic data generated with true equation `y = 2x + 1` + noise.
- After 1000 iterations the model recovers approximately `w ≈ 2.0`, `b ≈ 1.0`.

### Logistic Regression

The model learns to classify data into two classes (0 or 1) using gradient descent.

**Sigmoid Function**
- Linear regression outputs any number. We need a probability (0 to 1).
- Sigmoid maps any value to (0, 1): `sigmoid(z) = 1 / (1 + e^(-z))`
- Very large z → output near 1. Very small z → output near 0. z=0 → output = 0.5.

**Forward Pass**
- Same as linear but with sigmoid applied: `y_hat = sigmoid(w*x + b)`
- Output is interpreted as P(y=1).

**Loss function — Binary Cross-Entropy (BCE)**
- MSE is not suitable for probabilities — we use BCE instead.
- Formula: `loss = -mean( y*log(y_hat) + (1-y)*log(1-y_hat) )`
- When y=1: we want log(y_hat) to be large → y_hat close to 1.
- When y=0: we want log(1-y_hat) to be large → y_hat close to 0.

**Gradient Descent**
- Same update rule as linear regression: `w = w - lr * gradient`
- Thanks to sigmoid + BCE combination, gradients have the same clean form as in linear regression.

**Decision Boundary**
- If y_hat >= 0.5 → predict class 1, otherwise class 0.
