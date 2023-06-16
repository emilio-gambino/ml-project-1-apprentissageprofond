import numpy as np

from helpers import *


# MSE Gradient Descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: labels, shape=(N, )
        tx: data, shape=(N,D)
        initial_w: initial weights, shape=(D,)
        max_iters: number of iterations
        gamma: learning rate
    """

    w = initial_w
    for n_iter in range(max_iters):
        grad = -(1.0 / len(y)) * np.dot(np.transpose(tx), (y - np.dot(tx, w)))
        w = w - gamma * grad
    loss = mse_loss(y, tx, w)
    return w, loss


# MSE Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: labels, shape=(N, )
        tx: data, shape=(N,D)
        initial_w: initial weights, shape=(D,)
        max_iters: number of iterations
        gamma: learning rate
    """

    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1):
            grad = -(1.0 / N) * np.dot(
                np.transpose(batch_tx), (batch_y - np.dot(batch_tx, w))
            )
            w = w - gamma * grad
    loss = mse_loss(y, tx, w)
    return w, loss


# Least Squares
def least_squares(y, tx):
    """
    Args:
        y: labels, shape=(N, )
        tx: data, shape=(N,D)
    """

    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)), np.transpose(tx)), y)
    loss = mse_loss(y, tx, w)
    return w, loss


# Ridge Regression
def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: labels, shape=(N, )
        tx: data, shape=(N,D)
        lambda_: regularizer
    """

    N, D = tx.shape
    w = (
        np.linalg.inv((tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D))).dot(tx.T).dot(y)
    )
    loss = mse_loss(y, tx, w)
    return w, loss


# Logistic Regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2,). The vector of model parameters.
        max_iters: number of iterations
        gamma: learning rate
    """

    w = initial_w
    for i in range(max_iters):
        grad = (1.0 / len(y)) * np.dot(np.transpose(tx), (sigmoid(np.dot(tx, w)) - y))
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w, loss


# Regularized Logistic Regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2,). The vector of model parameters.
        lambda_: regularizer
        max_iters: number of iterations
        gamma: learning rate
    """
    w = initial_w
    for i in range(max_iters):
        grad = (1.0 / len(y)) * tx.T @ (sigmoid(tx @ w) - y) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w, loss
