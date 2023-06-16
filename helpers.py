import numpy as np


# Mean Squared Error loss
def mse_loss(y, tx, w):
    e = y - tx.dot(w)
    mse = np.dot(np.transpose(e), e) / (2.0 * len(e))
    return np.squeeze(mse)


# Logistic loss
def logistic_loss(y, tx, w):  # Compute the loss of the log-likelihood cost function.
    t = np.log(1.0 + np.exp(tx @ w))
    return (1.0 / len(y)) * np.squeeze(np.sum(t - (y * (np.dot(tx, w)))))


# Sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Standardization function
def standardize(x):
    return np.transpose(
        np.array(
            [(x[:, i] - np.mean(x[:, i])) / np.std(x[:, i]) for i in range(x.shape[1])]
        )
    )


# Computes accuracy
def accuracy_score(y, tx, w):
    y_pred = np.array([tx @ w > 0.5]).reshape(len(y))
    return 1 / len(y) * np.sum(y == y_pred)


# Random batch generator
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validate_function(func, Y, X, k, k_indices, lambdas):
    """
    Args:
        func:       validation function
        Y:          shape=(N,)
        X:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambdas:    scalar, cf. ridge_regression()
    Returns:
        weight,train and test accuracies
    """

    train_acc = []
    test_acc = []
    weights = []

    for lambda_ in lambdas:  # ranging over lambdas
        train_acc_l = []
        test_acc_l = []
        w_l = []

        for i in range(k):  # ranging over partitions
            test_idx = k_indices[i]
            train_idx = k_indices[~(np.arange(k_indices.shape[0]) == i)].reshape(-1)

            ws, loss_tr = func(Y[train_idx], X[train_idx], lambda_)
            train_acc_l.append(accuracy_score(Y[train_idx], X[train_idx], ws))
            test_acc_l.append(accuracy_score(Y[test_idx], X[test_idx], ws))
            w_l.append(ws)

        train_acc.append(np.mean(train_acc_l))
        test_acc.append(np.mean(test_acc_l))
        weights.append(np.mean(w_l, axis=0))

    idx = np.argmax(test_acc)
    return test_acc[idx], train_acc[idx], lambdas[idx], weights[idx]


def cross_validation(func, Y, X, k, degree, seed, lambdas):
    """
    Args:
        func:       validation function
        Y:          shape=(N,)
        X:          shape=(N,)
        k: number of partitions
        seed : random seed
        degree: integer, degree of the polynomial expansion
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    """

    k_indices = build_k_indices(Y, k, seed)
    test_acc = []
    ls = []
    weights = []
    degrees = []

    for d in range(1, degree):
        # Feat expansion
        X_expanded = polynomial_expansion(X, d)

        # Cross validate
        test, train, l, w = cross_validate_function(
            func, Y, X_expanded, k, k_indices, lambdas
        )
        test_acc.append(test)
        ls.append(l)
        weights.append(w)
        degrees.append(d)

    idx = np.argmax(test_acc)
    print(
        "For polynomial expansion up to degree %.f, best degree : %d, best lambda : %.20f, accuracy : %5f"
        % (degree - 1, degrees[idx], ls[idx], float(test_acc[idx]))
    )
    return degrees[idx], test_acc[idx], ls[idx], weights[idx]


def polynomial_expansion(x, degree):
    poly = x
    for deg in range(2, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    one = np.ones([poly.shape[0], 1])
    return np.c_[one, poly]
