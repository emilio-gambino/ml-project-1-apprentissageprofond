#!/usr/bin/env python3

import csv

from implementations import *


def load_data():
    # Opening files
    with open("resources/train.csv", newline="") as csvfile:
        train_data = np.array(list(csv.reader(csvfile)))
    with open("resources/test.csv", newline="") as csvfile:
        test_data = np.array(list(csv.reader(csvfile)))
    return train_data, test_data


def data_preprocessing(data):
    # Labels and removing headers
    ids = data[1:, 0]
    labels = data[1:, 1]
    signal = np.array(data[1:, 2:], dtype=float)
    signal = np.c_[signal, ids]
    signal = np.array(signal, dtype=float)
    labels = np.array(
        list(map(lambda x: 0 if x == "s" else 1, labels))
    )

    # Separate data into 4 samples based on PRI_jet_num values (col 22)
    sets = [[] for i in range(4)]
    set_labels = [[] for i in range(4)]
    for i in range(4):
        sets[i] = signal[signal[:, 22] == i]
        set_labels[i] = labels[signal[:, 22] == i]
        l = len(sets[i])

        # Remove features which are entirely -999 or with an unique value in the column
        noisy_feats = []
        for col in range(signal.shape[1]):
            proportion = np.count_nonzero(sets[i][:, col] == -999) / l
            deviation = np.std(sets[i][:, col])
            if proportion == 1 or deviation == 0:
                noisy_feats += [col]
        sets[i] = np.delete(sets[i], noisy_feats, axis=1)

    # Set remaining -999 values to 0
    for i in range(4):
        m = np.median(sets[i][sets[i] != -999])
        sets[i][sets[i] == -999] = m

    id_sets = [x[:, -1] for x in sets]
    for i in range(len(sets)):
        sets[i] = np.delete(sets[i], -1, 1)

    # Standardizing data
    sets = [standardize(s) for s in sets]
    return sets, set_labels, id_sets


def train_model():
    ws = []

    for i in range(4):
        X_expanded = polynomial_expansion(training_sets[i], degrees[i])
        w, _ = ridge_regression(training_labels[i], X_expanded, lambdas[i])
        accuracy = accuracy_score(training_labels[i], X_expanded, w)
        ws.append(w)
        print("For jet value : %d, produced accuracy of : %f" % (i, accuracy))

    return ws


def generate_predictions():
    predictions = np.array([])

    # Predict
    for s, w, deg in zip(test_sets, ws, degrees):
        # Feat expansion for test sets
        s = polynomial_expansion(s, deg)
        pred = s @ w
        predictions = np.concatenate((predictions, pred), axis=0)

    predictions = np.array(list(map(lambda x: 1 if x < 0.5 else -1, predictions)))

    ids = np.array([])
    for i in range(4):
        ids = np.concatenate((ids, test_ids[i]), axis=0)

    pred = np.concatenate((ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    pred = pred[pred[:, 0].argsort()]

    # save output file
    np.savetxt('ridge_regression_opti.csv', pred, delimiter=',', fmt='%d', header='Id,Prediction', comments='')


train_data, test_data = load_data()
print("Finished loading data")

training_sets, training_labels, _ = data_preprocessing(train_data)
test_sets, test_labels, test_ids = data_preprocessing(test_data)
print("Finished preprocessing")

lambdas = [0.008, 0.017, 0.008, 0.008]  # Obtained with cross validation
degrees = [8, 8, 9, 9]

ws = train_model()
print("Finished training")

generate_predictions()
print("Predictions generated!")
