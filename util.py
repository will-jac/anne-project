
from collections import namedtuple

Data = namedtuple('Data', 'X y U')

## see pseudo-labels for an example of usage

## Things for data generation

import numpy as np

def mse(predict, true, axis=0):
    t = 0
    for i in range(len(predict)):
        t += (predict[i] - true[i])**2
    return np.sqrt(t / len(predict))

# for classification
def percent_wrong(predict, true):
    n = len(predict)
    wrong = 0
    for i in range(n):
        if predict[i] != true[i]:
            wrong += 1
    return 1.0 * wrong / n

def label_unlabel_split(X, y, num_lab, shuffle=True, num_classes=10, one_hot = True):
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        X, y = X[permutation], y[permutation]
    # split, ensuring that the ratio of classes is the same
    out_X = []
    out_y = []
    out_U = []
    for c in range(num_classes):
        if one_hot:
            one_hot_c = np.zeros(num_classes)
            one_hot_c[c] = 1
            ind = y == one_hot_c 
        else:
            ind = y == c
        out_X.append(X[ind][num_lab:])
        out_y.append(y[ind][num_lab:])
        out_U.append(U[ind][0:num_lab])

    return Data(out_X, out_y, out_U)

def train_test_valid_split(X, y, split=(0.8, 0.1, 0.1), shuffle=True, U=None):
    assert sum(split) == 1
    assert X.shape[0] == y.shape[0]

    # first, shuffle the data
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        X, y = X[permutation], y[permutation]
        if U is not None:
            np.random.shuffle(U)

    # train will have all the unlabeled data!
    n = X.shape[0]
    start = 0
    stop = 0
    # Data object used for implicit type checking / inferrence
    split_data = [None]*len(split)

    # split, ensuring that the ratio of classes is the same
    out_X = []
    out_y = []
    for c in range(num_classes):
        if one_hot:
            one_hot_c = np.zeros(num_classes)
            one_hot_c[c] = 1
            ind = y == one_hot_c 
        else:
            ind = y == c
        out_X.append(X[ind])
        out_y.append(y[ind])
        
    # for train, test, valid
    for i in range(len(split)):
        stop  = int(n * sum(split[0:i+1]))

        if i == 0 and (U is not None):
            # put all the unlabeled data in the first element
            split_data[i] = Data(out_X[start:stop,:], out_y[start:stop], U)
        else:
            split_data[i] = Data(out_X[start:stop,:], out_y[start:stop], None)
        start = stop
    return tuple(split_data)

def shuffle(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    return a[permutation], b[permutation]