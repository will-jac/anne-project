
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

def label_unlabel_split(X, y, num_lab, num_classes=10, one_hot = True, shuffle=True):
    # split, ensuring that the ratio of classes is the same
    X_by_class = []
    y_by_class = []
    for c in range(num_classes):
        if one_hot:
            one_hot_c = np.zeros(num_classes)
            one_hot_c[c] = 1
            ind = y == one_hot_c
            ind = np.all(ind, axis=1) 
        else:
            ind = y == c
        X_by_class.append(X[ind])
        y_by_class.append(y[ind])

    X = np.concatenate([x_c[0:(num_lab // num_classes)] for x_c in X_by_class])
    y = np.concatenate([y_c[0:(num_lab // num_classes)] for y_c in y_by_class])
    U = np.concatenate([x_c[(num_lab // num_classes) :] for x_c in X_by_class])
    
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        X, y = X[permutation], y[permutation]

        np.random.shuffle(U)

    return Data(X, y, U)

def train_test_valid_split(X, y, num_classes=10, split=(0.8, 0.1, 0.1), one_hot=True, shuffle=True, U=None):
    assert sum(split) == 1
    assert X.shape[0] == y.shape[0]

    n = X.shape[0]
    start = 0
    stop = 0
    # Data object used for implicit type checking / inferrence
    split_data = [None]*len(split)

    # split, ensuring that the ratio of classes is the same
    X_by_class = []
    y_by_class = []
    for c in range(num_classes):
        if one_hot:
            one_hot_c = np.zeros(num_classes)
            one_hot_c[c] = 1
            ind = y == one_hot_c 
            ind = np.all(ind, axis=1) 
        else:
            ind = y == c
        X_by_class.append(X[ind])
        y_by_class.append(y[ind])
        
    # for train, test, valid
    for i in range(len(split)):
        stop  = int(n // num_classes * sum(split[0:i+1]))

        out_X = np.concatenate([x_c[start:stop] for x_c in X_by_class])
        out_y = np.concatenate([y_c[start:stop] for y_c in y_by_class])

        if shuffle:
            # Shuffle the arrays by giving the permutation in the square brackets.
            permutation = np.random.permutation(X.shape[0])
            X, y = X[permutation], y[permutation]

        # put all the unlabeled data in the first element
        if i == 0:
            split_data[i] = Data(out_X, out_y, U)
        else:
            split_data[i] = Data(out_X, out_y, None)
        start = stop
    return tuple(split_data)

def shuffle(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    return a[permutation], b[permutation]