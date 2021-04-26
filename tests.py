import numpy as np

import util

# data sets
import data.adult as adult
import data.cifar_10 as cifar_10

# other things
from sklearn.metrics import roc_curve, plot_roc_curve

# All test functions need to 
# Import the data and set-up train, test, validation sets.
# Call model.fit on the training and validation test.
# Return the accuracy based on the test set. 
# test-specific models

from base_models import Scifar10Model

def adult_test(model, u=0.8):

    # turn the labels into a proper matrix
    adult.y = np.reshape(adult.y, (adult.y.shape[0], 1))    

    (data, _) = util.train_test_valid_split(adult.X, adult.y, split = (0.1, 0.9))
    # data = adult

    (label, unlabeled) = util.train_test_valid_split(data.X, data.y, split=(u, 1 - u))

    (train, test, valid) = util.train_test_valid_split(label.X, label.y, split=(0.6, 0.2, 0.2), U = unlabeled.X)

    # for model, name in zip(models, model_names):
    model.fit(train, valid)

    y_pred = model.predict(test.X)
    y_pred = y_pred.ravel()

    # classify
    for i, y_p in enumerate(y_pred):
        if y_p > 0.0: # TODO: should be 0?
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    acc = 1.0 - wrong
    print(model.name, ' : acc:', acc)

    return acc

def cifar_10_test (model, u=0.8):
    # Load in training and test data

    # One-hot encode cifar_10.y_train and cifar_10.y_test

    # Prepare cifar_10.X_train and cifar_10.X_test
    

# return test function (which should accept a model and return the accuracy) and the input / output dimmensions
def scifar10_test(model):
    ...
    # TODO
# return test-specific model and the testing function (which should accept a model and return the accuracy)
tests = {
    'scifar10' : (Scifar10Model, scifar10_test)
}