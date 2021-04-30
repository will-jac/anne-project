import random
import numpy as np

# I'd really love for this to be deterministic, but it won't be, and really, that's okay
def training_generator(data, batch_size=10):
    # x_index = u_index = 0
    labeled_to_total_ratio = data.X.shape[0] / (data.X.shape[0] + data.U.shape[0])

    x_size = int(round(batch_size*(labeled_to_total_ratio)))
    u_size = int(round(batch_size*(1-labeled_to_total_ratio)))
    # x_end = X.shape[0] - 1
    # u_end = U.shape[0] - 1

    print('creating generator..', 'batch', batch_size, 'ratio', labeled_to_total_ratio, 'sizes', x_size, u_size)
    while True:
        # Randomly select a set of example indices
        Xi = np.random.randint(0, data.X.shape[0], x_size)
        Ui = np.random.randint(0, data.U.shape[0], u_size)

        # The generator will produce a pair of return values: one for labeled indexes and one for unlabeled indexes
        yield (Xi, Ui)
        
def pseudo_label_generator(data, batch_size=10, unlabeled=-1):
    if data.y.shape[1] is None:
        label_shape = tuple(data.U.shape[0])
    else:
        label_shape = (data.U.shape[0], data.y.shape[1])
    pseudo_labels = unlabeled * np.ones(label_shape, dtype=float)
        
    X = np.append(data.X, data.U, axis=0)
    y = np.append(data.y, pseudo_labels, axis=0)

    print('creating generator..', 'batch', batch_size, 'ratio', labeled_to_total_ratio, 'sizes', x_size, u_size)
    while True:
        # Randomly select a set of example indices
        Xi = np.random.randint(0, X.shape[0], batch_size)

        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield (X[Xi], y[Xi])