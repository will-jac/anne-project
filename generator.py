import random
import numpy as np

# I'd really love for this to be deterministic, but it won't be, and really, that's okay
def training_generator(data, batch_size=10, min_labeled_per_epoch=None):
    # x_index = u_index = 0
    if min_labeled_per_epoch is None:
        labeled_to_total_ratio = data.X.shape[0] / (data.X.shape[0] + data.U.shape[0])
    
        x_size = int(round(batch_size*(labeled_to_total_ratio)))
        u_size = int(round(batch_size*(1-labeled_to_total_ratio)))
    else:
        x_size = min_labeled_per_epoch
        u_size = batch_size - min_labeled_per_epoch

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
    # prelabel the psuedo labels
    pseudo_labels = unlabeled * np.ones(label_shape, dtype=float)

    print('creating generator..', 'batch', batch_size)
    while True:
        # Randomly select a set of example indices
        Xi = np.random.randint(0, X.shape[0], batch_size)

        # The generator will produce a pair of return values: one for inputs and one for outputs
        # print(Xi)
        yield (X[Xi], y[Xi])