import random
import numpy as np

# I'd really love for this to be deterministic, but it won't be, and really, that's okay
def training_generator(X, y, U, labeled_to_total_ratio, batch_size=10):
    # x_index = u_index = 0
    x_size = int(round(batch_size*(labeled_to_total_ratio)))
    u_size = int(round(batch_size*(1-labeled_to_total_ratio)))
    # x_end = X.shape[0] - 1
    # u_end = U.shape[0] - 1

    print('creating generator..', 'batch', batch_size, 'ratio', labeled_to_total_ratio, 'sizes', x_size, u_size)
    while True:
        # Randomly select a set of example indices
        Xi = np.random.randint(0, X.shape[0], x_size)
        Ui = np.random.randint(0, U.shape[0], u_size)

        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield(Xi, Ui)
        