import tensorflow as tf 
from tensorflow.keras.layers import InputLayer, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

def build_nn(n_input, n_hidden_list, n_output,
        lrate, activation, 
        out_activation=None,  metrics=None, loss='mse',
        dropout=None, L2_reg=None):

    if out_activation is None:
        out_activation = activation
    
    model = Sequential()
    model.add(InputLayer(input_shape=(n_input,)))

    if dropout is not None:
        model.add(Dropout(dropout))

    # allow L2 regularization
    if L2_reg is not None:
        layer = lambda shape, name, activation : Dense(shape, name='layer_' + name,
            use_bias=True, activation=activation,
            kernel_regularizer=L2(L2_reg))
    else:
        layer = lambda shape, name, activation : Dense(shape, name='layer_' + name,
            use_bias=True, activation=activation)

    for i, shape in enumerate(n_hidden_list):

        model.add(layer(shape, str(i), activation))

        if dropout is not None:
            model.add(Dropout(dropout))
    
    model.add(layer(n_output, "output", out_activation))

    opt = Adam(lr=lrate)

    model.compile(loss=loss, metrics=metrics, optimizer=opt)
    return model

