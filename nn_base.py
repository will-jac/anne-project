import tensorflow as tf 
from tensorflow.keras.layers import InputLayer, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

def add_layer(model, shape, name, activation, use_bias=True, L2_reg=None, dropout=None, layer=Dense):
    if L2_reg is not None:
        model.add(layer(shape, name=name, use_bias=use_bias, kernel_regularizer=L2(L2_reg)))
    else:
        model.add(layer(shape, name=name, use_bias=use_bias))
    if dropout is not None:
        model.add(Dropout(dropout))

def build_nn(input_shape, hidden_shape_list, n_output,
        lrate, activation, 
        out_activation=None,  metrics=None, loss='mse',
        dropout=None, L2_reg=None, return_params=False):

    if out_activation is None:
        out_activation = activation
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    for i, shape in enumerate(hidden_shape_list):

        add_layer(model, shape, 'hidden' + str(i), activation, L2_reg=L2_reg, dropout=dropout)
    
    add_layer(model, n_output, "output", out_activation)
    
    opt = Adam(lr=lrate)

    model.compile(loss=loss, metrics=metrics, optimizer=opt)
    # additional returns are for use in modifying and recompiling
    if return_params:
        return model, loss, opt, metrics
    else:
        return model

def pretrain_dae_nn(train_data, validation_data, hidden_shape_list, n_output,
        lrate, activation, epochs, callbacks,
        out_activation=None, metrics=None, loss='mse',
        dropout=None, L2_reg=None, verbose=False, return_params=False):

    if out_activation is None:
        out_activation = activation

    # start by building a nn with the first hidden layer
    # This probably won't work for 2-D inputs without some modification
    print('constructing initial model:', train_data.X.shape[1], hidden_shape_list[0], train_data.X.shape[1])
    base_model, loss, opt, metrics = build_nn(train_data.X.shape[1], [hidden_shape_list[0]], train_data.X.shape[1],
            lrate=lrate, activation=activation, out_activation=out_activation,
            metrics=metrics, loss=loss, dropout=dropout, L2_reg=L2_reg, return_params=True)

    if verbose:
        base_model.summary()
        print('training on :')
        print(train_data.X.shape, train_data.X)
        print(train_data.y.shape, train_data.y)

    # now, fit to train data
    base_model.fit(train_data.X, train_data.y, validation_data=validation_data,
            epochs=epochs, callbacks=callbacks)

    # once we're fit, train the next layer
    for i, h in enumerate(hidden_shape_list[1:]):
        # freeze the hidden layer
        base_model.trainable = False
        # remove the output layer
        # normally, could just call model.layers.pop() , but this isn't working
        # construct a new model with just the layers we need
        model = Sequential()
        model.add(InputLayer(input_shape=train_data.X.shape[1]))
        for layer in base_model.layers[:-1]: # up to, but not including, the last layer
            model.add(layer)

        # add a new hidden layer
        add_layer(model, h, 'hidden' + str(i+1), activation=activation, L2_reg=L2_reg)
        # add the output layer back
        add_layer(model, train_data.X.shape[1], 'output' + str(i+1), activation=out_activation, L2_reg=L2_reg)
        # recompile the model
        model.compile(loss=loss, metrics=metrics, optimizer=opt)
        if verbose:
            model.summary()
        # train again
        model.fit(train_data.X, train_data.y, validation_data=validation_data,
                epochs=epochs, callbacks=callbacks)

        # reset for next iteration
        base_model = model

    # now, add the real output
    # remove the output layer
    model = Sequential()
    model.add(InputLayer(input_shape=train_data.X.shape[1]))
    # TODO: dropout for the input layer ? Probably not

    for layer in base_model.layers[:-1]: # up to, but not including, the last layer
        model.add(layer)
        # add dropout between each layers
        if dropout is not None:
            model.add(Dropout(dropout))

    add_layer(model, n_output, 'output', activation=out_activation, L2_reg=L2_reg)

    # unfreeze everything
    model.trainable = True    

    model.compile(loss=loss, metrics=metrics, optimizer=opt)
    
    # additional returns are for use in modifying and recompiling
    if return_params:
        return model, loss, opt, metrics
    else:
        return model


        