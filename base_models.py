from weight_normalization import MeanOnlyWeightNormalization
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def gaussian_noise(input, std):

    noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
    return input + noise

def image_augmentation(image):
    
    random_shifts = np.random.randint(-2, 2, (image.shape[0], 2))
    random_transformations = tfa.image.translations_to_projective_transforms(random_shifts)
    image = tfa.image.transform(image, random_transformations, 'NEAREST', output_shape=tf.convert_to_tensor(image.shape[1:3], dtype=np.int32)) 

    return image

class Cifar10Model(tf.keras.Model):

    def __init__(self, do_image_augmentation=True):

        super(Cifar10Model, self).__init__() 

        self.out_size = 10

        self.do_image_augmentation = do_image_augmentation
        # create the network with dropout

        ## TODO: check relu usage. I changed to tanh because that's what's more typically use

        normalization = lambda layer : tfa.layers.WeightNormalization(layer, data_init=False)
        # normalization = lambda layer : tf.keras.layers.BatchNormalization(layer)
        
        self._conv1a = normalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv1a_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv1b = normalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv1b_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv1c = normalization( 
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv1c_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")
        self._dropout1 = tf.keras.layers.Dropout(0.5)


        self._conv2a = normalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv2a_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv2b = normalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv2b_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv2c = normalization(
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv2c_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)


        self._conv3a = normalization(
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=[3, 3],
                padding="valid",
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv3a_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv3b = normalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[1, 1],
                padding="valid", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv3b_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._conv3c = normalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[1, 1],
                padding="valid", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                # activation=lrelu
            )
        )
        self._conv3c_out = tf.keras.layers.LeakyReLU(alpha=0.1)
        # TODO: no normalization on the output, I think
        self._dense = (
            tf.keras.layers.Dense(
                units=10, 
                activation='softmax',## TODO: check what this should be - I changed to sigmoid bc that makes more sense with crossentropy
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )

    @tf.function
    def call(self, inputs, training=True):
        
        # add the stochastic augmentation to the input if we're training
        if training and self.do_image_augmentation:
            h = gaussian_noise(inputs, 0.15)
            h = image_augmentation(h)
        else:
            h = inputs
        
        # pass the (augmented) input through the model
        # print('in', h.shape)
        h = self._conv1a(h, training=training)
        h = self._conv1a_out(h)
        # print('1a', h.shape)
        h = self._conv1b(h, training=training)
        h = self._conv1b_out(h)
        # print('1b', h.shape)
        h = self._conv1c(h, training=training)
        h = self._conv1c_out(h)
        # print('1c', h.shape)
        h = self._pool1(h, training=training)
        # print('1p', h.shape)
        h = self._dropout1(h, training=training)
        # print('chunk 1 done', h.shape)
        h = self._conv2a(h, training=training)
        h = self._conv2a_out(h)
        # print('2a', h.shape)
        h = self._conv2b(h, training=training)
        h = self._conv2b_out(h)
        # print('2b', h.shape)
        h = self._conv2c(h, training=training)
        h = self._conv2c_out(h)
        # print('2c', h.shape)
        h = self._pool2(h, training=training)
        # print('2p', h.shape)
        h = self._dropout2(h, training=training)
        # print('chunck 2 done', h.shape)
        h = self._conv3a(h, training=training)
        h = self._conv3a_out(h)
        # print('3a', h.shape)
        h = self._conv3b(h, training=training)
        h = self._conv3b_out(h)
        # print('3b', h.shape)
        h = self._conv3c(h, training=training)
        h = self._conv3c_out(h)
        # print('3c', h.shape)

        # Average Pooling
        h = tf.reduce_mean(h, axis=[1, 2])
        # print('average pooling done', h.shape)
        h = self._dense(h, training=training)
        # print('dense', h.shape)
        return h

    def pretrain(self, train, validation, epochs, lrate=0.001, L2_reg=0.001,
            loss='mse', out_activation='sigmoid', callbacks=None, metrics=None):
        opt = tf.keras.optimizers.Adam(lr=lrate)
        # pretraining will work on each conv2d chunk
        base_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(train.X.shape), 
                self._conv1a, self._conv1a_out, self._conv1b, self._conv1b_out, self._conv1c, self._conv1c_out,
                self._pool1, self._dropout1,
                tf.keras.layers.Dense(train.y.shape[1], activation='softmax', name='output', use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(L2_reg))
            ]
        )
        base_model.compile(loss=loss, metrics=metrics, optimizer=opt)
        base_model.fit(train.X, train.y, validation_data=(validation.X,validation.y),
            epochs=epochs, callbacks=callbacks)

        for layers in [
            [self._conv2a, self._conv2a_out, self._conv2b, self._conv2b_out, self._conv2c, self._conv2c_out, self._pool2, self._dropout2],
            [self._conv3a, self._conv3b, self._conv3c]
        ]:
            base_model.trainable = False

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(train.X.shape))
            for layer in base_model.layers[:-1]:
                model.add(layer)

            # add the next batch of layers
            for layer in layers:
                model.add(layer)

            # add the output back
            model.add(tf.keras.layers.Dense(train.X.shape[1], activation=out_activation, name='output1', use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(L2_reg)))

            # recompile and run
            model.compile(loss=loss, metrics=metrics, optimizer=opt)
            model.fit(train.X, train.y, validation_data=(validation.X,validation.y),
                epochs=epochs, callbacks=callbacks)
            
            # reset for next iteration
            base_model = model
        
        # now, unfreeze everything
        model.trainable = True

     
# define other models for other problems here

def cifar10_model(
        do_image_augmentation=True, 
        noise_stdev=0.15,
        height_mod = (0.1,0.1), width_mod = (0.1,0.1),
        dropout=0.5, 
        l2=0.001,
        output_activation='softmax'
):
    input_layer = model = tf.keras.Input(shape=(32, 32, 3))

    if do_image_augmentation:
        model = tf.keras.layers.GaussianNoise(noise_stdev)(model)
        model = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')(model)
        model = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_mod, width_mod, interpolation='nearest')(model)

    for f in [128, 256]:
        # Add the conv2d layers
        for _ in range(2):
            model = tfa.layers.WeightNormalization(
                tf.keras.layers.Conv2D(
                    filters=f, 
                    kernel_size=[3, 3],
                    padding='same', 
                    kernel_initializer=tf.keras.initializers.he_uniform(),
                    bias_initializer=tf.keras.initializers.constant(0.1),
                    kernel_regularizer=tf.keras.regularizers.l2(l2)
                ),
                data_init=False
            )(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.1)(model)
            # model = tf.keras.layers.BatchNormalization()(model)

        # Add the pooling and dropout
        model = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')(model)
        model = tf.keras.layers.Dropout(dropout)(model)

    # Add the final conv2D layers 
    model = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=[3, 3],
            padding='valid', 
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.constant(0.1),
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        ),
        data_init=False
    )(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.1)(model)
    # model = tf.keras.layers.BatchNormalization()(model)

    model = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=[1, 1],
            padding='valid', 
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.constant(0.1),
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        ),
        data_init=False
    )(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.1)(model)
    # model = tf.keras.layers.BatchNormalization()(model)

    model = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv2D(
            filters=128, 
            kernel_size=[1,1],
            padding='valid', 
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.constant(0.1),
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        ),
        data_init=False
    )(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.1)(model)
    # model = tf.keras.layers.BatchNormalization()(model)

    model = tf.keras.layers.AveragePooling2D((6,6))(model)

    model = tf.keras.layers.Flatten()(model)

    # TODO: normalize the output?
    output_layer = model = tf.keras.layers.Dense(
        units=10,
        activation=output_activation,
        kernel_initializer=tf.keras.initializers.he_uniform(),
        bias_initializer=tf.keras.initializers.constant(0.1),
        kernel_regularizer=tf.keras.regularizers.l2(l2)
    )(model)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
 
    return model


class Supervised():

    def __init__(self, model, args):
        self.model = model()
        self.args = args
        self.name = 'supervised'

    def fit(self, train, valid):

        opt = tf.keras.optimizers.Adam(self.args.lrate)
        self.model.compile(
            optimizer=opt, 
            loss='categorical_crossentropy',
            metrics=[
                tf.keras.metrics.CategoricalAccuracy()
            ]
        )

        #model.summary()

        history = self.model.fit(
            train.X, train.y,
            validation_data=(valid.X, valid.y), 
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_categorical_accuracy',
                    patience=self.args.patience,
                    restore_best_weights=True
                )
            ]
        ) 
        return self.model, history.history

    def predict(self, X):
        return self.model.predict(X)