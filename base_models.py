from weight_normalization import MeanOnlyWeightNormalization
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def gaussian_noise(input, std):

    noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
    return input + noise

def image_augmentation(image):

    random_shifts = np.random.randint(-2, 2, (image.numpy().shape[0], 2))
    random_transformations = tfa.image.translations_to_projective_transforms(random_shifts)
    image = tfa.image.transform(image, random_transformations, 'NEAREST', output_shape=tf.convert_to_tensor(image.numpy().shape[1:3], dtype=np.int32)) 

    return image

class Cifar10Model(tf.keras.Model):

    def __init__(self, do_image_augmentation=True):

        super(Cifar10Model, self).__init__() 

        self.out_size = 10

        self.do_image_augmentation = do_image_augmentation
        # create the network with dropout

        ## TODO: check relu usage. I changed to tanh because that's what's more typically used

        activation = 'tanh'
        self._conv1a = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._conv1b = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation,
            )
        )
        self._conv1c = tfa.layers.WeightNormalization( 
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation,
            )
        )
        self._pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._conv2a = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._conv2b = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._conv2c = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=[3, 3],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)
        self._conv3a = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._conv3b = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[1, 1],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._conv3c = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[1, 1],
                padding="same", 
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation=activation
            )
        )
        self._dense = tfa.layers.WeightNormalization(
            tf.keras.layers.Dense(
                units=10, 
                activation='sigmoid',## TODO: check what this should be - I changed to sigmoid bc that makes more sense with crossentropy
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )

        self._layers = [
            [self._conv1a, self._conv1b, self._conv1c, self._pool1, self._dropout1],
            [self._conv2a, self._conv2b, self._conv2c, self._pool2, self._dropout2],
            [self._conv3a, self._conv3b, self._conv3c, self._dense]
        ]

    def call(self, inputs, training=True):
        
        # add the stochastic augmentation to the input if we're training
        if training and self.do_image_augmentation:
            h = gaussian_noise(inputs, 0.15)
            h = image_augmentation(h)
        else:
            h = inputs
        
        # pass the (augmented) input through the model
        h = self._conv1a(h, training=training)
        h = self._conv1b(h, training=training)
        h = self._conv1c(h, training=training)
        h = self._pool1(h, training=training)
        h = self._dropout1(h, training=training)

        h = self._conv2a(h, training=training)
        h = self._conv2b(h, training=training)
        h = self._conv2c(h, training=training)
        h = self._pool2(h, training=training)
        h = self._dropout2(h, training=training)

        h = self._conv3a(h, training=training)
        h = self._conv3b(h, training=training)
        h = self._conv3c(h, training=training)

        # Average Pooling
        h = tf.reduce_mean(h, axis=[1, 2])
        return self._dense(h, training=training)

    def pretrain(self, train, validation, epochs, lrate=0.001, L2_reg=0.001,
            loss='mse', out_activation='sigmoid', callbacks=None, metrics=None):
        opt = tf.keras.optimizers.Adam(lr=lrate)
        # pretraining will work on each conv2d chunk
        base_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=train.X.shape[1]), 
                self._conv1a, self._conv1b, self._conv1c, 
                self._pool1, self._dropout1,
                tf.keras.layers.Dense(train.y.shape[1], activation=out_activation, name='sigmoid', use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(L2_reg))
            ]
        )
        base_model.compile(loss=loss, metrics=metrics, optimizer=opt)
        base_model.fit(train.X, train.y, validation_data=(validation.X,validation.y),
            epochs=epochs, callbacks=callbacks)

        for layers in [
            [self._conv2a, self._conv2b, self._conv2c, self._pool2, self._dropout2],
            [self._conv3a, self._conv3b, self._conv3c]
        ]:
            base_model.trainable = False

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=train.X.shape[1]))
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