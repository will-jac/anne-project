from weight_normalization import WeightNormalization
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

class Scifar10Model(tf.keras.Model):

    def __init__(self, do_image_augmentation=True):

        super(Scifar10Model, self).__init__() 

        self.do_image_augmentation = do_image_augmentation
        # create the network with dropout

        self._conv1a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv1b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv1c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._conv2a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv2b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv2c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)
        self._conv3a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=[3, 3],
                padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv3b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[1, 1],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv3c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[1, 1],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._dense = WeightNormalization(
            tf.keras.layers.Dense(
                units=10, 
                activation=tf.nn.softmax,
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )

    def call(self, input, training=True):
        
        # add the stochastic augmentation to the input if we're training
        if training and self.do_image_augmentation:
            h = gaussian_noise(input, 0.15)
            h = image_augmentation(h)
        else:
            h = input
        
        # pass the (augmented) input through the model
        h = self._conv1a(h, training)
        h = self._conv1b(h, training)
        h = self._conv1c(h, training)
        h = self._pool1(h)
        h = self._dropout1(h, training=training)

        h = self._conv2a(h, training)
        h = self._conv2b(h, training)
        h = self._conv2c(h, training)
        h = self._pool2(h)
        h = self._dropout2(h, training=training)

        h = self._conv3a(h, training)
        h = self._conv3b(h, training)
        h = self._conv3c(h, training)

        # Average Pooling
        h = tf.reduce_mean(h, axis=[1, 2])
        return self._dense(h, training)

    def pretrain(self, train_X, validation_X, epochs, lrate, L2_reg,
            loss='mse', out_activation='softmax', callbacks=None, metrics=None):
        opt = tf.keras.optimizers.Adam(lr=lrate)
        # pretraining will work on each conv2d chunk
        base_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=train_X.shape[1]), 
                self._conv1a, self._conv1b, self._conv1c, 
                self._pool1, self._dropout1,
                tf.keras.layers.Dense(train_X.shape[1], activation=out_activation, name='output', use_bias=True, kernel_regularizer=L2(L2_reg))
            ]
        )
        base_model.compile(loss=loss, metrics=metrics, optimizer=opt)
        base_model.fit(train_X, train_X, validation_data=(validation_X,validation_X),
            epochs=epochs, callbacks=callbacks)

        for layers in [
            [self._conv2a, self._conv2b, self._conv2c, self._pool2, self._dropout2],
            [self._conv3a, self._conv3b, self._conv3c]
        ]:
            base_model.trainable = False

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=train_X.shape[1]))
            for layer in base_model.layers[:-1]:
                model.add(layer)

            # add the next batch of layers
            for layer in layers:
                model.add(layer)

            # add the output back
            model.add(tf.keras.layers.Dense(train_X.shape[1], activation=out_activation, name='output1', use_bias=True, kernel_regularizer=L2(L2_reg)))

            # recompile and run
            model.compile(loss=loss, metrics=metrics, optimizer=opt)
            model.fit(train_X, train_X, validation_data=(validation_X,validation_X),
                epochs=epochs, callbacks=callbacks)
            
            # reset for next iteration
            base_model = model
        
        # now, unfreeze everything
        model.trainable = True


        



# define other models for other problems here