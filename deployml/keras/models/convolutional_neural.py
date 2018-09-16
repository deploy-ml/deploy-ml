from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization

from deployml.keras.train.training import TrainingBase
from deployml.keras.deploy.pickle_prep import make_keras_picklable


class ConvolutionalNeuralNetwork(TrainingBase):

    def __init__(self, optimizer='adam', input_dims=(28, 28), activation_fn='relu',
                 dropout_option=False, batch_norm=False, con_hidden_layers=[(32, 3, 3), (32, 3, 3)],
                 hidden_layers=(10, 10), pool_size=(2, 2), use_bias=True, alpha=0.3, first_layer=(32, 3, 3)):

        self.pos_path = None
        self.neg_path = None
        self.input_dim_one = input_dims[0]
        self.input_dim_two = input_dims[1]

        make_keras_picklable()

        if activation_fn == "leaky relu":
            activation_fn = LeakyReLU(alpha=alpha)
        elif activation_fn == 'elu':
            activation_fn = ELU(alpha=1.0)

        model = Sequential()

        model.add(Convolution2D(first_layer[0], (first_layer[1], first_layer[2]), activation='relu',
                                input_shape=(input_dims[0], input_dims[1], 3),
                                # data_format='channels_first'
                                ))

        # convolutional layers
        if dropout_option:
            if batch_norm:
                for i in con_hidden_layers:
                    model.add(Convolution2D(i[0], (i[1], i[2]), activation=activation_fn))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.5))
            else:
                for i in con_hidden_layers:
                    model.add(Convolution2D(i[0], ([1], i[2]), activation=activation_fn))
                    model.add(Dropout(0.5))

        else:
            if batch_norm:
                for i in con_hidden_layers:
                    model.add(Convolution2D(i[0], (i[1], i[2]), activation=activation_fn))
                    model.add(BatchNormalization())

            else:
                for i in con_hidden_layers:
                    model.add(Convolution2D(i[0], (i[1], i[2]), activation=activation_fn))

        model.add(MaxPooling2D(pool_size=(pool_size[0], pool_size[1]), strides=(2, 2)))
        if dropout_option:
            model.add(Dropout(0.5))

        # bridge
        model.add(Flatten())

        # normal layers
        if dropout_option:
            if batch_norm:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=use_bias))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.5))
            else:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=use_bias))
                    model.add(Dropout(0.5))
        else:
            if batch_norm:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=use_bias))
                    model.add(BatchNormalization())
            else:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=use_bias))

        # decision function (currently having issue,
        # expected to have shape (1,) but got array with shape (2,))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Activation("softmax"))

        # model.add(Dense(10, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        super().__init__(selected_model=model, convolutional=True, input_dims=input_dims)

    def set_image_paths(self, positive_path, negative_path):
        """
        Set image paths for loading data for neural network for testing and training.


        :param positive_path: string, path to where the positive images are
        :param negative_path: string, path to where the negative images are
        :return: defined paths ready for loading data
        """
        self.pos_path = positive_path
        self.neg_path = negative_path

test = ConvolutionalNeuralNetwork()
