from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU, ELU

from deployml.keras.train.training import TrainingBase
from deployml.keras.deploy.pickle_prep import make_keras_picklable


class RecurrentNeuralNetwork(TrainingBase):

    def __init__(self, hidden_layers=[10, 10], n_classes=10,
                 batch_size=50, activation_fn='relu',
                 first_layer=15, optimizer='adam', alpha=0.3, dropout_option=False,
                 batch_norm=False, use_bias=True
                 ):

        make_keras_picklable()

        if activation_fn == "leaky relu":
            activation_fn = LeakyReLU(alpha=alpha)
        elif activation_fn == 'elu':
            activation_fn = ELU(alpha=1.0)

        model = Sequential()
        model.add(Dense(first_layer, input_dim=n_classes, activation=activation_fn, use_bias=use_bias))
        if dropout_option:
            if batch_norm:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=True))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.5))
            else:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=True))
                    model.add(Dropout(0.5))
        else:
            if batch_norm:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=True))
                    model.add(BatchNormalization())
            else:
                for i in hidden_layers:
                    model.add(Dense(i, activation=activation_fn, use_bias=True))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        super().__init__(selected_model=model, batch_size=batch_size)