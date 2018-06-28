from sklearn.neural_network import MLPClassifier

from deployml.keras.train.training import TrainingBase


class NeuralNetworkBase(TrainingBase):

    def __init__(self, tensor=False, hidden_units=[10, 10], n_classes=10, train_data=None,
                 batch_size=50, steps=40000, optimizer='Adagrad', activation_fn='relu',
                 dropout=None, config=None, keras=False,
                 first_layer=15, keras_optimizer='adam', alpha=0.3, dropout_option=False,
                 batch_norm=False
                 ):
        if tensor:
            import tensorflow as tf
            if activation_fn == 'relu':
                activation_fn = tf.nn.relu
            feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(train_data)
            net_model = tf.contrib.learn.DNNClassifier(hidden_units=hidden_units,
                                                       n_classes=n_classes,
                                                       feature_columns=feature_cols,
                                                       # batch_size=batch_size,
                                                       # steps=steps,
                                                       optimizer=optimizer,
                                                       activation_fn=activation_fn,
                                                       dropout=dropout,
                                                       # input_layer_partitioner=input_layer_partitioner,
                                                       config=config,
                                                       # warm_start_from=warm_start_from
                                                       )
            super().__init__(selected_model=tf.contrib.learn.SKCompat(net_model),
                             tensor=True, batch_size=batch_size, steps=steps)
            self.model_title = "TensorFlow Neural Network"
        elif keras:
            from keras.models import Sequential
            from keras.layers import Dense, Activation
            from keras.layers import Dropout
            from keras.layers.normalization import BatchNormalization

            if activation_fn == "leaky relu":
                from keras.layers import LeakyReLU
                activation_fn = LeakyReLU(alpha=alpha)
            elif activation_fn == 'elu':
                from keras.layers import ELU
                activation_fn = ELU(alpha=1.0)

            model = Sequential()
            model.add(Dense(first_layer, input_dim=n_classes, activation=activation_fn, use_bias=True))
            if dropout_option:
                if batch_norm:
                    for i in hidden_units:
                        model.add(Dense(i, activation=activation_fn, use_bias=True))
                        model.add(BatchNormalization())
                        model.add(Dropout(0.5))
                else:
                    for i in hidden_units:
                        model.add(Dense(i, activation=activation_fn, use_bias=True))
                        model.add(Dropout(0.5))
            else:
                if batch_norm:
                    for i in hidden_units:
                        model.add(Dense(i, activation=activation_fn, use_bias=True))
                        model.add(BatchNormalization())
                else:
                    for i in hidden_units:
                        model.add(Dense(i, activation=activation_fn, use_bias=True))
            model.add(Dense(1, activation='sigmoid'))
            # if dropout_option:
            #     model.add(Dropout(0.5))

            model.compile(loss='binary_crossentropy', optimizer=keras_optimizer, metrics=['accuracy'])
            super().__init__(selected_model=model, keras=True, batch_size=batch_size, steps=steps)
        else:
            super().__init__(selected_model=MLPClassifier(activation=activation_fn))
            self.model_title = "Neural Network"
        self.hidden_layers = (2, 2)
        self.weights = {}
        self.intercept = 0
        self.penalty = None
        self.best_penalty = None
        self.structure = None
        self.activation_function = 'relu'

    def define_hidden_layers(self):
        """
        Defines the hidden layers structure from the hidden layers
        attribute
        :return: model attribute with redefined layer structure
        """
        self.model.hidden_layer_sizes = self.hidden_layers

    def define_activation_function(self, activation_function):
        self.model.activation = activation_function
        self.activation_function = activation_function

    def define_solver(self, solver):
        self.model.solver = solver

    def define_network_structure(self):
        """
        Defines the structure of the matrix weights, useful for
        printing
        """
        self.structure = [matrix.shape for matrix in self.model.coefs_]