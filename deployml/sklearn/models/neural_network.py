from sklearn.neural_network import MLPClassifier

from deployml.sklearn.train.training import TrainingBase


class NeuralNetworkBase(TrainingBase):

    def __init__(self, hidden_layers=(100,), activation_fn='relu',
                 solver='adam', alpha=0., learning_rate="constant",
                 learning_rate_init=0.001, momentum=0.9):
        super().__init__(selected_model=MLPClassifier(activation=activation_fn, hidden_layer_sizes=hidden_layers,
                                                      solver=solver, alpha=alpha, learning_rate=learning_rate,
                                                      learning_rate_init=learning_rate_init, momentum=momentum))
        self.model_title = "Neural Network"
        self.hidden_layers = hidden_layers
        self.structure = None
        self.activation_function = activation_fn

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
