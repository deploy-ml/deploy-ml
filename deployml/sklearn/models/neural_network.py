from sklearn.neural_network import MLPClassifier

from deployml.sklearn.train.training import TrainingBase


class NeuralNetworkBase(TrainingBase):

    def __init__(self, activation_fn='relu'):
        super().__init__(selected_model=MLPClassifier(activation=activation_fn))
        self.model_title = "Neural Network"
        self.hidden_layers = (2, 2)
        self.weights = {}
        self.intercept = 0
        self.penalty = None
        self.best_penalty = None
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
