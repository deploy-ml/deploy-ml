from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import numpy as np

from deployml.sklearn.deploy.base import DeploymentBase


class TrainingBase(DeploymentBase):

    def __init__(self, selected_model, batch_size=None, steps=None):
        """
        Base training functions, this class is usually inherited by a machine learning model
        so it's usually not created by itself
        :param selected_model: represents machine learning model. Usually passed by a
                               machine learning model object inheriting this class
        """
        super().__init__()
        self.batch_size = batch_size
        self.steps = steps
        self.auc = 0
        self.cross_val = 0
        self.model = selected_model
        self.data = None
        self.outcome_pointer = None
        self.X = None
        self.scaled_inputs = False
        self.scaling_tool = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.outcome_metrics = None
        self.train_errors = []
        self.test_errors = []
        self.predictions = None
        self.trained = False
        self.learning_curve = False
        self.grid = 0
        self.X_report = None
        self.y_report = None
        self.general_report = "General Report not generated when model was trained"
        self.scaling_title = None
        self.input_order = None
        self.support_vector = False
        self.best_epoch = None
        self.best_model = None

    def plot_learning_curve(self, batch_size=100, starting_point=100, scale=False, scaling_tool='standard',
                            resample=False, resample_ratio=1.0, early_stopping=False, cut_off=30):
        """
        Generates lists of training and testing error through the training process
        which can be plotted to check for over fitting
        :param batch_size: How many data points get trained in each cycle (cannot be zero)
        :param starting_point: first batch to be trained (cannot be zero)
        :param scale: if set True, the input data is scaled
        :param scaling_tool: defines the type of scaling tool used when pre-processing data
        :return: trained model with a learning curve
        """
        self.train_errors = []
        self.test_errors = []
        self.scaled_inputs = False
        self.X = self.data.drop(self.outcome_pointer, axis=1)
        self.input_order = list(self.X.columns.values)
        self.y = self.data[self.outcome_pointer]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=101
                                                                                )

        if resample:
            sm = SMOTE(ratio=resample_ratio)
            self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

            # self.X_train = sampling_data.drop(self.outcome_pointer, axis=1)
            # self.y_train = sampling_data[self.outcome_pointer]

        self.X_report = np.array(self.X_test)
        self.y_report = np.array(self.y_test)

        if scale:
            self.scaled_inputs = True
            self.scaling_title = scaling_tool
            if scaling_tool == 'standard':
                self.scaled_inputs = True
                self.scaling_tool = StandardScaler()
                self.scaling_tool.fit(self.X_train)
                self.X_train = self.scaling_tool.transform(self.X_train)
                self.X_test = self.scaling_tool.transform(self.X_test)
            elif scaling_tool == 'min max':
                self.scaled_inputs = True
                self.scaling_tool = MinMaxScaler()
                self.scaling_tool.fit(self.X_train)
                self.X_train = self.scaling_tool.transform(self.X_train)
                self.X_test = self.scaling_tool.transform(self.X_test)
            elif scaling_tool == 'normalize':
                self.scaling_tool = normalize(self.X_train)

        else:
            self.scaled_inputs = False

        if early_stopping:
            for i in range(starting_point, len(self.X_train), batch_size):

                self.model.fit(self.X_train[:i], self.y_train[:i])

                y_train_predict = self.model.predict(self.X_train[:i])
                y_test_predict = self.model.predict(self.X_test)

                self.train_errors.append(mean_squared_error(y_train_predict, self.y_train[:i]))
                self.test_errors.append(mean_squared_error(y_test_predict, self.y_test))
                if len(self.train_errors) == cut_off:
                    break

        else:
            for i in range(starting_point, len(self.X_train), batch_size):

                self.model.fit(self.X_train[:i], self.y_train[:i])

                y_train_predict = self.model.predict(self.X_train[:i])
                y_test_predict = self.model.predict(self.X_test)

                self.train_errors.append(mean_squared_error(y_train_predict, self.y_train[:i]))
                self.test_errors.append(mean_squared_error(y_test_predict, self.y_test))

    def quick_train(self, scale=False, scaling_tool='standard',
                    resample=False, resample_ratio=1.0):
        """
        Trains a model quickly
        :param scale: if set True, the input data is scaled
        :param scaling_tool: defines the type of scaling tool used when pre-processing data
        :return: a trained model with no learning curve
        """
        self.learning_curve = False
        self.X = self.data.drop(self.outcome_pointer, axis=1)
        self.y = self.data[self.outcome_pointer]
        self.input_order = list(self.X.columns.values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=101
                                                                                )

        if resample:
            sm = SMOTE(ratio=resample_ratio)
            self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

            # self.X_train = sampling_data.drop(self.outcome_pointer, axis=1)
            # self.y_train = sampling_data[self.outcome_pointer]

        self.X_report = np.array(self.X_test)
        self.y_report = np.array(self.y_test)

        if scale:
            self.scaled_inputs = True
            if scaling_tool == 'standard':
                self.scaling_tool = StandardScaler()
            elif scaling_tool == 'min max':
                self.scaling_tool = MinMaxScaler()
            elif scaling_tool == 'normalize':
                self.scaling_tool = normalize()
            self.scaling_tool.fit(self.X_train)
            self.X_train = self.scaling_tool.transform(self.X_train)
            self.X_test = self.scaling_tool.transform(self.X_test)
        else:
            self.scaled_inputs = False

        self.model.fit(self.X_train, self.y_train)

    def show_learning_curve(self, save=False):
        """
        :param save: if set to True plot will be saved as file
        Plots the learning curve of test and train sets
        """
        plt.figure(figsize=(15, 7))
        plt.plot(np.sqrt(self.train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(self.test_errors), "b-", linewidth=3, label="val")
        plt.xlabel("Iterations")
        plt.ylabel('Error')
        plt.title('Learning Curve for {}'.format(self.model_title))
        plt.legend(loc='upper right')
        if save:
            plt.savefig('{} learning_curve'.format(self.model_title))
        plt.show()

    def show_roc_curve(self, save=False):
        """
        Plots the ROC curve to see True and False positive trade off
        :param save: if set to True plot will be saved as file
        :return: self.auc which can be used as a score
        """
        logit_roc_auc = roc_auc_score(self.y_test, self.model.predict(self.X_test))
        self.auc = logit_roc_auc
        fpr, tpr, thresholds = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='RPC Curve (area = {}0.2f)'.format(logit_roc_auc))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if save:
            plt.savefig('{} ROC'.format(self.model_title))
        plt.show()

    def evaluate_outcome(self, best=False):
        """
        Prints classification report of finished model
        :return: list of predictions from the X_test data subset
        """
        if best:
            self.predictions = self.best_model.predict(self.X_test)
        else:
            self.predictions = self.model.predict(self.X_test)
        self.general_report = classification_report(self.y_test, self.predictions)
        print(self.general_report)

    def evaluate_cross_validation(self, n_splits=10, random_state=7):
        """
        Performs a cross validation score evaluating how the model performs in different subsets
        of the data, model needs to be trained first
        :return: average value of all 10 scores
        """
        k_fold = KFold(n_splits=n_splits, random_state=random_state)
        scoring = 'accuracy'
        self.cross_val = cross_val_score(self.model, self.X_train, self.y_train, cv=k_fold, scoring=scoring)
        print("{}-fold cross validation average accuracy: {}".format(n_splits, self.cross_val.mean()))

    def grid_search(self):
        """
        override this in you machine learning model class
        :return: Nothing, supposed to be overridden in parent class
        """
        self.grid = 1

    def calculate(self, input_array, happening=True, override=False):
        """
        Calculates probability of outcome
        WARNING [CANNOT BE USED ONCE MODEL IS PICKLED]
        :param input_array: array of inputs (should be same order as training data)
        :param happening: if set False, returns probability of event not happening
        :param override: set to True if you want to override scaling
        :return: float between 0 and 1
        """
        if self.scaled_inputs and not override:
            input_array = self.scaling_tool.transform(input_array)
        if happening:
            return self.model.predict_proba([input_array])[0][1]
        else:
            return self.model.predict_proba([input_array])[0][0]