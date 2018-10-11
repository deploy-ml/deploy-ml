from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.metrics import roc_curve, classification_report, auc
from keras.preprocessing.image import ImageDataGenerator
from deployml.keras.train.loading_pictures import load_picture_data
from keras.preprocessing.image import img_to_array
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import numpy as np
import cv2

from deployml.keras.deploy.base import DeploymentBase


class TrainingBase(DeploymentBase):

    def __init__(self, selected_model, convolutional=False, input_dims=(28, 28)):
        """
        Base training functions, this class is usually inherited by a machine learning model
        so it's usually not created by itself
        :param selected_model: represents machine learning model. Usually passed by a
                               machine learning model object inheriting this class
        """
        super().__init__()
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
        self.predictions = None
        self.trained = False
        self.history = None
        self.X_report = None
        self.y_report = None
        self.general_report = "General Report not generated when model was trained"
        self.scaling_title = None
        self.input_order = None
        self.support_vector = False
        self.best_epoch = None
        self.best_model = None
        self.convolutional = convolutional
        self.dims_one = input_dims[0]
        self.dims_two = input_dims[1]
        self.y_cache = None

    def train(self, scale=False, scaling_tool='standard',
                    resample=False, resample_ratio=1, epochs=150, batch_size=100, verbose=0):
        """
        Trains a model quickly
        :param scale: if set True, the input data is scaled
        :param scaling_tool: defines the type of scaling tool used when pre-processing data
        :return: a trained model with no learning curve
        """

        if self.convolutional:
            self.X_train, self.X_test, self.y_train, self.y_test = load_picture_data(
                dims_one=self.dims_one, dims_two=self.dims_two
            )

            aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                     horizontal_flip=True, fill_mode="nearest")

            self.model.fit_generator(aug.flow(self.X_train, self.y_train, batch_size=batch_size),
                                     validation_data=(self.X_test, self.y_test),
                                     steps_per_epoch=len(self.X_train) / batch_size,
                                     epochs=epochs, verbose=1
                                     )

        else:
            self.X = self.data.drop(self.outcome_pointer, axis=1)
            self.y = self.data[self.outcome_pointer]
            self.input_order = list(self.X.columns.values)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                    random_state=101
                                                                                    )

            if resample:
                sm = SMOTE(ratio=resample_ratio)
                self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

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

            self.history = self.model.fit(self.X_train, self.y_train, validation_split=0.33,
                                          epochs=epochs, batch_size=batch_size, verbose=verbose)

    def show_learning_curve(self, save=False, metric='loss'):
        """
        :param save: if set to True plot will be saved as file
        Plots the learning curve of test and train sets
        """
        plt.figure(figsize=(15, 7))
        if metric == 'loss':
            plt.plot(self.model.history.history['loss'], "r-+", linewidth=2, label="train")
            plt.plot(self.model.history.history['val_loss'], "b-", linewidth=3, label="val")

        elif metric == 'accuracy':
            plt.plot(self.model.history.history['acc'], "r-+", linewidth=2, label="train")
            plt.plot(self.model.history.history['val_acc'], "b-", linewidth=3, label="val")

        plt.xlabel("Iterations")
        plt.ylabel('Error')
        plt.title('Learning Curve for {}'.format(self.model_title))
        plt.legend(loc='upper right')
        if save:
            plt.savefig('learning_curve')
        plt.show()

    def show_roc_curve(self, save=False):
        """
        Plots the ROC curve to see True and False positive trade off
        :param save: if set to True plot will be saved as file
        :return: self.auc which can be used as a score
        """
        y_pred_keras = self.model.predict(self.X_test).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.y_test, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

        if save:
            plt.savefig('ROC')
        plt.show()

# this cross val needs work, it's currently not supported by Keras
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
# this cross val needs work, it's currently not supported by Keras

    def calculate(self, input_array=None, happening=True, override=False, image_path=None):
        """
        Calculates probability of outcome
        WARNING [CANNOT BE USED ONCE MODEL IS PICKLED]
        :param input_array: array of inputs (should be same order as training data)
        :param happening: if set False, returns probability of event not happening
        :param override: set to True if you want to override scaling
        :param image_path: string of the path to the image being fed into the model
        :return: float between 0 and 1
        """
        if self.convolutional:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.dims_one, self.dims_two))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            return self.model.predict(image)[0][0]

        else:
            if self.scaled_inputs and not override:
                input_array = self.scaling_tool.transform(input_array)
            if happening:
                return self.model.predict([input_array])[0][0]
            else:
                return self.model.predict([input_array])[0][0]

    def evaluate_outcome(self, threshold=0.5):
        """
        Prints classification report of finished model
        :return: list of predictions from the X_test data subset
        """

        if self.convolutional:
            self.predictions = []
            for i in self.X_test:
                image = np.expand_dims(i, axis=0)
                calculation = self.model.predict(image)[0][0]
                if calculation >= threshold:
                    self.predictions.append(1)
                else:
                    self.predictions.append(0)

            self.y_cache = []
            for i in self.y_test:
                self.y_cache.append(i[0])
            self.general_report = classification_report(self.y_cache, self.predictions)
            print("Metrics for a cut off of: {}".format(0.5))
            print(self.general_report)

        else:
            self.predictions = self.model.predict_classes(self.X_test)

            self.general_report = classification_report(self.y_test, self.predictions)
            print(self.general_report)
