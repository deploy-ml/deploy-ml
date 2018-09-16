from keras.preprocessing.image import img_to_array
import numpy as np
import pickle
import keras
import sys
import cv2


class KerasLoader:

    def __init__(self, file_path):
        if file_path[-3:] == "sav":
            package = pickle.load(open(file_path, "rb"))
        else:
            print("error wrong format no 'sav' found")
            package = None

        self.model = package["model"]
        self.convolutional = package["convolutional"]
        if self.convolutional:
            self.dims_one, self.dims_two = package["image dims"]
        self.scaling_tool = package["scaler"]
        if self.scaling_tool:
            self.scaled_inputs = True
        else:
            self.scaled_inputs = False
        if package["system version"] != str(sys.version):
            print("warning! model was trained in {}. You're running {}".format(package["system version"],
                                                                               str(sys.version)))
        if package["Keras Version"] != str(keras.__version__):
            print("warning! model was trained on {}. You're running {}".format(package["Keras Version"],
                                                                               str(keras.__version__)))

    def calculate(self, input_array=None, happening=True, override=False, image=None):
        """
        Calculates probability of outcome
        :param input_array: array of inputs (should be same order as training data)
        :param happening: if set False, returns probability of event not happening
        :param override: set to True if you want to override scaling
        :param image: image object that has been read
        :return: float between 0 and 1
        """
        if self.convolutional:
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