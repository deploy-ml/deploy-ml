from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import numpy as np
import random
import cv2
import os
import glob


def load_picture_data(dims_one=28, dims_two=28, outcome_pointer="positive",
                      positive_file="positive_images", negative_file="negative_images", file_type="png"):

    positive_images = glob.glob("{}/*.{}".format(positive_file, file_type))

    negative_images = glob.glob("{}/*.{}".format(negative_file, file_type))

    data = []
    labels = []

    image_paths = positive_images + negative_images

    random.seed(42)
    random.shuffle(image_paths)

    # loop over the input images
    for imagePath in image_paths:

        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (dims_one, dims_two))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == positive_file else 0
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    # self.X_train, self.X_test, self.y_train, self.y_test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return X_train, X_test, y_train, y_test

    # construct the image generator for data augmentation
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #                          horizontal_flip=True, fill_mode="nearest")
