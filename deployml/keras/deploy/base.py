from datetime import datetime
import sys
import keras
import pickle


class DeploymentBase:

    def __init__(self):
        self.package = {}
        self.model_title = "please attach a trained model for deployment"
        self.model = None
        self.general_report = None
        self.outcome_pointer = None
        self.scaled_inputs = None
        self.scaling_title = None
        self.scaling_tool = None
        self.convolutional = None
        self.dims_one = None
        self.dims_two = None

    def deploy_model(self, description, author, organisation, file_name, contact="no contact information provided"):
        """
        Run before deploying the package
        :param description: short description of what the model does
        :param author: short string of the person deploying the model
        :param organisation: short string of organisation
        :param file_name: string, name of file to be saved
        :param contact: short string of where to find you, can be left
        :return: populates the package dictionary with the model, scaler if used,
                 inputs, versions the model was trained in, datetime, and inputs
                 so it is suitable for deployment
        """
        self.package["model title"] = self.model_title
        self.package["model"] = self.model
        self.package["description"] = description
        self.package["author"] = author
        self.package["organisation"] = organisation
        self.package["contact"] = contact
        self.package["date"] = str(datetime.now())
        self.package["metrics"] = self.general_report
        self.package["convolutional"] = self.convolutional
        if self.scaled_inputs:
            self.package["scaler"] = self.scaling_tool
            self.package["scaling used"] = self.scaling_title
        else:
            self.package["scaler"] = None
            self.package["scaling used"] = "Data was not scaled for training"
        self.package["system version"] = str(sys.version)
        self.package["Keras Version"] = str(keras.__version__)
        self.package["package version"] = "1"
        if self.convolutional:
            self.package["image dims"] = (self.dims_one, self.dims_two)
        else:
            self.package["input order"] = list(self.X.columns.values)
            self.package["prediction target"] = self.outcome_pointer
        self.package["package type"] = "pickle"
        pickle.dump(self.package, open(file_name, 'wb'))

    def deploy_keras_model(self, description, author, organisation, file_name, contact="no contact information provided"):
        """
        Run before deploying the package
        :param description: short description of what the model does
        :param author: short string of the person deploying the model
        :param organisation: short string of organisation
        :param file_name: string, name of file to be saved
        :param contact: short string of where to find you, can be left
        :return: populates the package dictionary with the model, scaler if used,
                 inputs, versions the model was trained in, datetime, and inputs
                 so it is suitable for deployment
        """
        import keras
        self.package["model title"] = self.model_title
        # self.package["model"] = self.model
        self.package["description"] = description
        self.package["author"] = author
        self.package["organisation"] = organisation
        self.package["contact"] = contact
        self.package["date"] = str(datetime.now())
        self.package["metrics"] = self.general_report
        self.package["input order"] = list(self.X.columns.values)
        self.package["prediction target"] = self.outcome_pointer
        if self.scaled_inputs:
            self.package["scaler"] = self.scaling_tool
            self.package["scaling used"] = self.scaling_title
        else:
            self.package["scaler"] = "Data was not scaled for training"
        self.package["system version"] = str(sys.version)
        self.package["Keras Version"] = keras.__version__
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

