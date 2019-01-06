import numpy as np


class DmlLoader:

    def __init__(self, file_path):
        if file_path.split(".")[-1] != "dml":
            print("file is not dml file")
        else:
            f = open(file_path, "r")
            data = f.read()
            data = data.split("$$@")
            self.data = {}

            for i in data:
                i = i.split("££@")
                if len(i) == 2:
                    self.data[i[0]] = i[1]

        self.intercept = None
        self.weights = None

    def configure_model(self):
        if self.data["model title"] == "Logistic Regression":
            self.data["weight vector"] = self.data["weight vector"].replace("[", "").replace("]", "").split(",")
            self.data["weight vector"] = [float(i) for i in self.data["weight vector"]]
            self.weights = np.array(self.data["weight vector"])
            self.intercept = float(self.data["intercept"])
        else:
            print("unrecognised model")

    def calculate(self, data):
        data = np.array(data)

        if self.data["model title"] == "Logistic Regression":
            x = np.matmul(self.weights, np.transpose(data)) + self.intercept
            return 1/(1+np.exp(-x))

        else:
            print("unrecognised model")
