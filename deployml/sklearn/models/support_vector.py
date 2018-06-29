from sklearn.svm import SVC

from deployml.sklearn.train.training import TrainingBase


class SVMBase(TrainingBase):

    def __init__(self):
        super().__init__(selected_model=SVC())
        self.model_title = "Support Vector Machine"
        self.support_vector = True
