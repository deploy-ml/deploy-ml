from sklearn.svm import SVC

from deployml.sklearn.train.training import TrainingBase


class SVMBase(TrainingBase):

    def __init__(self, kernel=None):
        if kernel:
            super().__init__(selected_model=SVC(kernel=kernel))
            self.model_title = "Linear Support Vector Machine"
        else:
            super().__init__(selected_model=SVC())
            self.model_title = "Support Vector Machine"
        self.support_vector = True
