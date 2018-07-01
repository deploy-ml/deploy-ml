from deployml.sklearn.train.training import TrainingBase


class ImportBase(TrainingBase):

    def __init__(self, model, model_title):
        super().__init__(selected_model=model)
        self.model_title = model_title
