from deployml.keras.train.training import TrainingBase

from deployml.keras.deploy.pickle_prep import make_keras_picklable


class ImportBase(TrainingBase):

    def __init__(self, model, model_title):
        make_keras_picklable()
        super().__init__(selected_model=model)
        self.model_title = model_title
