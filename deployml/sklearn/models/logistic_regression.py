from deployml.sklearn.train.training import TrainingBase
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class LogisticRegressionBase(TrainingBase):

    def __init__(self, penalty='l2'):
        super().__init__(selected_model=LogisticRegression(penalty=penalty))
        self.log = []
        self.weights = {}
        self.intercept = 0
        self.penalty = penalty
        self.best_penalty = None
        self.model_title = "Logistic Regression"

    def define_penalty(self, penalty):
        """
        Changes penalty of logistic regression model
        :param penalty: string, should be l1 or l2
        :return: redefines the model attribute with penalty
        """
        self.penalty = penalty
        self.model = LogisticRegression(penalty=self.penalty)

    def define_weights(self):
        """
        Defines the weights from the last training event
        :return: Fills the weights dictionary with weights and their labels
        """
        for i in range(0, len(self.model.coef_[0]) - 1):
            self.weights[self.X.columns.values[i]] = self.model.coef_[0][i]
        self.intercept = self.model.intercept_[0]

    def grid_search(self):
        """
        Trains the model with both l1 and l2 penalty
        :return: best penalty based on accuracy score
        """
        self.X = self.data.drop(self.outcome_pointer, axis=1)
        self.y = self.data[self.outcome_pointer]
        penalty_list = ['l1', 'l2']
        hyper_parameters = {'penalty': penalty_list}
        clf = GridSearchCV(self.model, hyper_parameters, cv=5, scoring='accuracy')
        best_model = clf.fit(self.X, self.y)
        self.best_penalty = best_model.best_estimator_.get_params()['penalty']
