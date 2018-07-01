from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from deployml.sklearn.train.training import TrainingBase


class DecisionTree(TrainingBase):

    def __init__(self, forrest=False, number_of_trees=None):
        """
        Creates the decision tree object with training base functions
        :param forrest: If set to True, multiple decision trees are created
                        and averaged when calculating defaults
        :param number_of_trees: Defines the number of trees in the forrest,
                                higher may increase accuracy but slows down
                                computational speed
        """
        if forrest:
            self.model_title = "Random Forrest"
            if number_of_trees:
                super().__init__(selected_model=RandomForestClassifier(n_estimators=number_of_trees))
            else:
                super().__init__(selected_model=RandomForestClassifier())
        else:
            super().__init__(selected_model=DecisionTreeClassifier())
            self.model_title = "Decision Tree"
        self.penalty = None
        self.best_penalty = None
