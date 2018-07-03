from sklearn.ensemble import RandomForestClassifier

from deployml.sklearn.train.training import TrainingBase


class DecisionTree(TrainingBase):

    def __init__(self, number_of_trees=1, max_features='auto',
                 max_depth=None, min_samples_leaf=1):
        """
        Creates the decision tree object with training base functions
        :param forrest: If set to True, multiple decision trees are created
                        and averaged when calculating defaults
        :param number_of_trees: Defines the number of trees in the forrest,
                                higher may increase accuracy but slows down
                                computational speed
        """

        super().__init__(selected_model=RandomForestClassifier(n_estimators=number_of_trees,
                                                               max_features=max_features,
                                                               max_depth=max_depth,
                                                               min_samples_leaf=min_samples_leaf))
        if number_of_trees == 1:
            self.model_title = "Decision Tree"
        else:
            self.model_title = "Random Forest"
