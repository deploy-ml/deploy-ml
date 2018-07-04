"""
In this file we use deploy-ml to apply an sk-learn logistic regression to
cleaned Titanic data, we then evaluate it and deploy it
"""
import pickle

# we import logistic regression
from deployml.sklearn import LogisticRegressionBase

import pandas as pd

# we load the data
train = pd.read_csv('titanic_example.csv')

# Here we define the model
log = LogisticRegressionBase()

# we then define the data
log.data = train
train.drop('Unnamed: 0', inplace=True, axis=1)

# we then define what we're trying to predict in the data, in this case we are trying to predict survival
log.outcome_pointer = 'Survived'

# we then plot the learning curve, this also trains the model, and we scale the data using a min max
# scaler, the larger the batch size the quicker the training. This function also supports early stopping
log.plot_learning_curve(scale=True, scaling_tool='min max', batch_size=15)

# we then show the learning curve (with this small example data don't expect a good learning curve)
log.show_learning_curve()

# we then evaluate the outcome. This is just the sk-learn metrics function wrapped by deploy-ml
# how it's encouraged to be used as the metrics will be attached to the object and included in
# deployment
log.evaluate_outcome()

# We can also plot the ROC curve (again small data count so it will not be a smooth curve)
log.show_roc_curve()

# if we're happy with it we can deploy the algorithm, the scaler and variable input order will also be saved
log.deploy_model(description="trial to test the function", author="maxwell flitton",
                 organisation="Test", contact="maxwellflitton@gmail.com", file_name="trial.sav")

# we can then load the model
loaded_algorithm = pickle.load(open("trial.sav", 'rb'))

# we want to know what we need to put in and it's input order
print(loaded_algorithm['input order'])

# We can scale new data
input_data = loaded_algorithm['scaler'].transform([[2, 34, 0, 0, 50, 1, 0, 1]])

# and we can make a new prediction with the scaled data
new_prediction = loaded_algorithm['model'].predict_proba(input_data)[0][1]
print(new_prediction)

from deployml.sklearn import DecisionTree

log = DecisionTree(number_of_trees=10)

train = pd.read_csv('titanic_example.csv')
train.drop('Unnamed: 0', inplace=True, axis=1)

log.data = train

# we then define what we're trying to predict in the data, in this case we are trying to predict survival
log.outcome_pointer = 'Survived'

# we then plot the learning curve, this also trains the model, and we scale the data using a min max
# scaler, the larger the batch size the quicker the training. This function also supports early stopping
log.plot_learning_curve(scale=True, scaling_tool='min max', batch_size=15)

# we then show the learning curve (with this small example data don't expect a good learning curve)
log.show_learning_curve()

# we then evaluate the outcome. This is just the sk-learn metrics function wrapped by deploy-ml
# how it's encouraged to be used as the metrics will be attached to the object and included in
# deployment
log.evaluate_outcome()

# We can also plot the ROC curve (again small data count so it will not be a smooth curve)
log.show_roc_curve()

# if we're happy with it we can deploy the algorithm, the scaler and variable input order will also be saved
log.deploy_model(description="trial to test the function", author="maxwell flitton",
                 organisation="Test", contact="maxwellflitton@gmail.com", file_name="trial.sav")

# we can then load the model
loaded_algorithm = pickle.load(open("trial.sav", 'rb'))

# we want to know what we need to put in and it's input order
print(loaded_algorithm['input order'])

# We can scale new data
input_data = loaded_algorithm['scaler'].transform([[2, 34, 0, 0, 50, 1, 0, 1]])

# and we can make a new prediction with the scaled data
new_prediction = loaded_algorithm['model'].predict_proba(input_data)[0][1]
print(new_prediction)


from deployml.sklearn import NeuralNetworkBase

log = NeuralNetworkBase(hidden_layers=(5, 2))

train = pd.read_csv('titanic_example.csv')
train.drop('Unnamed: 0', inplace=True, axis=1)

log.data = train

# we then define what we're trying to predict in the data, in this case we are trying to predict survival
log.outcome_pointer = 'Survived'

# we then plot the learning curve, this also trains the model, and we scale the data using a min max
# scaler, the larger the batch size the quicker the training. This function also supports early stopping
log.plot_learning_curve(scale=True, scaling_tool='min max', batch_size=15)

# we then show the learning curve (with this small example data don't expect a good learning curve)
log.show_learning_curve()

# we then evaluate the outcome. This is just the sk-learn metrics function wrapped by deploy-ml
# how it's encouraged to be used as the metrics will be attached to the object and included in
# deployment
log.evaluate_outcome()

# We can also plot the ROC curve (again small data count so it will not be a smooth curve)
log.show_roc_curve()

# if we're happy with it we can deploy the algorithm, the scaler and variable input order will also be saved
log.deploy_model(description="trial to test the function", author="maxwell flitton",
                 organisation="Test", contact="maxwellflitton@gmail.com", file_name="trial.sav")

# we can then load the model
loaded_algorithm = pickle.load(open("trial.sav", 'rb'))

# we want to know what we need to put in and it's input order
print(loaded_algorithm['input order'])

# We can scale new data
input_data = loaded_algorithm['scaler'].transform([[2, 34, 0, 0, 50, 1, 0, 1]])

# and we can make a new prediction with the scaled data
new_prediction = loaded_algorithm['model'].predict_proba(input_data)[0][1]
print(new_prediction)
