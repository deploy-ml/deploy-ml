# Deploy-ML for SK-Learn Documentation
Deploy-ML supports the training, evaluation, and packaging of the SK-Learn library:

* [models](# Sk-Lean Models)
* [training/testing](# SK-Learn Training)
* [deploying your model](# SK-Learn Deployment)
* [loading and using your model](# Loading Model)


# Sk-Lean Models
Deploy-ML supports, trains and packages models by the machine learning package SK-Learn. The following models are currently supported:

* [Import](## import)
* [Logistic Regression](## Logistic Regression)
* [Support Vector Machine](## Support Vector Machine)
* [Neural Network](## Neural Network)
* [Decision Tree and Random Forest](## Decision Tree and Random Forest)

## Import 
Deploy-ML aims to make the process of training, testing and deploying machine learning easier and simplier. However, sometimes the model required is not directly supported, or a collegue has already defined a model. The import class is for users who define their own sk-learn model, but want to use the package for training and deployment reasons. To use your sk-learn model, import the import class:

```python
from deployml.sklearn.models.import import ImportBase
```

### Parameters

* **model** = Sk-Learn model that you have imported and defined 
* **model_title** = string describing the title of the model for deployment package

### Practical Example

Below is an example of importing a logistic regression model from sk-learn, and using the deploy-ml import class:

```python
from sklearn.linear_model import LogisticRegression
from deployml.sklearn.models.import import ImportBase


imported_model = LogisticRegression(penalty=’l2’, dual=False, 
								    tol=0.0001, C=1.0, fit_intercept=True, 
								    intercept_scaling=1, class_weight=None, 
								    random_state=None, solver=’liblinear’, 
								    max_iter=100, multi_class=’ovr’, verbose=0, 
								    warm_start=False, n_jobs=1)
							
model = ImportBase(model=imported_model,
				     model_title="Logistic Regression")
```

Now the imported model has all the training and deployment functions of deploy-ml.

## Logistic Regression
Logistic Regression is a well established statistical classification technique. 

### Parameters

* **penalty** = string defining the logistic regression penality, default is "l2" but can also be "l1"

### Attributes 

* **weights**: dictionary that gets populated with weights and labels for statistical use after the model is trained and the define_weights method is fired.
*  **intercept**: The model intercept, defined after the model is trained and the define_weights method is fired
*  **penalty**: Penality used in the model 
*  **best_penalty**: initially set to zero, this is the penality that gives the best accuracy score. This is defined when the grid_search function is fired
*  **model_title**: Set to "Logistic Regression" for deployment package  

### Methods 
Logistic regression has the following methods:
#### grid_search
This method takes no arguements. When fired the log model is trained with "l1" and "l2" penalities. The accuracy scores are compared. The penality with the best accuracy score is defines the **best_penalty** attributes. Data needs to be defined with an **outcome_pointer** before this method can be fired. For information on assigning data and defining the outcome_pointer please look at the training documentation. 
#### define_weights
This method takes no arguements. It gets the weights of the trained model and populates the **weights** attribute with labels and weights. Please understand training before firing this function.

### Practical Example 
Below is a practical example of training a logistic regression model 


```python 
from deployml.sklearn.models.logistic_regression import LogisticRegressionBase

# We define the model
log = LogisticRegressionBase()

# We define the data (pandas data frame)
log.data = input_data

# We define the key of the column we are trying to predict
log.outcome_pointer = 'attended'

# We then use grid search to find out the best penalty
log.grid_search()

# We then define the penalty with the best penalty 
log.define_penalty(penalty=log.best_penalty)

# Now we have the best penalty, we produce a training curve and show it 
# with scaled data using a standard scaler
log.plot_learning_curve(scale=True, batch_size=100)
log.show_learning_curve()

# We then print out the precision, recall and F-1 score 
log.evaluate_outcome()

# And show the ROC curve 
log.show_roc_curve()
```

This is how the logistic regression model is trained and evaluated. Please look at the deployment documentation on how to package and deploy your trained model.

## Support Vector Machine 

Support vector machines are robust with little to no parameters to tune. They can handle inbalanced data well without resampling. However, they give either a 1 or a 0. Not a range of numbers between 1 and 0 reducing the overall control the end user has as they cannot look at the outcome and decide on cut off parameters to suit their needs. Support vector machines are useful for filters, chopping out loan defaults or ruling someone in for a medical test. However, be careful, only employ this if the recall (percentage of category actually caught and recalled) is high enough for your satifaction. 

### Parameters 
The support vector machine has no parameters as of yet.

### Attributes 

* ***model_title***: Set to "Support Vector Machine" for deployment package
* ***support_vector***: set to True. This has implications in the predicting probability function as the model can only give 0 or 1 and nothing inbetween

### Methods 
The support vector machine has no unique methods yet, but does have all the standard training and deployment methods.

### Practical Example
Below is a practical example of training a support vector machine:


```python 
from deployml.sklearn.models.support_vector import SVMBase

# We define the model
svm = SVMBase()

# We define the data (pandas data frame)
svm.data = input_data

# We define the key of the column we are trying to predict
svm.outcome_pointer = 'attended'

# We now train the support vector machine. These things can take a lot
# of time to train so it's advise to use the quick_train method with 
# scaled data 
svm.quick_train(scale=True)

# We then print out the precision, recall and F-1 score 
svm.evaluate_outcome()

# And show the ROC curve 
svm.show_roc_curve()
```

## Neural Network
SK-Learn supports some form of basic neural network. This is good as an initial go to approach. However, it is advised to look into packages such as Keras and Tensorflow for more complex neural networks. 
### Parameters 

* ***hidden_layers*** = a tuple defining the number of neurons per hidden layer. default is (100,) which means one hidden layer of 100 neurons. (100, 100) would mean two hidden layers of 100 neurons.
* ***activation_fn*** = the activation function used in the neurons. Default is "relu". Other activation functions include: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
* ***solver*** = This is the type of algorithm used for the gradient descent. Default is "adam", other choices include: {‘lbfgs’, ‘sgd’, ‘adam’}
* ***alpha*** = The L2 regularization penalty, default is: 0.0001
* ***learning_rate*** = default: "constant", choices: {‘constant’, ‘invscaling’, ‘adaptive’}
* ***learning_rate_init*** = default: 0.001
* ***momentum*** = momentum of the gradient descent, default: 0.9

### Attributes

* ***model_title***: set to "Neural Network" for deployment package
* ***structure***: Initially None, gets defined to the matrix dimensions when the define_structure method is fired. Whilst hard coding algorithms is not recomended, this gives the user a high-level blueprint of what structure the neural network should undergo 

### Methods 
The neural network has the following methods: 

#### define activation function
This method takes an activation function arguement, defining it for the neural network. This enables the user to try and compare multiple activation functions without having to create new model classes.

```python
neural = NeuralNetworkBase()
neural.define_activation_function("logistic")
```

#### define network structure
This method takes no arguements. When fired is defines the ***structure*** atribute with a list of shapes of the matrices of the neural network model. Useful for reference.

### Practical Example 

```python 
from deployml.sklearn.models.neural_network import NeuralNetworkBase

# We define the model
NN = NeuralNetworkBase(hidden_layers=(4, 4))

# We define the data (pandas data frame)
NN.data = input_data

# We define the key of the column we are trying to predict
NN.outcome_pointer = 'attended'

# Now we've defined the network we produce a training curve and show it 
# with scaled data using a standard scaler
NN.plot_learning_curve(scale=True, batch_size=100)
NN.show_learning_curve()

# We then print out the precision, recall and F-1 score 
NN.evaluate_outcome()

# And show the ROC curve 
NN.show_roc_curve()
```

## Decision Tree and Random Forest
Decision trees can be useful in classification. With just one tree we call is a decision tree. With multiple trees we call it a random forest. 

### Parameters 

* ***number_of_trees*** = this is the number of trees involved in training and calculating a probability. The more trees the less likely the model is going to over fit. You will start to see diminishing returns past the number of 300.
* ***max_features*** = defualt is "auto". This is the number of features considered when looking for a best split. If “auto”, then max_features=sqrt(n_features), If “sqrt”, then max_features=sqrt(n_features) (same as “auto”), If “log2”, then max_features=log2(n_features), If None, then max_features=n_features
* ***max_depth*** = defualt is None. This is an int and denotes the maximum depth of a tree. This reduces the chances of the model over fitting. 
* ***min_samples _leaf*** =  default is 1. This is the minimum number of samples required for a category to be a leaf node. 

### Attributes

* ***model_title***: set to "Random Forest" for deployment package if number of trees is above one. "Decision Tree" if not.

### Methods 
The tree model has no unique methods yet, but does have all the standard training and deployment methods.

### Practical Example 

```python 
from deployml.sklearn.models.decision_tree import DecisionTree

# We define the model
DT = DecisionTree(number_of_trees=200)

# We define the data (pandas data frame)
DT.data = input_data

# We define the key of the column we are trying to predict
DT.outcome_pointer = 'attended'

# We now train the random forest. These things can take a lot
# of time to train so it's advise to use the quick_train method with 
# scaled data 
DT.quick_train(scale=True)

# We then print out the precision, recall and F-1 score 
DT.evaluate_outcome()

# And show the ROC curve 
DT.show_roc_curve()
```

# SK-Learn Training
All models from the SK-Learn folder inherit the same training class. This gives the model the following training methods:

## plot_learning _curve
This trains the model. It's slower than the quick train but it gives you a training curve, and the ability to stop the training early. The data that was defined in the model is split into test, train.

### Parameters 
* ***batch_size*** = defualt is 100. This is the number of datapoints for each step. If the dataset size is big, a bigger batch_size is recomended. 
* ***scale*** = default is False. If set to True then a data scaler is fitted around the training data and stored. The training data is then scaled. The testing data is scaled using the same scaler that was fitted on the training data. The saved scaling tool is packaged with the model when deployed so it can be used in production.
* ***scaling_tool*** = default is "standard". The scaling tool can also be "min max" or "normalize". However, be careful with normalizing. This is just precrocessing the data. It's not stored like other scalers as it's not fitted to any data. 
* ***remsample*** = default is False. If set to True, the training data is resampled using the SMOTE algorithm in order to address unbalanced data. This is done after the spliting of data into test and train to prevent resampled datapoints bleeding into the test data giving a false high accuracy in testing. 
* ***resample_ratio*** = default is 1. This is the ratio of one outcome to another. If left at one, this means that there will be 50% of the training data belonging to one category, and the other 50% belonging to the other (a ratio of 1 to 1)
* ***early_stopping*** = default is False. If set to True, the training stops at a defined number of iterations. This should only be set to True after you have seen a full learning curve to see where the model is overfitting.
* ***cut_off*** = default is 30. This is the number of iterations before the model stops training. This will vary depending on the ***batch_size***. When a full training curve is produced, if you spot signs of where the algorithm is starting to overfit, the ***cut_off*** should be the X-value on the graph of where this is happening.

## quick_train
This is pretty much the same as the learning curve. The only difference is that all the training is done in one cycle. The positive to this is that it's much quicker. The downside is that a learning curve cannot be displayed, and early stopping cannot be supported. 

### Parameters
* ***scale*** = default is False. If set to True then a data scaler is fitted around the training data and stored. The training data is then scaled. The testing data is scaled using the same scaler that was fitted on the training data. The saved scaling tool is packaged with the model when deployed so it can be used in production.
* ***scaling_tool*** = default is "standard". The scaling tool can also be "min max" or "normalize". However, be careful with normalizing. This is just precrocessing the data. It's not stored like other scalers as it's not fitted to any data. 
* ***remsample*** = default is False. If set to True, the training data is resampled using the SMOTE algorithm in order to address unbalanced data. This is done after the spliting of data into test and train to prevent resampled datapoints bleeding into the test data giving a false high accuracy in testing. 
* ***resample_ratio*** = default is 1. This is the ratio of one outcome to another. If left at one, this means that there will be 50% of the training data belonging to one category, and the other 50% belonging to the other (a ratio of 1 to 1)

## show_learning _curve
This function shows the learning curve. Must only be fired if the ***plot_learning _curve*** function is fired. 

### Parameters 

*  ***save*** = default is False. If set to True, learning curve will be saved as a file in the folder where the script is running. 


## show_roc _curve
This function displays the ROC curve for false positives and true positives trade-off

### Parameters

*  ***save*** = default is False. If set to True, ROC curve will be saved as a file in the folder where the script is running. 


## evaluate_outcome
This function takes no arguements. It gets the testing data, and evaluates the model, giving accuracy and recall. The report is also cached so it can be packaged when the model is deployed. 

## evaluate_cross _validation
This function produces a cross validation test printing the average score. To access the list of all the scores in the cross validation, simply access the ***cross_val*** attribute.

### Parameters 

* ***n_splits*** = number of folds used for cross-validation. default is 10


# SK-Learn Deployment
If the metrics are to your liking, then its ready for deployment. This can be done by the following code is an example of deploying the logistic regression model which has the arbitrary definition of log:

```python
log.deploy_model(description="example model for documentation",
					author="Maxwell Flitton",
					organisation="Example Organisation",
					contact="example@gmail.com",
					file_name="example_model.sav")

```
This then saves the model in the directory of where the script is running. 

The saved package is essentially a dictionary with the following keys:

* ***model_title*** = This is automatically defined by the model you selected
* ***model*** = This is the actual model that accepts data and makes a prediction.
* ***description*** = Short description of what the model does, this is defined as parameter in the ***deploy_model*** method
* ***author*** = String describing the name of the person who trained the model, this is defined as a parameter in the ***deploy_model*** method
* ***organisation*** = String describing the organisation where the model was trained, this is defined as a parameter in the ***deploy_model*** method
* ***prediction_target*** = This is the outcome that the model is trying to predict. This is automatically defined in the training process
* ***scaler*** = scaling object used in the training process. This is automatically defined in the training process
* ***scaling used*** = string describing the type of scaler used. Automatically defined in the training process.
* ***contact*** = This can be left blank. It's the contact info of the person who trained the model. This is defined in the parameter of the ***deploy_model*** method 
* ***date*** = date and time when the model is packaged for deployment. Automatically defined in the deployment process
* ***metrics*** = This is the precision, recall, and accuracy of the model. This is automatically defined in the training process
* ***input order*** = This is a list of strings that defines the order in which parameters need to be fed into the algorithm for calculation. This is automatically defined in the training process. 
* ***system version*** = This is the version of python that the model was trained and packaged on. This shouldn't matter too much but might be important if having to configure a server to run the model on. This is automatically configured in the deployment process.
* ***sklearn version*** = This is automatically defined in the deployment process

# Loading Model
You can use Pickle to load the saved model:

```python 
import pickle

loaded_model = pickle.load(open("example_model.sav", "rb"))

```
We can use the scaler in the loaded model to scale the input data:

```python
input_data = loaded_model['scaler'].transform([[24, 500, 0, 0, 400,
    										           500, 10, 1, 12, 1, 0]])

```

Now the data is scaled, we can get a prediction from the packaged model:

```python
loaded_model['model'].predict_proba(input_data)[0][1]

```