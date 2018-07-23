# deploy-ml
This open source library is esentually a wrapper for popular machine learning libraries. It streamlines, and simplifies the training and deployment of algrithms. Right now the library is in its infancy. We are working to get the core functionality up and running in order to be tested at Imperial College London.


## Installation
You don't need this source code unless you want to modify the package. If you just want to use the package, just run:
```
pip install --upgrade deployml`
```

Install from source with:

```
python3 setup.py install
```

## Requirements

Python 3+ (PyPy supported)

You should be good to go from there.

## Examples
You can find a simple example for each module under the folder examples/[MODULE-NAME]

## Documentation
You can find documentation for each module on how to train, test and package machine learning in the README file.

## Getting Involved
As this is still a very new project, we have enabled bug reporting and feature requesting. Also if you're good at python or doing research please let us know and get involved in the development.

## Dependancies
Modules are activated seperately meaning that you don't have to have everything. However, you will have to have the sk-learn library as this is used a lot for the data processing in pretty much all the models. Keras and Tensorflow are only needed if you activate
