# Ember Classification Problem

This example shows you how to build `train`, `test` and `infer` processes
for tabular data. The ember dataset is an open source cybersecurity dataset that is a binary classification problem.
Either examples are 'benign' or 'malicious'. You can find the full details here: https://github.com/endgameinc/ember

All of the following steps assume that your terminal points to the current directory, i.e. `./examples/tabular_data/ember`

### 1. Set up a virtual environment (optional) and install Alectio SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. 

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment. 

To install the AlectioSDK from within the current directory (`./examples/tabular_data`) run:

```
pip install ../../.
```

Also create a directory `log` and `data` to store model checkpoints:
```
mkdir log
mkdir data
```

Within this directory download and pip install the ember dataset from the ember github repository.
Place the vectorized features within your data directory.

### 2. Build Train, Test and Infer Processes
Inside `processes.py` you will find the functions `infer()`, `test()`, and `train()` implemented. These functions
are the core functions that are needed to implement your model. In this case, LightGBM is used with the built in 
sklearn wrapper. Therefore, if you are using an sklearn model, this process should be similar. The only part that is
necessary for us to do active learning using an sklearn model is that the sklearn model has a `predict_proba()` function.


### 3. Build Flask App 
Finally, to run the flask app, execute:
```
python main.py --config config.yaml
```

### Return of the Test Process
The return of the test process is a dictionary where keys are the indices of the current unlabeled pool of data, and 
within each value is another dictionary which contains the predict_proba output for that example (under the key 'softmax'), 
and 'predictions' which is just the model's prediction on that index.
```python
{"softmax": predict_proba[i], "predictions": prediction[i]}
```

#### Return of the Infer Process
The return of the infer process is a dictionary
```python
{"outputs": outputs}
```

`outputs` is a dictionary whose keys are the indices of the unlabeled
images. The value of `outputs[i]` is a dictionary that records the output of
the model on training example i.
