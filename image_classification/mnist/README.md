# Image Classification on MNIST

This example shows you how to build `train`, `test` and `infer` processes
for image classification problems. In particular, it will show you the format
of the return of `test` and `infer`. For an object detection problem, those
returns can be a little bit involved. But to get most out of Alectio's platform,
those returns needs to be correct. 

All of the following steps assume that your terminal points to the current directory, i.e. `./examples/image_classification`

### 1. Set up a virtual environment (optional) and install Alectio SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. 

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment. 

To install the AlectioSDK from within the current directory (`./examples/object_detection`) run:

```
pip install ../../.
```

Also create a directory `log` to store model checkpoints:
```
mkdir log
```

### 2. Build Train Process
We will train a [Basic CNN based on the official PyTorch tutorial] (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for
this demo. The model is defined in `model.py`. Feel free to change it as you please. 

To try out this step, run:

```
python model.py
```

### 3. Build Test Process
The test process tests the model trained in each active learning loop.
In this example, the test process is the `test` function defined 
in `processes.py`. 

```
python processes.py
```

#### Return of the Test Process 
You will need to run non-maximum suppression on the predictions on the test images and return 
the final detections along with the ground-truth bounding boxes and objects
on each image. 

The return of the `test` function is a dictionary 
```
{"predictions": predictions, "labels": labels}
```

Both `predictions` and `labels` are lists where `labels` denotes the ground truths for the images.

### 4. Build Infer Process
The infer process is used to apply the model to the unlabeled set to run inference. 
We will use the inferred output to estimate which of those unlabeled data will
be most valuable to your model.

Refer to main [AlectioSDK ReadMe](../../README.md) for general information regarding the 
arguments of this process.

#### Return of the Infer Process
The return of the infer process is a dictionary
```python
{"outputs": outputs}
```

`outputs` is a dictionary whose keys are the indices of the unlabeled
images. The value of `outputs[i]` is a dictionary that records the output of
the model on training image `i`. 

### 5. Build Flask App 
First you have to set up your main.py file to contain the token that is specific to your experiment. After you
create an experiment on the Alectio platform, you will receive a unique token that will be necessary to run your experiment.

Copy and paste that token into the main.py file under the token field within the Pipeline object.
```python
app = Pipeline(
    name="cifar10",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='<your-token-here>'
)
```
Once you have updated that file, execute the python file and you should be able to begin running your experiment.
```shell script
python main.py
```


