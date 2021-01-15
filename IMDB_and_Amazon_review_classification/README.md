### Running steps

Make sure you are on the `develop` branch, with `IMDB_and_Amazon_review_classification` as your working directory. 

Note: If you want to run an MNIST task, set the DATASET argument in config.yaml to "MNIST", 
otherwise if you want to run FASHION-MNIST, set the DATASET argument in config.yaml to "Fashion". 

Configure demo state
1. Create a data directory `mkdir data` and a log directory `mkdir log`
3. Modify hyper-parameters in the `config.yaml` file as needed (to set the train_epochs, batch_size, learning rate, momentum, etc)

Set up environment

4. Find and set the Alectio backend API key (found in the front end platform) with `export ALECTIO_API_KEY="KEY_HERE"`
5. Run `aws configure` to set up connection with AWS
6. Install the alectio-sdk with `pip install ../../.` (assuming you are in this current directory).
7. After activating your virtual environment (see project-level README for instructions), install all dependencies in the requirements.txt found in this repository `pip install -r requirements.txt`
8. Finally, run `python main.py --config config.yaml` to start the traning process. 

## Instructions for downloading the IMDB dataset
1. Download the IMDB dataset here: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Rename the CSV file to "imdb_reviews.csv" and place in the root directory. 

## Instructions for downloading the Amazon Reviews dataset. 
1. Because this dataset is so vast, we will only be training on a very very small subset of it (approximately 72k samples). You can change the amount we train on currently (3% of train.csv) by modifying the arg AMAZON_DATSET_TRAINING_RATIO in config.yaml
2. Download the file amazon_review_full_csv.tar.gz from this public google drive folder with many common NLP datasets. 
https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
3. Download and unzip the files. Place amazon_review_full_csv folder in this experiment folder.
4. Please note that for this task we will be grouping reviews of 4-5 as being "good", reviews of 1-2 as being "bad" and eliminating all neutral reviews (3)



# IMDB and Amazon Review Classification

This example shows you how to build `train`, `test` and `infer` processes
for image classification problems. In particular, it will show you the format
of the return of `test` and `infer`. For an object detection problem, those
returns can be a little bit involved. But to get most out of Alectio's platform,
those returns needs to be correct. 

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

