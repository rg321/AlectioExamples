# Object detection on COCO

This example shows you how to build `train`, `test` and `infer` processes
for object detection problems. In particular, it will show you the format
of the return of `test` and `infer`. For an object detection problem, those
return can be a little bit involved. But to get most out of Alectio's platform,
those returns needs to be correct.

Since the goal of this tutorial is to show you how to setup a pipeline,
we are not going to train on the entire COCO dataset. Instead, we will
only use 100 samples for training set and 100 samples for test set.
And we will use yolo-v3 as the model.

*** All of the following steps assume that your terminal points to the current directory, i.e. `./examples/object_detection` ***

### 1. Set up a virtual environment and install Alectio SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. The most important part of that tutorial, if you are a paid customer, is that you set the backend ip address to the address that we give you by using the set_backend_ip.py script. **If your backend ip is not configured, then this SDK will be non-functional.**

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment.

To install the AlectioSDK from within the current directory (`./examples/object_detection`) run:

```
pip install ../../.
```

Make sure to also set the environment variables so that you replace the value in set_env.sh with your API key. You can also directly assign the API key to the environment variable named ALECTIO_API_KEY. 

```
export ALECTIO_API_KEY=<your downloaded api key>
```

### 2. Install Requirements

Install the requirements via:
```
pip install -r requirements.txt
```

### 3. Download Pre-Processed Data
Next, you can create a data and a log directory in the current directory and download the data via:

```
mkdir data
mkdir log
cd data
aws s3 cp s3://alectio-coco/COCO-mega . --recursive
cd ..
mkdir weights
cd weights
aws s3 cp s3://alectio-resources/cocoweights . --recursive
```

For most part the data is preprocessed according to the Darknet convention. The only difference is that we use the `xyxy` format for the ground-truth bounding boxes.

### 3. Build Train Process
We will train a [Darknet yolov3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) for
this demo. The model is defined in `model.py` the configuration file that specifies the
architecture of the model is defined in `yolov3.cfg`.

To try out this step, run:

```
python model.py
```

Refer to main [AlectioSDK ReadMe](../../README.md) for general information regarding the
arguments of this process.

### 4. Build Test Process
The test process tests the model trained in each active learning loop.
In this example, the test process is the `test` function defined
in `processes.py`.

```
python processes.py
```

Refer to main [AlectioSDK Readme](../../README.md) for general information regarding the
arguments of this process.

#### Return of the Test Process
You will need to run non-maximum suppression on the predictions on the test images and return
the final detections along with the ground-truth bounding boxes and objects
on each image.

The return of the `test` function is a dictionary
```
    {"predictions": prd, "labels": lbs}

```

`prd` is a dictionary whose keys are the indices of the test
images. The value of `prd[i]` is a dictionary that records the final
detections on test image `i`,

**Keys of `prd[i]`**

boxes (`List[List[float]]`)
>  A list of detected bouding boxes.
    Each bounding box should be normalized according
    to the dimension of test image `i` and it
    should be in `xyxy`-format.

scores (`List[float]`)
> A list of objectedness of each detected
   bounding box. Objectness should be in \[0, 1\].

objects (`List[int]`)
> A list of class label of each detected
    bounding box.


`lbs` is a dictionary whose keys are the indices of the test images.
The value `lbs[i]` is a dictionary that records the ground-truth bounding
boxes and class labels on the image `i`.

**Keys of `lbs[i]`**

boxes (`List[List[float]]`)
> A list of ground-truth bounding boxes on image `i`.
    Each bounding box should normalized according to the dimension
    of test image `i` and it should be in `xyxy`-format.

objects (`List[int]`)
> A list of class label of each ground-truth bounding box.

difficulties (`Optional[List[{0,1}]]`)
> A list of difficulties of each ground-truth object.
   An object is 'difficult' if it is difficult for human to detect.
   For example, an object can be difficult if it is extremely small.
   Use 1 for difficult objects and 0 for non-difficult objects.
   Difficult objects will not be accounted when calculating mAP.
   If you skip this field, then all objects are assumed to be non-difficult


### 5. Build Infer Process
The infer process is used to apply the model to the unlabeled set to run inference.
We will use the inferred output to estimate which of those unlabeled data will
be most valuable to your model.

#### Return of the Infer Process
There is no return value for the infer process. Rather, you will be adding pre-softmax values to 
the sqlite database using the sql client provided in the alectio sdk. Refer to the `infer()` function
within `processes.py` on an example of specific usage. 

Generally speaking you can access this client by using
`alectio_sdk.sdk.sql_client` where there exist functions `add_index()` and `create_database()`. 
We use a local sqlite database where you specify to be stored using your config.yaml file. Before you
begin recording your infer outputs, you will create a connection to this local database, and then
begin adding rows to the database. When you are done, you close the connection, and that is all that is needed.


### 6. Build Dataset state process
The dataset state process helps the Alectio team to generate a reference database with a mapping between list of indices and the corresponding imagenames within your custom dataset object.
The return of the `getdatasetstate` function is a dictionary. For example  a dict named ` trainstate ` that returns the indices(int) in the dataset object mapped to imagenames(str) will look like this

```
trainstate ={ 1: "images/train/cat1.jpg,
              2: "images/train/dog1.jpg,
              3: "images/train/bike0.jpg,
              ...
              ...
            }

```
Refer to `processes.py` to get more information about the format of this function. This `getdatasetstate` function is the fourth and final process that needs to be created and wrapped into the our custom wrapper `pipeline` in `main.py`

### 7. Run the Flask App 
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


