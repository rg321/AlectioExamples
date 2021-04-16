# Sentiment Classification on the Yelp Review Polarity Dataset using TensorFlow v2

This example is intended to show you how to build the `train`, `test` and `infer` processes for the `AlectioSDK` for text sentiment
classification problems. We will use the [Yelp Review Polarity](https://www.kaggle.com/irustandi/yelp-review-polarity/version/1) dataset. Each sample in this dataset is a news article. The objective is to classify the each Yelp review as either positive or negative as seen below:

| label | sentiment|
| ----- | ------  |
| 1    | Positive |
| 2     | Negative |


Since the size of this dataset is large, it shall be downloaded when the program is run.

*** All of the following steps assume that your terminal points to the current directory, i.e. `./examples/sentiment_analysis` ***

### 1. Set up a virtual environment and install Alectio SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment.

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment.

To install the AlectioSDK from within the current directory (`./examples/sentiment_analysis`) run:

```
pip install ../../.
```

#### 2. Download the data and create a log directory
Create a log directory in the project root to save checkpoints and download the dataset in a separate `data` folder.
```
mkdir data && mkdir log
aws s3 sync s3://alectio-datasets/Yelppolarity
```
### 3. Build Train, Test and Infer Processes
The train, test and infer processes are implemented in [`processes.py`](./processes.py).

### 6. Run
Finally, execute:

```
python main.py
```
## Return Types

The return from the `test` and `infer` will be sent to Alectio's platform for
computing the model performance and making active learning decisions.
Therefore, it is important to make sure that they have the expected format.

#### Format of the Test Outputs
Return from the `test` process will be used to compute performance metrics of
your model. The return is a dictionary with two keys

| key | value |
| --- | ----- |
| predictions | a dictionary of test data index and model's prediction |
| labels | a dictionary of test data index and its ground-truth label |

For classification problem, the model's prediction and
the ground-truth of each test sample is an integer indicating the class label.

For example, if the test set consists of `(n + 1)` samples indexed by `(0, 1, ..., n)`,
then the value of `predictions` looks like
```python
{
    0: x0,
    1: x1,
    ...
    n: xn
}
```
where `xi` is the integer class label of sample `i` predicted by the model.

The value of `labels` looks like
```python
{
    0: y0,
    1: y1,
    ...
    n: yn
}
```
where `yi` is the integer ground-truth class label of sample `i`.

#### Format of the Infer Outputs
Return from the `infer` process will be used to understand which data points to labeled
in the next step.

```
outputs_fin = {}
for j in range(len(outputs)):
    outputs_fin[j] = {}
    outputs_fin[j]["prediction"] = predicted[j].item()
    outputs_fin[j]["logits"] = outputs[j].cpu().numpy()
```

The above code block can be used as a template. Here, the returned items should be a dictionary with
the following information for each batch: predictions by the model, and the activations of the penultimate layer.

Finally, after filling that up, we return the result of the `infer` step like so:
```
return {"outputs": outputs_fin}
```
