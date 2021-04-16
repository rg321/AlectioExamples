## Toxic Comment Classification README

A simple example demonstrating the use of Alectio's SDK for the usecase "Multilabel Text Classification." This is relevant in cases where each sentence / document can have more than one label. Here, we use the popular kaggle dataset for [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). 

There's two examples provided. The one in the `../pytorch` directory is based on a pretrained BERT model while the one in this folder is based on Logistic Regression. It's based on this [baseline](https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams) provided by the competition itself.

#### Download data
```
aws s3 cp s3://alectio-datasets/toxic-classification ../data --recursive
```

The **most important thing** to note in this example is the format of the return of the `infer` function as demonstrated in line 119 of the example:
```
infer_preds = {k: {'logits': reverse_sigmoid(val)} for k, val in enumerate(infer_preds)}
```
So in practice, it would look something like this:
```
infer_preds = {1: {'logits': <logits>}, 2: {'logits': <logits>}, ...}
return {"output": op}
```

Note: This example does not train on the entire dataset. Please download the full dataset from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).