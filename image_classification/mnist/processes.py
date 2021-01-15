# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:22 2020

@author: arun
"""
import os
import yaml
import pandas as pd
import tensorflow as tf
from utils import *
from model import digitClassifier
from tensorflow.examples.tutorials.mnist import input_data

tf.enable_eager_execution()
tfe = tf.contrib.eager


##### Global variables
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
NN = digitClassifier(256, 256, 10)
optimizer = tf.train.AdamOptimizer(learning_rate=args["INITIAL_LR"])
checkpoint_prefix = os.path.join(args["OUTPUT_DIRECTORY"], "ckpt_0")
checkpoint = tf.train.Checkpoint(model=NN, optimizer=optimizer)


def train(args, labeled, resume_from, ckpt_file):

    labelledimages = mnist.train.images[labeled, ...]
    labelledclasses = mnist.train.labels[labeled, ...]

    dataset = tf.data.Dataset.from_tensor_slices((labelledimages, labelledclasses))

    train_count = labelledimages.shape[0]
    steps_per_epoch = train_count // args["BATCH_SIZE"]
    average_loss = 0.0
    average_acc = 0.0
    best_acc = 0.0

    grad = tfe.implicit_gradients(loss_fn)
    dataset = dataset.repeat().batch(args["BATCH_SIZE"]).prefetch(args["BATCH_SIZE"])
    dataset_iter = tfe.Iterator(dataset)

    for epoch in range(args["train_epochs"]):
        for step in range(steps_per_epoch):
            d = dataset_iter.next()
            x_batch = d[0]
            y_batch = tf.cast(d[1], dtype=tf.int64)
            batch_loss = loss_fn(NN, x_batch, y_batch)
            average_loss += batch_loss
            batch_accuracy = accuracy_fn(NN, x_batch, y_batch)
            average_acc += batch_accuracy

            if step == 0:
                print("Initial loss= {:.9f}".format(average_loss))

            optimizer.apply_gradients(grad(NN, x_batch, y_batch))

            # Display info
            if (step + 1) % args["DISPLAY_STEP"] == 0 or step == 0:
                if step > 0:
                    average_loss /= args["DISPLAY_STEP"]
                    average_acc /= args["DISPLAY_STEP"]
                print(
                    "Epoch:",
                    "%03d" % (epoch + 1),
                    "Step:",
                    "%04d" % (step + 1),
                    " loss=",
                    "{:.9f}".format(average_loss),
                    " accuracy=",
                    "{:.4f}".format(average_acc),
                )
                average_loss = 0.0
                average_acc = 0.0

        if average_acc > best_acc:
            best_acc = average_acc
            checkpoint.save(file_prefix=checkpoint_prefix)

    return args["OUTPUT_DIRECTORY"]


def test(args, ckpt_file):
    testX = mnist.test.images
    testY = mnist.test.labels
    print("Loading latest checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(args["OUTPUT_DIRECTORY"]))

    test_acc = accuracy_fn(NN, testX, testY)
    print("Testset Accuracy: {:.4f}".format(test_acc))
    y_pred, y_true = getpreds(NN, testX, testY)

    return {
        "predictions": y_pred.numpy().reshape(len(y_pred), 1),
        "labels": y_true.numpy().reshape(len(y_true), 1),
    }


def infer(args, unlabeled, ckpt_file):

    # Load the last best model
    print("Loading latest checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(args["OUTPUT_DIRECTORY"]))
    unlabelledimages = mnist.train.images[unlabeled, ...]
    unlabelledclasses = mnist.train.labels[unlabeled, ...]
    presoft = NN(unlabelledimages)
    presoftarr = presoft.numpy().reshape(len(presoft), 1)

    predictions = {i: row for i, row in enumerate(presoftarr)}

    return {"outputs": predictions}


def getdatasetstate(args, split="train"):
    columns = ["Label"] + list(range(args["NUM_INPUT"]))
    if split == "train":
        train_label_file = "{}mnist_train.csv".format(args["LABEL_DIRECTORY"])
        dataframe = pd.read_csv(train_label_file)
        dataframe.columns = columns
    else:
        test_label_file = "{}mnist_test.csv".format(args["LABEL_DIRECTORY"])
        dataframe = pd.read_csv(test_label_file)
        dataframe.columns = columns

    pathdict = {}
    for ix, row in dataframe.iterrrows():
        pathdict[ix] = row[
            "Label"
        ]  ######### here index carries the necessary reference

    return pathdict


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
