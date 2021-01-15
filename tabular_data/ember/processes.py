import time
import os
import random
import json
import sys
import lightgbm as lgb
import yaml
import pandas as pd
import numpy as np
import ember
import argparse
import joblib
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report


def get_classification_report(model, x_test, y_test):
    # save model
    pred = model.predict(x_test)
    acc_score = accuracy_score(y_true=y_test, y_pred=pred)
    print()
    print(strat + " classification report: ")
    print(classification_report(y_true=y_test, y_pred=pred))
    print()
    print(strat + " accuracy: " + str(acc_score))


def preprocess_data(args):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(
        args["DATA_DIR"] + "/ember_2017_2/"
    )
    X_orig_train = pd.DataFrame(X_train)
    X_orig_test = pd.DataFrame(X_test)
    y_orig_train = pd.DataFrame(y_train)
    y_orig_test = pd.DataFrame(y_test)

    train_samples = random.sample(list(X_orig_train.index), 100000)
    test_samples = random.sample(list(X_orig_test.index), 25000)
    X_train = X_orig_train.loc[train_samples, :]
    X_train.reset_index(inplace=True, drop=True)
    X_test = X_orig_test.loc[test_samples, :]
    X_test.reset_index(inplace=True, drop=True)
    y_train = pd.Series(y_orig_train.loc[train_samples, :].values.ravel())
    y_test = pd.Series(y_orig_test.loc[test_samples, :].values.ravel())

    test_neg_label = [i for i, x in enumerate(y_test) if x == -1]
    X_test.drop(test_neg_label, axis=0, inplace=True)
    y_test.drop(test_neg_label, axis=0, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    train_neg_label = [i for i, x in enumerate(y_train) if x == -1]
    X_train.drop(train_neg_label, axis=0, inplace=True)
    y_train.drop(train_neg_label, axis=0, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_train = X_train[: args["TRAIN_SIZE"]]
    y_train = y_train[: args["TRAIN_SIZE"]]
    X_test = X_test[: args["TEST_SIZE"]]
    y_test = y_test[: args["TEST_SIZE"]]

    X_train.to_pickle(args["DATA_DIR"] + "X_train.pkl")
    y_train.to_pickle(args["DATA_DIR"] + "y_train.pkl")
    X_test.to_pickle(args["DATA_DIR"] + "X_test.pkl")
    y_test.to_pickle(args["DATA_DIR"] + "y_test.pkl")


def load_data(args):
    X_train = pd.read_pickle(args["DATA_DIR"] + "X_train.pkl")
    y_train = pd.read_pickle(args["DATA_DIR"] + "y_train.pkl")
    X_test = pd.read_pickle(args["DATA_DIR"] + "X_test.pkl")
    y_test = pd.read_pickle(args["DATA_DIR"] + "y_test.pkl")
    return X_train, y_train, X_test, y_test


def train(args, labeled, resume_from, ckpt_file):
    print("Starting training...")
    if resume_from is None:
        preprocess_data(args)
    X_train, y_train, X_test, y_test = load_data(args)
    model = LGBMClassifier(objective="binary", n_jobs=-1)
    if resume_from:
        print("Resuming from previous...")
        resume_from = os.path.join(args["EXPT_DIR"], resume_from)
        model = joblib.load(resume_from)
    # print(X_train.head)
    model.fit(X_train.loc[labeled], y_train[labeled])
    ckpt_path = os.path.join(args["EXPT_DIR"], ckpt_file)
    joblib.dump(model, ckpt_path)
    print("Finished Training. Saving the model as {}".format(ckpt_path))
    return


def test(args, ckpt_file):
    X_train, y_train, X_test, y_test = load_data(args)
    ckpt_path = os.path.join(args["EXPT_DIR"], ckpt_file)
    model = joblib.load(ckpt_path)
    predictions = model.predict(X_test)
    targets = y_test
    acc_score = accuracy_score(y_true=y_test, y_pred=predictions)
    print()
    print(" Test Classification Report: ")
    print(classification_report(y_true=y_test, y_pred=predictions))
    print()
    print(" Accuracy: " + str(acc_score))
    # save prediction
    with open(os.path.join(args["EXPT_DIR"], "prediction.pkl"), "wb") as f:
        pickle.dump(predictions, f)
    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    # load unlabeled data
    X_train, y_train, X_test, y_test = load_data(args)
    # load ckpt trained in the current loop
    ckptpath = os.path.join(args["EXPT_DIR"], ckpt_file)
    model = joblib.load(ckptpath)
    predict_proba = model.predict_proba(X_train)
    predictions = model.predict(X_train)
    output = {}
    for rec in unlabeled:
        output[rec] = {}
        output[rec]["softmax"] = predict_proba[rec]
        output[rec]["prediction"] = predictions[rec]

    return {"outputs": output}


def getdatasetstate(args={}):
    return {k: k for k in range(40000)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config.yaml", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)

    labeled = [int(i) for i in range(500)]
    unlabeled = [i for i in range(1000, 2000)]
    ckpt_file = "ckpt_0"
    preprocess_data(args)
    train(args, labeled, None, ckpt_file)
    test(args, ckpt_file)
    infer(args, unlabeled, ckpt_file)
