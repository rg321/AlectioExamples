# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:22 2020

@author: arun
"""

import os
import argparse
from tqdm import tqdm
import os
from zipfile import ZipFile
from urllib.request import urlopen
import shutil
import pandas as pd
from time import time
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)
from keras.optimizers import Adam
import csv
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras import backend as K
from skimage.io import imread
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from utils import *


##### Global variables
FOLD = 1


def train(args, labeled, resume_from, ckpt_file):
    # Create new output directory for individual folds from timestamp
    print("Currently processing fold ", FOLD)
    output_directory = "{}{}/".format(args["OUTPUT_DIRECTORY"], FOLD)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Prepare training, validation and testing labels for kth fold
    train_label_file = "{}train_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
    val_label_file = "{}val_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
    train_dataframe = pd.read_csv(train_label_file)
    train_dataframe["Absolutefilename"] = (
        args["IMG_DIRECTORY"] + train_dataframe["Filename"]
    )
    val_dataframe = pd.read_csv(val_label_file)
    val_dataframe["Absolutefilename"] = (
        args["IMG_DIRECTORY"] + val_dataframe["Filename"]
    )
    currlabelledtrain_dataframe = train_dataframe.iloc[labeled]
    val_image_count = train_dataframe.shape[0]
    train_image_count = currlabelledtrain_dataframe.shape[0]

    # Training image augmentation
    train_data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),
        horizontal_flip=True,
        rotation_range=360,
        channel_shift_range=25,
        brightness_range=(0.75, 1.25),
    )

    # Validation image augmentation
    val_data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),
        horizontal_flip=True,
        rotation_range=360,
        channel_shift_range=25,
        brightness_range=(0.75, 1.25),
    )

    # Load train images in batches from directory and apply augmentations
    train_data_generator = train_data_generator.flow_from_dataframe(
        currlabelledtrain_dataframe,
        args["IMG_DIRECTORY"],
        x_col="Absolutefilename",
        y_col="Label",
        target_size=args["RAW_IMG_SIZE"],
        batch_size=args["BATCH_SIZE"],
        has_ext=True,
        classes=args["CLASSES_STR"],
        class_mode="categorical",
    )

    # Load validation images in batches from directory and apply rescaling
    val_data_generator = val_data_generator.flow_from_dataframe(
        val_dataframe,
        args["IMG_DIRECTORY"],
        x_col="Absolutefilename",
        y_col="Label",
        target_size=args["RAW_IMG_SIZE"],
        batch_size=args["BATCH_SIZE"],
        has_ext=True,
        classes=args["CLASSES_STR"],
        class_mode="categorical",
    )

    # Crop augmented images from 256x256 to 224x224
    train_data_generator = crop_generator(train_data_generator, args["IMG_SIZE"])
    val_data_generator = crop_generator(val_data_generator, args["IMG_SIZE"])

    # Load ImageNet pre-trained model with no top, either InceptionV3 or ResNet50
    if args["model_name"] == "resnet":
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=args["INPUT_SHAPE"]
        )
    elif args["model_name"] == "inception":
        base_model = InceptionV3(
            weights="imagenet", include_top=False, input_shape=args["INPUT_SHAPE"]
        )
    x = base_model.output
    # Add a global average pooling layer
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    # Add fully connected output layer with sigmoid activation for multi label classification
    outputs = Dense(len(args["CLASSES"]), activation="sigmoid", name="fc9")(x)
    # Assemble the modified model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Checkpoints for training
    model_checkpoint = ModelCheckpoint(
        output_directory + ckpt_file, verbose=1, save_best_only=True
    )
    early_stopping = EarlyStopping(
        patience=args["STOPPING_PATIENCE"], restore_best_weights=True
    )
    tensorboard = TensorBoard(
        log_dir=output_directory, histogram_freq=0, write_graph=True, write_images=False
    )
    reduce_lr = ReduceLROnPlateau(
        "val_loss", factor=0.5, patience=args["LR_PATIENCE"], min_lr=0.000003125
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=args["INITIAL_LR"]),
        metrics=["categorical_accuracy"],
    )
    csv_logger = CSVLogger(output_directory + "training_metrics.csv")

    # Train model until MAX_EPOCH, restarting after each early stop when learning has plateaued
    global_epoch = 0
    restarts = 0
    last_best_losses = []
    last_best_epochs = []
    while global_epoch < args["MAX_EPOCH"]:
        history = model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch=train_image_count // args["BATCH_SIZE"],
            epochs=args["MAX_EPOCH"] - global_epoch,
            validation_data=val_data_generator,
            validation_steps=val_image_count // args["BATCH_SIZE"],
            callbacks=[
                tensorboard,
                model_checkpoint,
                early_stopping,
                reduce_lr,
                csv_logger,
            ],
            shuffle=False,
        )
        last_best_losses.append(min(history.history["val_loss"]))
        last_best_local_epoch = history.history["val_loss"].index(
            min(history.history["val_loss"])
        )
        last_best_epochs.append(global_epoch + last_best_local_epoch)
        if early_stopping.stopped_epoch == 0:
            print("Completed training after {} epochs.".format(args["MAX_EPOCH"]))
            break
        else:
            global_epoch = (
                global_epoch
                + early_stopping.stopped_epoch
                - args["STOPPING_PATIENCE"]
                + 1
            )
            print(
                "Early stopping triggered after local epoch {} (global epoch {}).".format(
                    early_stopping.stopped_epoch, global_epoch
                )
            )
            print(
                "Restarting from last best val_loss at local epoch {} (global epoch {}).".format(
                    early_stopping.stopped_epoch - args["STOPPING_PATIENCE"],
                    global_epoch - args["STOPPING_PATIENCE"],
                )
            )
            restarts = restarts + 1
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(lr=args["INITIAL_LR"] / 2 ** restarts),
                metrics=["categorical_accuracy"],
            )
            model_checkpoint = ModelCheckpoint(
                output_directory + ckpt_file,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            )

    # Save last best model info
    with open(output_directory + "last_best_models.csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Model file", "Global epoch", "Validation loss"])
        for i in range(restarts + 1):
            writer.writerow([ckpt_file, last_best_epochs[i], last_best_losses[i]])

    return output_directory


def test(args, ckpt_file):
    print("Currently processing fold ", FOLD)
    output_directory = "{}{}/".format(args["OUTPUT_DIRECTORY"], FOLD)
    test_label_file = "{}test_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
    test_dataframe = pd.read_csv(test_label_file)
    test_dataframe["Absolutefilename"] = (
        args["IMG_DIRECTORY"] + test_dataframe["Filename"]
    )
    test_image_count = test_dataframe.shape[0]

    # No testing image augmentation (except for converting pixel values to floats)
    test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

    # Load test images in batches from directory and apply rescaling
    test_data_generator = test_data_generator.flow_from_dataframe(
        test_dataframe,
        args["IMG_DIRECTORY"],
        x_col="Absolutefilename",
        y_col="Label",
        target_size=args["IMG_SIZE"],
        batch_size=args["BATCH_SIZE"],
        has_ext=True,
        shuffle=False,
        classes=args["CLASSES_STR"],
        class_mode="categorical",
    )

    # Load the last best model
    model = load_model(output_directory + ckpt_file)
    # Evaluate model on test subset for kth fold
    predictions = model.predict_generator(
        test_data_generator, test_image_count // args["BATCH_SIZE"] + 1
    )
    y_true = test_data_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    y_pred[
        np.max(predictions, axis=1) < 1 / 9
    ] = 8  # Assign predictions worse than random guess to negative class

    return {"predictions": y_pred, "labels": y_true}


def infer(args, unlabeled, ckpt_file):
    # Load the last best model
    output_directory = "{}{}/".format(args["OUTPUT_DIRECTORY"], FOLD)
    model = load_model(output_directory + ckpt_file)
    train_label_file = "{}train_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
    train_dataframe = pd.read_csv(train_label_file)
    train_dataframe["Absolutefilename"] = (
        args["IMG_DIRECTORY"] + train_dataframe["Filename"]
    )
    currunlabelledtrain_dataframe = train_dataframe.iloc[unlabeled]

    # No testing image augmentation (except for converting pixel values to floats)
    unlabelled_data_generator = ImageDataGenerator(rescale=1.0 / 255)
    # Load test images in batches from directory and apply rescaling
    unlabelled_data_generator = unlabelled_data_generator.flow_from_dataframe(
        currunlabelledtrain_dataframe,
        args["IMG_DIRECTORY"],
        x_col="Absolutefilename",
        y_col="Label",
        target_size=args["IMG_SIZE"],
        batch_size=args["BATCH_SIZE"],
        has_ext=True,
        shuffle=False,
        classes=args["CLASSES_STR"],
        class_mode="categorical",
    )

    # Evaluate model on test subset for kth fold
    preds = model.predict_generator(
        unlabelled_data_generator, test_image_count // BATCH_SIZE + 1
    )

    predictions = {i: row for i, row in enumerate(preds)}

    return {"outputs": predictions}


def getdatasetstate(args, split="train"):
    if split == "train":
        train_label_file = "{}train_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
        dataframe = pd.read_csv(train_label_file)
        dataframe["Absolutefilename"] = args["IMG_DIRECTORY"] + dataframe["Filename"]
    else:
        test_label_file = "{}test_subset{}.csv".format(args["LABEL_DIRECTORY"], FOLD)
        dataframe = pd.read_csv(test_label_file)
        dataframe["Absolutefilename"] = args["IMG_DIRECTORY"] + dataframe["Filename"]

    pathdict = {}
    for ix, row in dataframe.iterrrows():
        pathdict[ix] = row["Absolutefilename"]

    return pathdict


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
