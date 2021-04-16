import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import utils

if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
    print("Using GPU")
else:
    print("Using CPU")

print("Loading dataset ...")
dataset, info = tfds.load(
    "yelp_polarity_reviews/subwords8k",
    data_dir="./data",
    as_supervised=True,
    with_info=True,
)

train_dset, test_dset = (
    list(dataset["train"].as_numpy_iterator()),
    list(dataset["test"].as_numpy_iterator()),
)


encoder = info.features["text"].encoder


def getdatasetstate(args={}):
    return {k: k for k in range(len(train_dset))}


def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    epochs = args["train_epochs"]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(encoder.vocab_size, 200),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    labeled_train_data = [train_dset[i] for i in labeled]
    labeled_train_dset = tf.data.Dataset.from_generator(
        lambda: labeled_train_data,
        (tf.float32, tf.float32),
        (tf.TensorShape([None]), tf.TensorShape([])),
    ).padded_batch(batch_size)

    if resume_from is not None:
        model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], resume_from))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    model.fit(labeled_train_dset, epochs=epochs)

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    tf.keras.models.save_model(
        model, os.path.join(args["EXPT_DIR"], ckpt_file), save_format="h5"
    )

    return


def test(args, ckpt_file):

    batch_size = args["batch_size"]
    labeled_test_dset = tf.data.Dataset.from_generator(
        lambda: test_dset,
        (tf.float32, tf.float32),
        (tf.TensorShape([None]), tf.TensorShape([])),
    ).padded_batch(batch_size)

    model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    predictions, targets = [], []
    for data, labels in tqdm(labeled_test_dset, desc="Testing"):
        pred = model(data).numpy().flatten()
        pred[pred <= 0] = 0
        pred[pred > 0] = 1
        predictions.extend(pred.tolist())
        targets.extend(labels.numpy().tolist())

    print(
        "Testing Accuracy : {}".format(
            1
            - (
                np.sum(np.abs(np.array(predictions) - np.array(targets)))
                / len(predictions)
            )
        )
    )

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    batch_size = args["batch_size"]

    unlabeled_train_data = [train_dset[i] for i in unlabeled]
    unlabeled_train_dset = tf.data.Dataset.from_generator(
        lambda: unlabeled_train_data,
        (tf.float32, tf.float32),
        (tf.TensorShape([None]), tf.TensorShape([])),
    ).padded_batch(batch_size)

    model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], ckpt_file))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    outputs_fin = {}
    i = 0
    for data, labels in tqdm(unlabeled_train_dset, desc="Inferring"):
        outputs = model(data).numpy().flatten()
        pred = np.copy(outputs)
        pred[pred <= 0] = 0
        pred[pred > 0] = 1

        for j in range(len(pred)):
            outputs_fin[i] = {}
            outputs_fin[i]["prediction"] = pred[j]
            outputs_fin[i]["logits"] = [
                utils.logit(1 - tf.math.sigmoid(outputs[j])).numpy(),
                outputs[j],
            ]
            i += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    # for debugging purposes
    labeled = list(range(100))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=labeled, ckpt_file=ckpt_file)
