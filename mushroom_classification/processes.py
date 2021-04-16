from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from mushroom_data import MushroomDataset
from model import NeuralNet
import torch.optim as optim
import os
import yaml
import argparse
import torch.nn.functional as F
from scipy.special import logit as logit_fn
from scipy.special import expit

device = "cuda" if torch.cuda.is_available() else "cpu"


def getdatasetstate(args={}):
    return {k: k for k in range(args["train_size"])}


def train(args, labeled, resume_from, ckpt_file):
    print("========== In the train step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]

    train_split = args["split_train"]

    CSV_FILE = "./data/mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_dataset = torch.utils.data.Subset(
        dataset, list(range(int(train_split * len(dataset))))
    )

    train_subset = Subset(train_dataset, labeled)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    net = NeuralNet()
    net = net.to(device=device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=float(lr), momentum=momentum)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from + ".pth"))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    net.train()

    for epoch in tqdm(range(args["train_epochs"]), desc="Training"):

        running_loss = 0

        for i, batch in enumerate(train_loader, start=0):
            data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000:
                print(
                    "epoch: {} batch: {} running-loss: {}".format(
                        epoch + 1, i + 1, running_loss / 1000
                    ),
                    end="\r",
                )
                running_loss = 0

    print("Finished Training. Saving the model as {}".format(ckpt_file))

    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))

    return


def test(args, ckpt_file):
    print("========== In the test step ==========")
    batch_size = args["batch_size"]

    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    CSV_FILE = "./data/mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    test_dataset = torch.utils.data.Subset(
        dataset, list(range(int(train_split * len(dataset)), len(dataset)))
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    net = NeuralNet()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()
    predix = 0
    predictions = {}
    truelabels = {}

    n_val = len(test_dataset)
    with tqdm(total=n_val, desc="Testing round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                prediction = net(batch_x)

            for logit, label in zip(prediction, batch_y):
                predictions[predix] = logit.cpu().numpy().tolist()
                truelabels[predix] = label.cpu().numpy().tolist()
                predix += 1

            pbar.update()

    truelabels_ = []
    predictions_ = []

    for key in predictions:
        if predictions[key][0] > 0.5:
            predictions_.append(1)
        else:
            predictions_.append(0)

    for key in truelabels:
        truelabels_.append(truelabels[key][0])

    truelabels = truelabels_
    predictions = predictions_

    # print("predictions",predictions)

    return {"predictions": predictions, "labels": truelabels}


def infer(args, unlabeled, ckpt_file):

    print("========== In the inference step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    CSV_FILE = "./data/mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_dataset = torch.utils.data.Subset(
        dataset, list(range(int(train_split * len(dataset))))
    )

    train_subset = Subset(train_dataset, unlabeled)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    net = NeuralNet()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()

    n_val = len(unlabeled)
    predictions = {}
    predix = 0

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = net(batch_x)

            for logit in prediction:
                predictions[unlabeled[predix]] = {}
                prediction = 0

                # since I include sigmoid as the last layer's activation function, logit = %
                if logit.cpu().numpy() > 0.5:
                    prediction = 1
                predictions[unlabeled[predix]]["prediction"] = prediction

                predictions[unlabeled[predix]]["logits"] = [
                    [
                        logit_fn(logit.cpu().numpy()[0]),
                        logit_fn(1 - logit.cpu().numpy()[0]),
                    ]
                ]

                predix += 1

            pbar.update()

    # print("predictions",predictions)

    return {"outputs": predictions}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.getcwd(), "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)

    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    print("Testing getdatasetstate")
    getdatasetstate(args=args)
    train(args=args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(args=args, ckpt_file=ckpt_file)
    print(infer(args=args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file))
