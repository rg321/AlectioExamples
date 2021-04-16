import argparse
import logging
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from sys import exit
from torch import optim
from tqdm import tqdm
from dice_loss import dice_coeff
from eval import eval_net
from unet import UNet
from FolderWithPaths import FolderWithPaths
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split, Subset
from alectio_sdk.sdk.sql_client import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, labeled, resume_from, ckpt_file):
    lr = args["INITIAL_LR"]
    img_scale = args["IMG_SCALE"]
    batch_size = args["BATCH_SIZE"]
    epochs = args["train_epochs"]

    traindataset = BasicDataset(
        args["TRAINIMAGEDATA_DIR"], args["TRAINLABEL_DIRECTORY"], img_scale
    )
    train = Subset(traindataset, labeled)
    n_train = len(train)
    valdataset = BasicDataset(
        args["VALIMAGEDATA_DIR"], args["VALLABEL_DIRECTORY"], img_scale
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        valdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    global_step = 0
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device=device)

    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if net.n_classes > 1 else "max", patience=2
    )
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(
        comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")

    predictions = {}
    predix = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]
                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                with torch.no_grad():
                    mask_pred = net(imgs)

                for ix, logit in enumerate(mask_pred):
                    predictions[predix] = logit.cpu().numpy()
                    predix += 1

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])

        if not epoch % args["SAVE_PER_EPOCH"]:
            try:
                os.mkdir(args["EXPT_DIR"])
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), args["EXPT_DIR"] + ckpt_file)
            logging.info(f"Checkpoint {epoch + 1} saved !")

    return {"predictions": predictions, "labels": true_masks.cpu().numpy()}


def test(args, ckpt_file):
    img_scale = args["IMG_SCALE"]
    batch_size = args["BATCH_SIZE"]
    testdataset = BasicDataset(
        args["TESTIMAGEDATA_DIR"], args["TESTLABEL_DIRECTORY"], img_scale
    )
    val_loader = DataLoader(
        testdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"] + ckpt_file)))
    net.eval()
    n_val = len(val_loader)
    test_count = 0
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in val_loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred_sig = torch.sigmoid(mask_pred)
            pred = (pred_sig > 0.5).float()

            pbar.update()

    true_masks = true_masks.cpu().numpy()
    pred = pred.cpu().numpy()

    return {"predictions": pred, "labels": true_masks}


def infer(args, unlabeled, ckpt_file):
    # create connection to local database
    database = os.path.join(
        args['EXPT_DIR'], 'infer_outputs_{}.db'.format(args['cur_loop']))
    conn = create_database(database)

    # Load the last best model
    img_scale = args["IMG_SCALE"]
    batch_size = args["BATCH_SIZE"]
    traindataset = BasicDataset(
        args["TRAINIMAGEDATA_DIR"], args["TRAINLABEL_DIRECTORY"], img_scale
    )
    unlableddataset = Subset(traindataset, unlabeled)
    unlabeled_loader = DataLoader(
        unlableddataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    predix = 0
    predictions = {}
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"] + ckpt_file)))
    net.eval()
    n_val = len(unlabeled_loader)
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    with tqdm(total=n_val, desc="Generating Inference Outputs", unit="batch", leave=False) as pbar:
        for batch in unlabeled_loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            for ix, logit in enumerate(mask_pred):
                logits = logit.data.cpu().numpy()
                add_index(conn, predix, logits)
                predix += 1
            pbar.update()

    # make sure to not return a value here, otherwise an incorrect output will occur


def getdatasetstate(args, split="train"):
    if split == "train":
        dataset = FolderWithPaths(args["TRAINIMAGEDATA_DIR"])
    else:
        dataset = FolderWithPaths(args["TESTIMAGEDATA_DIR"])

    dataset.transform = tv.transforms.Compose(
        [tv.transforms.RandomCrop(32), tv.transforms.ToTensor()]
    )
    trainpath = {}
    batchsize = 1
    loader = DataLoader(dataset, batch_size=batchsize,
                        num_workers=2, shuffle=False)
    for i, (_, _, paths) in enumerate(loader):
        for path in paths:
            if split in path:
                trainpath[i] = path
    return trainpath


if __name__ == "__main__":
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    labeled = list(range(10))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(args=args, labeled=labeled,
          resume_from=resume_from, ckpt_file=ckpt_file)
    test(args, ckpt_file=ckpt_file)
    infer(args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
