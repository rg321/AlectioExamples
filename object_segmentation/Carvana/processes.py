# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:22 2020

@author: arun
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dice_loss import dice_coeff

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split, Subset


##### Global variables


def train(args, labeled, resume_from, ckpt_file):
    lr = args["INITIAL_LR"]
    img_scale = args["IMG_SCALE"]
    batch_size = args["BATCH_SIZE"]
    epochs = args["train_epochs"]
    traindataset = BasicDataset(
        args["TRAINIMAGEDATA_DIR"], args["TRAINLABEL_DIRECTORY"], img_scale
    )
    train = Subset(traindataset, labeled)
    valdataset = BasicDataset(
        args["VALIMAGEDATA_DIR"], args["VALLABEL_DIRECTORY"], img_scale
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device=device)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if net.n_classes > 1 else "max", patience=2
    )
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
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
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), global_step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                        )
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        "learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )

                    if net.n_classes > 1:
                        logging.info("Validation cross entropy: {}".format(val_score))
                        writer.add_scalar("Loss/test", val_score, global_step)
                    else:
                        logging.info("Validation Dice Coeff: {}".format(val_score))
                        writer.add_scalar("Dice/test", val_score, global_step)

                    writer.add_images("images", imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images("masks/true", true_masks, global_step)
                        writer.add_images(
                            "masks/pred", torch.sigmoid(masks_pred) > 0.5, global_step
                        )

        if not epoch % args["SAVE_PER_EPOCH"]:
            try:
                os.mkdir(args["EXPT_DIR"])
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), args["EXPT_DIR"] + ckpt_file)
            logging.info(f"Checkpoint {epoch + 1} saved !")

    return args["EXPT_DIR"]


def test(args, ckpt_file):
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
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in val_loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred_sig = torch.sigmoid(mask_pred)
                pred = (pred_sig > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    return {"predictions": pred, "labels": true_masks}


def infer(args, unlabeled, ckpt_file):
    # Load the last best model
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

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in val_loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            for ix, logit in enumerate(maskpred):
                predictions[predix] = logit.cpu().numpy()

                predix += 1

            pbar.update()

    return {"outputs": predictions}


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
    loader = DataLoader(dataset, batch_size=batchsize, num_workers=2, shuffle=False)
    for i, (_, _, paths) in enumerate(loader):
        for path in paths:
            if split in path:
                trainpath[i] = path
    return trainpath


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
