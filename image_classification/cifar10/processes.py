import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import resnet

device = "cuda" if torch.cuda.is_available() else "cpu"

image_width, image_height = 32, 32
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]
)


def getdatasetstate(args={}):
    return {k: k for k in range(50000)}


def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = args["lr"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    wtd = args["wtd"]

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainset = Subset(trainset, labeled)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    predictions, targets = [], []
    net = resnet.__dict__["resnet20"]().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wtd)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90, 95], last_epoch=-1
    )

    if resume_from is not None and not args["weightsclear"]:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    net.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        lr_scheduler.step()

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return {"predictions": predictions, "labels": targets}


def test(args, ckpt_file):
    batch_size = 16
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    predictions, targets = [], []
    net = resnet.__dict__["resnet20"]().to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_test
    )
    unlabeled = Subset(trainset, unlabeled)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled, batch_size=4, shuffle=False, num_workers=2
    )

    net = resnet.__dict__["resnet20"]().to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total, k = 0, 0, 0
    outputs_fin = {}
    for i, data in tqdm(enumerate(unlabeled_loader), desc="Inferring"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images).data

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(len(outputs)):
            outputs_fin[k] = {}
            outputs_fin[k]["prediction"] = predicted[j].item()
            outputs_fin[k]["pre_softmax"] = outputs[j].cpu().numpy().tolist()
            k += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
