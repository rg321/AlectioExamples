import os
import sys
import torch
import pickle
import warnings
import traceback
from model import *
from tqdm import tqdm
import torchvision as tv
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageFile
from datasets import *
from alectio_sdk.torch_utils.utils import non_max_suppression, bbox_iou
from alectio_sdk.sdk.sql_client import *
from FolderWithPaths import FolderWithPaths

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

image_width, image_height = 416, 416

########## Build your own train function like below ###############################################


def train(args, labeled, resume_from, ckpt_file):
    """
    Train function to train on the target data
    """
    ########Set reproduceablility

    seed = int(42)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # hyperparameters
    config_file = "yolov3.cfg"
    epochs, batch_size = args["train_epochs"], args["batch_size"]
    accumulated_batches = args["accumulated_batches"]
    best_mAP = args["best_mAP"]
    # Get hyper parameters
    hyperparams = parse_model_config(config_file)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    backbone_freeze = 1
    ckpt_save_interval = args["ckpt_save_interval"]
    datamap = getdatasetstate(
        args, split="train",
    )  ##### Since our dataset object accepts list of imagenames we are using the state function again
    imglist = [v for k, v in datamap.items() if k in labeled]
    trainDataset = ListDataset(imglist)
    trainDataloader = torch.utils.data.DataLoader(
        trainDataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = Darknet(config_file).to(device)
    model.load_weights(os.path.join(args["WEIGHTS_DIR"], "darknet53.conv.74"))
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # resume model and optimizer from previous loop
    if resume_from is not None and not args["weights_clear"]:
        model.load_weights(os.path.join(args["EXPT_DIR"], resume_from))

    for epoch in tqdm(range(epochs), desc="Training"):

        losses_x = (
            losses_y
        ) = (
            losses_w
        ) = (
            losses_h
        ) = (
            losses_conf
        ) = losses_cls = losses_recall = losses_precision = batch_loss = 0.0

        if backbone_freeze:
            if epoch < 30:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split(".")[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch >= 30:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split(".")[1]) < 75:  # if layer < 75
                        p.requires_grad = True

        optimizer.zero_grad()

        for n_batch, (_, imgs, targets) in enumerate(trainDataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            loss = model(imgs, targets)
            loss.backward()
            if ((n_batch + 1) % accumulated_batches == 0) or (
                n_batch == len(trainDataloader) - 1
            ):
                optimizer.step()
                optimizer.zero_grad()
            losses_x += model.losses["x"]
            losses_y += model.losses["y"]
            losses_w += model.losses["w"]
            losses_h += model.losses["h"]
            losses_conf += model.losses["conf"]
            losses_cls += model.losses["cls"]
            losses_recall += model.losses["recall"]
            losses_precision += model.losses["precision"]
            batch_loss += loss.item()
            loss_data = "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    epochs,
                    n_batch,
                    len(trainDataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )
            model.seen += imgs.size(0)

        if epoch % ckpt_save_interval == 0:
            save_fp = os.path.join(args['EXPT_DIR'], ckpt_file)
            print('Saving weights as ' + save_fp)
            model.save_weights(save_fp)

    return


def test(args, ckpt_file):
    """
    Test your model on the test set
    Note : The compute metrics part is implemented in pipeline.py for a set of
    object detection metrics, edit and include your own metrics if necessary
    and rebuild the app
    """
    batch_size = args["batch_size"]
    datamap = getdatasetstate(args, split="val")
    # WARNING HARDCODED VALUE at 100
    testDataset = ListDataset(list(datamap.values())[:100])
    testDataloader = torch.utils.data.DataLoader(
        testDataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    config_file = "yolov3.cfg"
    print("Loading trained model to perform Test task")
    model = Darknet(config_file).to(device)
    model.load_weights(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.eval()
    predix = 0
    predictions = {}
    labels = {}
    for n_batch, (_, imgs, targets) in enumerate(
        tqdm(testDataloader, desc="Running Inference on Test set")
    ):
        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            outputs, _ = model(imgs)
            outputs = non_max_suppression(outputs, 80, conf_thres=0.5, nms_thres=0.4)

        for preds, target in zip(outputs, targets):
            if preds is not None:
                predictions[predix] = {
                    "boxes": preds[:, :4].cpu().numpy().tolist(),
                    "objects": preds[:, -1].cpu().numpy().tolist(),
                    "scores": preds[:, 4].cpu().numpy().tolist(),
                }
                if any(target[:, -1] > 0):
                    rawboxes = target[target[:, -1] > 0, 1:]
                    converted_boxes = np.empty_like(rawboxes)
                    converted_boxes[:, 0] = rawboxes[:, 0] - rawboxes[:, 2] / 2
                    converted_boxes[:, 1] = rawboxes[:, 1] - rawboxes[:, 3] / 2
                    converted_boxes[:, 2] = rawboxes[:, 0] + rawboxes[:, 2] / 2
                    converted_boxes[:, 3] = rawboxes[:, 1] + rawboxes[:, 3] / 2

                    converted_boxes *= image_height

                    labels[predix] = {
                        "boxes": converted_boxes.tolist(),
                        "objects": target[target[:, -1] > 0, 0].cpu().numpy().tolist(),
                    }
                else:
                    labels[predix] = {"boxes": [], "objects": []}

                predix += 1
            else:
                predictions[predix] = {"boxes": [], "objects": [], "scores": []}

                if any(target[:, -1] > 0):
                    rawboxes = target[target[:, -1] > 0, 1:]
                    converted_boxes = np.empty_like(rawboxes)
                    converted_boxes[:, 0] = rawboxes[:, 0] - rawboxes[:, 2] / 2
                    converted_boxes[:, 1] = rawboxes[:, 1] - rawboxes[:, 3] / 2
                    converted_boxes[:, 2] = rawboxes[:, 0] + rawboxes[:, 2] / 2
                    converted_boxes[:, 3] = rawboxes[:, 1] + rawboxes[:, 3] / 2
                    labels[predix] = {
                        "boxes": converted_boxes.tolist(),
                        "objects": target[target[:, -1] > 0, 0].cpu().numpy().tolist(),
                    }
                else:
                    labels[predix] = {"boxes": [], "objects": []}
                predix += 1

    return {"predictions": predictions, "labels": labels}


def infer(args, unlabeled, ckpt_file):
    """
    Infer function to infer on the unlabelled data
    """
    # SQL operations, create connection
    database = os.path.join(args['EXPT_DIR'], 'infer_outputs_{}.db'.format(args['cur_loop']))
    conn = create_database(database)


    batch_size = args["batch_size"]
    datamap = getdatasetstate(
        args, split="train"
    )  ##### Since our dataset object accepts list of imagenames we are using the state function again
    unlabelledmap = [v for k, v in datamap.items() if k in unlabeled]
    testDataset = ListDataset(unlabelledmap)
    testDataloader = torch.utils.data.DataLoader(
        testDataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    config_file = "yolov3.cfg"
    print("Loading trained model to perform Infer task")
    model = Darknet(config_file).to(device)
    model.load_weights(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.eval()
    for n_batch, (_, imgs, targets) in enumerate(
        tqdm(testDataloader, desc="Running Inference on Unlabelled pool")
    ):
        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            _, presig = model(imgs)
            formatboxes = presig.new(presig.shape)
            formatboxes[:, :, 0] = presig[:, :, 0] - presig[:, :, 2] / 2
            formatboxes[:, :, 1] = presig[:, :, 1] - presig[:, :, 3] / 2
            formatboxes[:, :, 2] = presig[:, :, 0] + presig[:, :, 2] / 2
            formatboxes[:, :, 3] = presig[:, :, 1] + presig[:, :, 3] / 2
            presig[:, :, :4] = formatboxes[:, :, :4]

            for i, logit in enumerate(presig):
                true_mask = (logit[:, 4] >= 0.5).squeeze()
                logit = logit[true_mask]
                logits_np = logit[:, 5:].cpu().numpy()
                added_ind = unlabeled.pop(0)
                # add row to db
                add_index(conn, added_ind, logits_np)

    conn.close()

def getdatasetstate(args, split="train"):
    if split == "train":
        dataset = FolderWithPaths(args["TRAIN_DATA_DIR"])
    else:
        dataset = FolderWithPaths(args["TEST_DATA_DIR"])

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
    # debug
    # train
    """
    labeled = list(range(300))
    resume_from = None
    ckpt_file = "ckpt_0"
    logdir = "test"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Testing")
    test(ckpt_file)
    """
