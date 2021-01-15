import argparse
import time
from sys import platform
import os

from tqdm import tqdm
from models import *
from utils.datasets import *
from utils.utils import *
from alectio_sdk.torch_utils.utils import non_max_suppression, bbox_iou

import warnings
warnings.filterwarnings("ignore")

# This looks good!
def getdatasetstate(args, split="train"):
    files = {k:v for k,v in enumerate(sorted(glob.glob('%s/*.tif' % "train_images")))}
    return files

def train(args, labeled, resume_from, ckpt_file):
    """
    Train function to train on the target data
    """
    targets_path = 'utils/targets_c60.mat'
    os.makedirs('weights', exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True

    train_path = './data/train_images'

    # Initialize model
    import models
    model = models.Darknet(args['cfg'], args['img_size'])
    model.load_weights(os.path.join(args["WEIGHTS_DIR"], "darknet53.conv.74"))

    if (resume_from is not None) and (args['weightsclear'] is False):
        print("Loading old model")
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
        model.load_state_dict(ckpt["model"])

    model.train()
    # Get dataloader
    print("LABELED SIZE: {}".format(len(labeled)))
    dataloader = ListDataset(train_path, labeled, batch_size=args['batch_size'], img_size=args['img_size'], targets_path=targets_path)

    # reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')

    if torch.cuda.device_count() > 1:
        print('Using ', torch.cuda.device_count(), ' GPUs')
        model = nn.DataParallel(model)

    model.to(device).train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-4)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=0.00001, last_epoch=-1)
    # y = 0.001 * exp(-0.00921 * x)  # 1e-4 @ 250, 1e-5 @ 500
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99082, last_epoch=start_epoch - 1)

    modelinfo(model)
    t0, t1 = time.time(), time.time()
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))

    class_weights = xview_class_weights_hard_mining(range(60)).to(device)

    for epoch in range(args['epochs']):
        epoch += start_epoch
        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(4, 60)
        for i, (imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, w, h, conf, cls, lossyy, p, r, ngt, tp, fp, fn, tim = [], [], [], [], [], [], [], [], [], [], [], [], [], []
            n = len(imgs)  # number of pictures at a time
            for j in range(int(len(imgs) / n)):
                targets_j = targets[j * n:j * n + n]
                nGT = sum([len(x) for x in targets_j])
                if nGT < 1:
                    continue

                loss = model(imgs[j * n:j * n + n].to(device), targets_j, requestPrecision=True, weight=class_weights, epoch=epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ui += 1
                metrics += model.losses['metrics']
                for key, val in model.losses.items():
                    rloss[key] = (rloss[key] * ui + val) / (ui + 1)

                # Precision
                precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
                k = (metrics[0] + metrics[1]) > 0
                if k.sum() > 0:
                    mean_precision = precision[k].mean()
                else:
                    mean_precision = 0

                # Recall
                recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
                k = (metrics[0] + metrics[2]) > 0
                if k.sum() > 0:
                    mean_recall = recall[k].mean()
                else:
                    mean_recall = 0

                x.append(rloss['x'])
                y.append(rloss['y'])
                w.append(rloss['w'])
                h.append(rloss['h'])
                conf.append(rloss['conf'])
                cls.append(rloss['cls'])
                lossyy.append(rloss['loss'])
                p.append(mean_precision)
                r.append(mean_recall)
                ngt.append(model.losses['nGT'])
                tp.append(model.losses['TP'])
                fp.append(model.losses['FP'])
                fn.append(model.losses['FN'])
                tim.append(time.time() - t1)

                t1 = time.time()

        s = ('%10s%10s' + '%10.3g' * 14) % (
            '%g/%g' % (epoch, args['epochs'] - 1), '%g/%g' % (i, len(dataloader) - 1), np.mean(x),
            np.mean(y), np.mean(w), np.mean(h), np.mean(conf), np.mean(cls),
            np.mean(lossyy), np.mean(p), np.mean(r), np.mean(ngt), np.mean(tp),
            np.mean(fp), np.mean(fn), np.mean(tim))

        print(s)
        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss'] / (rloss['nGT'])
        if loss_per_target < best_loss:
            best_loss = loss_per_target
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))
            
    # model.save_weights("%s/%s" % (args["LOG_DIR"], ckpt_file))

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))

def test(args, ckpt_file):
    targets_path = 'utils/targets_c60.mat'
    os.makedirs('weights', exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # Configure run
    if platform == 'darwin':  # MacOS (local)
        train_path = '/Users/glennjocher/Downloads/DATA/xview/train_images'
    else: # linux (GCP cloud)
        train_path = 'train_images'

    # Initialize model
    import models
    model = models.Darknet(args['cfg'], args['img_size'])
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Wts loaded.")

    # Get dataloader
    dataloader = ListDataset(train_path, list(range(700, 846)), batch_size=1, img_size=args['img_size'], targets_path=targets_path)
    print("Dataloader created.")

    # model.load_weights(os.path.join(args["LOG_DIR"], ckpt_file))
    predix = 0
    predictions = {}
    labels = {}

    for i, (imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            outputs, _ = model(imgs)
            outputs = non_max_suppression(outputs, 60, conf_thres=0.9, nms_thres=0.4) # 60 number of classes

            for preds, target in zip(outputs, targets):
                if preds is not None:
                    predictions[predix] = {
                        "boxes": preds[:, :4].cpu().numpy().tolist(),
                        "objects": preds[:, -1].cpu().numpy().tolist(),
                        "scores": preds[:, 4].cpu().numpy().tolist(),
                    }
                    target = np.array(target)

                    if len(target.shape) > 1 and any(target[:, -1] > 0):
                        rawboxes = target[target[:, -1] > 0, 1:]
                        converted_boxes = np.empty_like(rawboxes)
                        converted_boxes[:, 0] = rawboxes[:, 0] - rawboxes[:, 2] / 2
                        converted_boxes[:, 1] = rawboxes[:, 1] - rawboxes[:, 3] / 2
                        converted_boxes[:, 2] = rawboxes[:, 0] + rawboxes[:, 2] / 2
                        converted_boxes[:, 3] = rawboxes[:, 1] + rawboxes[:, 3] / 2

                        converted_boxes *= args['img_size']

                        labels[predix] = {
                            "boxes": converted_boxes.tolist(),
                            "objects": target[target[:, -1] > 0, 0].tolist(),
                        }

                    else:
                        labels[predix] = {"boxes": [], "objects": []}

                    predix += 1

                else:
                    predictions[predix] = {"boxes": [], "objects": [], "scores": []}
                    target = np.array(target)
                    if len(target.shape) > 1 and any(target[:, -1] > 0):
                        rawboxes = target[target[:, -1] > 0, 1:]
                        converted_boxes = np.empty_like(rawboxes)
                        converted_boxes[:, 0] = rawboxes[:, 0] - rawboxes[:, 2] / 2
                        converted_boxes[:, 1] = rawboxes[:, 1] - rawboxes[:, 3] / 2
                        converted_boxes[:, 2] = rawboxes[:, 0] + rawboxes[:, 2] / 2
                        converted_boxes[:, 3] = rawboxes[:, 1] + rawboxes[:, 3] / 2

                        labels[predix] = {
                            "boxes": converted_boxes.tolist(),
                            "objects": target[target[:, -1] > 0, 0].tolist(),
                        }
                    else:
                        labels[predix] = {"boxes": [], "objects": []}
                    predix += 1

    return {"predictions": predictions, "labels": labels}

def infer(args, unlabeled, ckpt_file):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    targets_path = 'utils/targets_c60.mat'

    # Configure run
    if platform == 'darwin':  # MacOS (local)
        train_path = '/Users/glennjocher/Downloads/DATA/xview/train_images'
    else: # linux (GCP cloud)
        train_path = 'train_images'

    # Initialize model
    import models
    model = models.Darknet(args['cfg'], args['img_size'])
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Wts loaded.")

    # Get dataloader
    print("UNLABELED SIZE: {}".format(len(unlabeled)))
    dataloader = ListDataset(train_path, unlabeled, batch_size=1, img_size=args['img_size'], targets_path=targets_path)
    print("Dataloader created.")

    #model.load_weights(os.path.join(args["LOG_DIR"], ckpt_file))

    predix = 0
    predictions = {}
    labels = {}

    for i, (imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        n = len(imgs)
        with torch.no_grad():
            for j in range(int(len(imgs) / n)):
                _, presig = model(imgs[j * n: j * n + n])
                formatboxes = presig.new(presig.shape)
                formatboxes[:, :, 0] = presig[:, :, 0] - presig[:, :, 2] / 2
                formatboxes[:, :, 1] = presig[:, :, 1] - presig[:, :, 3] / 2
                formatboxes[:, :, 2] = presig[:, :, 0] + presig[:, :, 2] / 2
                formatboxes[:, :, 3] = presig[:, :, 1] + presig[:, :, 3] / 2
                presig[:, :, :4] = formatboxes[:, :, :4]

                for i, logit in enumerate(presig):
                    true_mask = (logit[:, 4] >= 0.9).squeeze()
                    logit = logit[true_mask]
                    predictions[predix] = {
                        "boxes": logit[:, :4].cpu().numpy().tolist(),
                        "pre_softmax": logit[:, 5:].cpu().numpy().tolist(),
                        "scores": logit[:, 4].cpu().numpy().tolist(),
                    }
                    predix += 1

    return {"outputs": predictions}

# FOR TESTING
def compute_metrics(predictions, ground_truth):
    from alectio_sdk.metrics.object_detection import Metrics, batch_to_numpy
    metrics = {}

    det_boxes, det_labels, det_scores, true_boxes, true_labels = batch_to_numpy(
        predictions, ground_truth
    )

    m = Metrics(
        det_boxes=det_boxes,
        det_labels=det_labels,
        det_scores=det_scores,
        true_boxes=true_boxes,
        true_labels=true_labels,
        num_classes=60,
    )

    metrics = {
        "mAP": m.getmAP(),
        "AP": m.getAP(),
        "precision": m.getprecision(),
        "recall": m.getrecall(),
        "confusion_matrix": m.getCM().tolist(),
        "class_labels": None,
    }

    print("========= TEST METRICS =========")
    print(metrics)

if __name__ == "__main__":
    labeled = list(range(700))
    unlabeled = list(range(100, 200))

    import yaml
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    resume_from = 1
    ckpt_file = "ckpt_0"

    print("Training")
    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Testing")
    output_test = test(args, ckpt_file)
    compute_metrics(output_test['predictions'], output_test['labels'])
    # os.system("sudo shutdown -h now")
    # print("Inferring")
    # infer(args, unlabeled, ckpt_file)
