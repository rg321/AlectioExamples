import os
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from functools import partial
from typing import Tuple, List
from models import BertClassifier
from dataset import ToxicDataset, collate_fn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
total = pd.read_csv('../data/train.csv').fillna(' ')

train_df, validate_df, test_df = np.split(total.sample(frac=1, random_state=42), [int(.7*len(total)), int(.8*len(total))])

print("Dataset sizes")
print("Train: {}; Validate: {}; Test: {}".format(len(train_df), len(validate_df), len(test_df)))

bert_model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

collate_fn = partial(collate_fn, device=device)
assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"

def setpytorchreproduceability(seed = 42):
    """
        Function enables reproduceablity for your experiments so that you can compare easily
        Parameters:
        seed (int): seed value for global randomness
    """

    logging.warning("*** Modules are set to be deterministic , randomness in your modules will be avoided , Current seed value is {} change seed value if you want to try a different seed of parameters***".format(seed))
    #Pythonic determinism
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    #Pythonic determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getdatasetstate(args):
    pass

def train(args, labeled, resume_from, ckpt_file):
    train_subset = train_df.iloc[labeled]

    setpytorchreproduceability(seed= 42)

    print("Training with {} records".format(len(train_subset)))
    train_dataset = ToxicDataset(tokenizer, train_subset, lazy=True)
    dev_dataset = ToxicDataset(tokenizer, validate_df, lazy=True)

    train_iterator = DataLoader(train_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, num_workers=0)   
    dev_iterator = DataLoader(dev_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, num_workers=0) 

    no_decay = ['bias', 'LayerNorm.weight']
    model = BertClassifier(BertModel.from_pretrained(bert_model_name), 6).to(device)
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not \
                                     any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, \
                                     {'params': [p for n, p in model.named_parameters() if \
                                     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    total_steps = len(train_iterator) * args["epochs"] - args["warmup_steps"]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(args["lr"]), eps=float(args["eps"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, args["warmup_steps"], total_steps)
    loss_fn = nn.BCELoss()

    for ep in range(args["epochs"]):

        # TRAINING

        model.train()
        total_loss = 0
        for x, y in tqdm(train_iterator, desc="Training"):
            optimizer.zero_grad()
            
            mask = (x != 0).float()
            logits = model(x, attention_mask=mask)
            loss = loss_fn(torch.sigmoid(logits), y)

            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

        print(f"Train loss ep {ep + 1} is {total_loss / len(train_iterator)}")

        # VALIDATION

        model.eval()
        pred, true = [], []

        with torch.no_grad():
            total_loss = 0
            for x, y in tqdm(dev_iterator, desc="Validating"):
                mask = (x != 0).float()
                preds = torch.sigmoid(model(x, attention_mask=mask))

                total_loss += loss.item()
                true += y.cpu().numpy().tolist()
                pred += preds.cpu().numpy().tolist()

        true = np.array(true)
        pred = np.array(pred)

        for i, name in enumerate(class_names):
            print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
            print(f"{name} accuracy {accuracy_score(true[:, i], 1 * (pred[:, i] > 0.5))}")
        
        print(f"Validation loss {total_loss / len(dev_iterator)}")

    torch.save(model.state_dict(), os.path.join(args["EXPT_DIR"], ckpt_file))

def test(args, ckpt_file):
    model = BertClassifier(BertModel.from_pretrained(bert_model_name), 6).to(device)
    model.load_state_dict(torch.load(open(os.path.join(args["EXPT_DIR"], ckpt_file), "rb")))

    test_dataset = ToxicDataset(tokenizer, test_df, lazy=True)
    test_iterator = DataLoader(test_dataset, batch_size=args["batch_size"], collate_fn=collate_fn) 

    model.eval()
    pred, true = [], []

    with torch.no_grad():
        for x, y in tqdm(test_iterator, desc="Testing"):
            mask = (x != 0).float()
            preds = torch.sigmoid(model(x, attention_mask=mask))

            for i in range(len(class_names)):
                preds[:, i] = 1 * (preds[:, i] > 0.5)

            true += y.cpu().numpy().tolist()
            pred += preds.cpu().numpy().tolist()

    true = np.array(true)
    pred = np.array(pred)
    
    return {'predictions': pred, 'labels': true}

def infer(args, unlabeled, ckpt_file):
    infer_subset = train_df.iloc[unlabeled]
    infer_dataset = ToxicDataset(tokenizer, infer_subset, lazy=True)
    infer_iterator = DataLoader(infer_dataset, batch_size=args["batch_size"], collate_fn=collate_fn)   

    model = BertClassifier(BertModel.from_pretrained(bert_model_name), 6).to(device)
    model.load_state_dict(torch.load(open(os.path.join(args["EXPT_DIR"], ckpt_file), "rb")))
    
    infer_op = []

    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(infer_iterator, desc="Inferring"):
            mask = (x != 0).float()
            logits = model(x, attention_mask=mask)
            infer_op += logits.cpu().numpy().tolist()

    infer_preds = {k: {'logits': val} for k, val in enumerate(infer_op)}

    return {"outputs": infer_preds}

if __name__ == "__main__":
    # Do your testing here.
    args = {"EXPT_DIR": '.', 'batch_size': 32, 'epochs': 2, 'warmup_steps': 1000, 'lr': 2e-5, 'eps': 1e-8}
    labeled = list(range(5000))
    resume_from = None
    ckpt_file = 'ckpt_0'
    unlabeled = list(range(10000, 15000))

    train(args, labeled, resume_from, ckpt_file)
    test(args, ckpt_file)
    infer(args, unlabeled, ckpt_file)