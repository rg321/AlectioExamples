import os
import io
import torch
import logging
import torchtext
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import text_classification
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import Vocab

from tqdm import tqdm
from model import TextSentiment

device = "cuda" if torch.cuda.is_available() else "cpu"


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.
        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit="lines") as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(
                    filter(
                        lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]
                    )
                )
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info("Row contains no tokens.")
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = " ".join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _setup_datasets(
    dataset_name, root=".data", ngrams=1, vocab=None, include_unk=False
):
    root = os.path.join(root, "ag_news_csv")
    extracted_files = os.listdir(root)

    for fname in extracted_files:
        if fname.endswith("train.csv"):
            train_csv_path = os.path.join(root, fname)
        if fname.endswith("test.csv"):
            test_csv_path = os.path.join(root, fname)

    if vocab is None:
        logging.info("Building Vocab based on {}".format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info("Vocab has {} entries".format(len(vocab)))
    logging.info("Creating training data")
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk
    )
    logging.info("Creating testing data")
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk
    )
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (
        TextClassificationDataset(vocab, train_data, train_labels),
        TextClassificationDataset(vocab, test_data, test_labels),
    )


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 0 : World
            - 1 : Sports
            - 2 : Business
            - 3 : Sci/Tech
    Create supervised learning dataset: AG_NEWS
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def getdatasetstate(args={}):
    return {k: k for k in range(120000)}


def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = 4.0
    momentum = 0.9
    epochs = args["train_epochs"]

    global train_dataset, test_dataset
    train_dataset, test_dataset = AG_NEWS(
        root="./data", ngrams=args["N_GRAMS"], vocab=None
    )

    global VOCAB_SIZE, EMBED_DIM, NUN_CLASS
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = args["EMBED_DIM"]
    NUN_CLASS = len(train_dataset.get_labels())

    chosen_train_dataset = Subset(train_dataset, labeled)
    trainloader = DataLoader(
        chosen_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generate_batch,
    )
    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    net.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        running_loss = 0.0
        train_acc = 0
        for i, data in enumerate(trainloader):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)
            loss = criterion(outputs, cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += (outputs.argmax(1) == cls).sum().item()
            running_loss += loss.item()
        scheduler.step()

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    print("Training accuracy: {}".format((train_acc / len(chosen_train_dataset) * 100)))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return


def test(args, ckpt_file):
    batch_size = args["batch_size"]
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=generate_batch
    )

    predictions, targets = [], []
    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(cls.cpu().numpy().tolist())
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    unlabeled = Subset(train_dataset, unlabeled)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=generate_batch,
    )

    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total, k = 0, 0, 0
    outputs_fin = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(unlabeled_loader), desc="Inferring"):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)

            _, predicted = torch.max(outputs.data, 1)
            total += cls.size(0)
            correct += (predicted == cls).sum().item()
            for j in range(len(outputs)):
                outputs_fin[unlabeled[k]] = {}
                outputs_fin[unlabeled[k]]["prediction"] = predicted[j].item()
                outputs_fin[unlabeled[k]]["logits"] = outputs[j].cpu().numpy()
                k += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
