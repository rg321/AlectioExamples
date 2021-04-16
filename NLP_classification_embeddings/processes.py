import os
import torch
import sys
import yaml
import torch
import argparse
import time
import pickle
import subprocess

# Get all the paths
def make_call_string(arglist):
    result_string = ""
    for arg in arglist:
        result_string += "".join(["--", arg[0], " ", str(arg[1]), " "])
    return result_string


def multi_gpu(n):
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        n = round(n / 1.414)
    return n


cwd = os.path.dirname(os.path.abspath(__file__))

huggingface_dir = os.path.join(cwd, "huggingface")

classification_dir = os.path.join(huggingface_dir, "examples", "text-classification")

data_dir = os.path.join(huggingface_dir, "glue_data", "Reuters")

gpu_num = torch.cuda.device_count()

batch_size_dict = {
    "albert-base-v2": 32,
    "albert-large-v2": 8,
    "distilbert-base-cased": 32,
    "xlm-roberta-base": 16,
    "roberta-base": 32,
    "bert-base-cased": 32,
    "xlnet-base-cased": 16,
    "bert-base-uncased": 32,
}


def getdatasetstate(config_args={}):  # Why is args empty here?
    return {k: k for k in range(config_args["trainsize"])}


def train(config_args, labeled, resume_from: int = 0, ckpt_file: str = ""):

    # Add the args from the config file
    model = config_args.get("model")
    per_device_train_batch_size = batch_size_dict[model]
    output_dir = os.path.join(cwd, config_args["EXPT_DIR"])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    labeled_file = os.path.join(output_dir, "indices.pkl")

    with open(labeled_file, "wb") as pkl_file:
        pickle.dump(labeled, pkl_file)

    max_seq_length = multi_gpu(config_args["max_seq_length"])
    logging_steps = multi_gpu(config_args["logging_steps"])
    warmup_steps = multi_gpu(config_args["warmup_steps"])
    max_steps = multi_gpu(config_args["max_steps"])

    learning_rate = config_args["learning_rate"]
    train_size = config_args["trainsize"]

    seed_val = config_args["seed"]
    if seed_val == -1:
        seed_val = int(time.time() * 1) % 100000

    arglist = [
        ["task_name", "Reuters"],
        ["do_train", ""],
        ["data_dir", data_dir],
        ["model_name_or_path", model],
        ["per_device_train_batch_size", per_device_train_batch_size],
        ["max_seq_length", max_seq_length],
        ["learning_rate", learning_rate],
        ["logging_steps", logging_steps],
        ["output_dir", output_dir],
        ["overwrite_output_dir", ""],
        ["warmup_steps", warmup_steps],
        ["max_steps", max_steps],
        ["labeled_file", labeled_file],
        ["train_size", train_size],
        ["seed", seed_val],
    ]

    call_string = " ".join(
        [
            f"python {os.path.join(classification_dir, 'run_glue.py')}",
            make_call_string(arglist),
        ]
    )

    subprocess.call(call_string, shell=True)

    return


def test(config_args, ckpt_file):

    output_dir = os.path.join(cwd, config_args["EXPT_DIR"])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    task_name = config_args["dataset"]
    train_size = config_args["trainsize"]

    arglist = [
        ["task_name", task_name],
        ["do_test", ""],
        ["data_dir", data_dir],
        ["model_name_or_path", output_dir],
        ["output_dir", output_dir],
        ["train_size", train_size],
    ]

    call_string = " ".join(
        [
            f"python {os.path.join(classification_dir, 'run_glue.py')}",
            make_call_string(arglist),
        ]
    )

    pred_output_test_file = os.path.join(
        output_dir, f"test_pred_results_{task_name.lower()}.pkl",
    )

    subprocess.call(call_string, shell=True)

    with open(pred_output_test_file, "rb") as pkl_file:
        return pickle.load(pkl_file)


def infer(config_args, unlabeled, ckpt_file):

    output_dir = os.path.join(cwd, config_args["EXPT_DIR"])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    task_name = config_args["dataset"]
    train_size = config_args["trainsize"]

    arglist = [
        ["task_name", task_name],
        ["do_predict", ""],
        ["data_dir", data_dir],
        ["model_name_or_path", output_dir],
        ["output_dir", output_dir],
        ["train_size", train_size],
    ]

    call_string = " ".join(
        [
            f"python {os.path.join(classification_dir, 'run_glue.py')}",
            make_call_string(arglist),
        ]
    )

    output_test_file = os.path.join(
        output_dir, f"raw_test_results_{task_name.lower()}.pkl",
    )

    subprocess.call(call_string, shell=True)

    with open(output_test_file, "rb") as pkl_file:
        d = pickle.load(pkl_file)

    outputs = {}
    for l in unlabeled:
        outputs[l] = {"logits": d[l]}

    return {"outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(cwd, "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)
    labeled = list(range(100))
    resume_from = None
    ckpt_file = "ckpt_0.pt"

    print("Running Training")
    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Running Testing")
    print(test(args, ckpt_file=ckpt_file))
    print("Running Inference")
    print(infer(args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file))
