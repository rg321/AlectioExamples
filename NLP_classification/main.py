import argparse
import yaml, json
import os
from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate
import logging

cwd = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=os.path.join(cwd, "config.yaml"),
    type=str,
    help="Path to config.yaml",
)

# parser.add_argument(
#     "--api_config",
#     default=os.path.join(cwd, "credentials.json"),
#     type=str,
#     help="Path to credentials.json",
# )

args = parser.parse_args()

# with open(args.api_config, "r") as stream:
#     api_key = json.load(stream)["token"]
#     logging.info("Setting Alectio API key.")
#     os.environ["ALECTIO_API_KEY"] = api_key

with open(args.config, "r") as stream:
    args = yaml.safe_load(stream)


# put the train/test/infer processes into the constructor
app = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token='1a37ef028dfa4f9197899ac7db464b61'
)

if __name__ == "__main__":
    # payload = json.load(open(args["sample_payload"], "r"))
    # app._one_loop(args=args, payload=payload)
    app(debug=True)
