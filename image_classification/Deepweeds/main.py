from alectio_sdk.skd import Pipeline
from processes import train, test, infer, getdatasetstate
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config.yaml", required=True)
args = parser.parse_args()
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
    token='<YOUR_TOKEN_HERE>'
)

if __name__ == "__main__":
    app()
