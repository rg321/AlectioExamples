from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate
import yaml

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name="carvana",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='88eb4b06ef3c41afb526bf224576d72e',
    args=args
)

if __name__ == "__main__":
    AlectioPipeline()
