from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate

# put the train/test/infer processes into the constructor
app = Pipeline(
    name="cifar10",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='<YOUR_TOKEN_HERE>'
)

if __name__ == "__main__":
    app()
