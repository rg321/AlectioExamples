from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer, getdatasetstate

# put the train/test/infer processes into the constructor
app = Pipeline(
    name="cifar10",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='hPrjnEpRi0jLikGNNU8lIiCulCbdHAMJeIcBOc2XB4'
)

if __name__ == "__main__":

    # SAMPLE PAYLOAD
    # payload = {  "project_id": "adbc569e9c9511ea84dd0242ac110002",
    #             "experiment_id": "cb09a5809c9511eaa1bf0242ac110002",
    #             "cur_loop": 0,
    #             "user_id": "8a90a570972811eaad5238c986352c36",
    #             "type": "Classification",
    #             "bucket_name": "alectio-demo"}
    # app._one_loop(payload)

    app(debug=True)
