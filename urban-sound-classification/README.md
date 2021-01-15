## Urban Sound Classification

Classify types of < 4 second sounds (.wav files) into 10 different sound-types

### Running steps

Make sure `urban-sound-classification` is your working directory. 

Configure demo state
1. Create a data directory `mkdir data` and a log directory `mkdir log`
2. Modify hyper-parameters in the `config.yaml` file as needed (to set the train_epochs, batch_size, learning rate, momentum, etc)

Set up environment

3. Run `aws configure` to set up connection with AWS
4. Install the alectio-sdk with `pip install ../../.` (assuming you are in this current directory).
5. After activating your virtual environment (see project-level README for instructions), install all dependencies in the requirements.txt found in this repository `pip install -r requirements.txt`
6. Finally, run `python main.py` to start the traning process. 

## Instructions for downloading the Urban Sound dataset
1. Download the dataset from https://urbansounddataset.weebly.com/urbansound8k.html
2. In the downloaded folder, move UrbanSound8K/audio to ./data. Move UrbanSound8K/metadata/UrbanSound8K.csv to ./data
3. The dataset is already split up into 10 folds, for the purposes of this demo, we will be training on folds 1-9 and testing on fold 10.
4. The first time running, the code will take considerably longer because it is genereating the feature vectors for each sound file (by computing the Mel-Frequency Spectogram of each file). Subsequent runs will use these stored feature vectors to save time. 
