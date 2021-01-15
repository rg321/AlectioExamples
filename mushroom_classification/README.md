## Mushroom Classification Running Instructions

> A task where you'll predict the toxicity (posionous or edible) of mushrooms in the wild using 23 numerical features describing quantitative and qualitative aspects of the mushrooms.

### Running steps

Make sure you are on the `develop` branch, with `mushroom_classification` as your working directory. 

Configure demo state
1. Create a data directory `mkdir data` and a log directory `mkdir log`
2. Download the .csv data from kaggle from this [link](https://www.kaggle.com/uciml/mushroom-classification). Place the csv data in the `./data` folder you created in step #1. Rename the CSV to `mushrooms.csv`
3. Modify hyper-parameters in the `config.yaml` file as needed (to set the train_epochs, batch_size, learning rate, momentum, etc)

Set up environment

4. Find and set the Alectio backend API key (found in the front end platform) with `export ALECTIO_API_KEY="KEY_HERE"`
5. Run `aws configure` to set up connection with AWS
6. Install the alectio-sdk with `pip install ../../.` (assuming you are in this current directory).
7. After activating your virtual environment (see project-level README for instructions), install all dependencies in the requirements.txt found in this repository `pip install -r requirements.txt`
8. Finally, run `python main.py --config config.yaml` to start the traning process. 