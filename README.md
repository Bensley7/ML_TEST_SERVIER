# Molecular Property Classification (Sevrier ML Test)

This repository contains two deep neural networks models.

The first one is a attention pooling LSTM model for sequential binary encoding input (such as exercice 1 -> GetMorganFingerprintAsBitVect) - a classic approach.

The second model that is much more accurate (and more complicated) is based on the following paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237).

Theses models aim to do classification or multi-classification (multi output) of some molecule properties based on molecule smiles.


## Requirements

In order to run the project with CUDA, please install CUDA and cudnn. You can run it on cpu too. Check the project config default.py for more intuition on project hyperparamaters.


## Installations

### Option 1: From source
    ```
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda 
    export PATH=~/miniconda/bin:$PATH
    conda update -n base conda
    conda create -y --name servier python=3.7
    conda activate servier
    pip install torch
    conda install -c conda-forge rdkit
    git clone git@github.com:Bensley7/ML_TEST_SERVIER.git
    cd ML_TEST_SERVIER
    pip install -e .
    ```

### Option 2: From Docker

1. `git clone git@github.com:Bensley7/ML_TEST_SERVIER.git`
2. `cd ML_TEST_SERVIER`
3. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
4. `docker build -t servier .`
5. `docker run -it servier:latest bash`
6. In order to run the project with external data (data_single_train, val etc ...), group these data in a single folder in the host machine (let call the full path $servier_data_path) and then run the following that binds (mounts) docker folder with host machine folder `docker run -v $servier_data_path:/opt/app/data -it servier bash` Note that $servier_data_path should be an absolute path


## Run Project

### For Training

#### Data Split
In order to train, the project first split data into train, test and val. We don't let the training do the splitting task since we want to compare models and thus to ensure reproductibility.

To split you can choose two modes  "classification" or "multiclass" based on the config file in configs.
For `classification` split run :
```
python dataset/digest --raw_data_path data/dataset_single.csv
```

For `mutliclass` go to `configs/multi_output_config.yml ` or you can create your own config based on the previous template and add desired columns in `labels_name` attribute of the config. Then run 

```
python dataset/digest --raw_data_path data/dataset_multi.csv --config_file ../configs/multi_output_config.yml
```
- PS: `data/` folder is an example. You can add raw data in any folder - you just need to mention it in the parser

- You will find train, val and test data on the same root of your raw data.
 
#### Training 

1. For the first exercice - we will run the attention pooling lstm model. Run 
```
servier train --train_data_path data/dataset_single_train.csv --valid_data_path data/dataset_single_valid.csv --test_data_path data/dataset_single_test.csv --model_dir lstm_model
```

2. For the second exercice - we will run mpn_ffc model. Run: 
```
servier train --train_data_path data/dataset_single_train.csv --valid_data_path data/dataset_single_valid.csv --test_data_path data/dataset_single_test.csv --config_file configs/binary_mpn_fft_config.yml --model_dir mpn_model
```
- You can change the config script `defaults.py` and modify attributes,  but we recommand using the predifined configs for each task. 
- model_dir is an example but it the name of the folder should be different at each run since it is overwrited.

3. For the third exercice - we will run mpn_ffc model with multiple columns. Run:

```
servier train --train_data_path data/dataset_multi_train.csv --valid_data_path data/dataset_multi_valid.csv --test_data_path data/dataset_multi_test.csv --config_file configs/multi_output_config.yml --model_dir mpn_multi_model
```

You can change loss fucntions and hyperparameters in the config file. Accepted options are commented.

#### Evaluation
```
servier evaluate --model_path ./trained_models/model_name.pt --val_data_path ./data/dataset_single_test.csv
```
- As for training you add the configs depending on the exercice

#### prediction
```
servier predict  --model_path ./trained_models/model_name.pt --smile 'NC(=O)NC(Cc1ccccc1)C(=O)O'
```

## Flask

Run `python molecular_ml/web/server.py  path_to_your_desired_model --config_file path_of_your_desired_config_file` 

There are options about host and port too in the server arguments .

## Result

- ATTENTION_LSTM - basic features : M1
- MPN_FFC_SINGLE : M2
- MPN_FFC_multi : M3

Method  | metric  | score
------------- |------------- | -------------
M1      | accuracy  | 0.822
M1      | f1 score  | 0.90
M1      | ROC CURVE  | 0.83
M2      | accuracy  | 0.83
M2      | f1 score  | 0.93
M2      | ROC CURVE  | 0.9
M3      | accuracy  | 0.81
M3      | f1 score  | 0.898

#### Next
- Better Split of data (molecules that are look alike - better knowledge in chemistry) to avoid data leakage
- Techniques of sampling molecules - it is not like images- find rights isomorphisms for data augmenation too
- Better statistical sampling - tried with digest.py , but still more effort in that matter especially that the dataset is imbalanced (1's are more present that 0's)
- Transfer learning
- Add more features to mpn_ffc graph to grasp interaction and learn a better representation of graphs
- Fine tune models - everythong is in config - we should spend time testing
- Oversampling
- etc




