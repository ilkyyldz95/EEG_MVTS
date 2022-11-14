# EEG_MVTS

This code implements the accepted manuscript:
```
I. Yıldız Potter, G. Zerveas, C. Eickhoff, D. Duncan, “Unsupervised Multivariate Time-Series Transformers for Seizure Identification on EEG”, IEEE Conference on Machine Learning and Applications (ICMLA), 2022
```
and extends the codebase of George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14-18, 2021_.

# Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

`cd EEG_MVTS/`

Inside an already *existing* root directory, each experiment will create a time-stamped output directory, which contains
model checkpoints, performance metrics per epoch, predictions per sample, the experiment configuration, log files etc.
The following commands assume that you have created a new root directory inside the project directory like this: 
`mkdir experiments`.

[We recommend creating and activating a `conda` or other Python virtual environment (e.g. `virtualenv`) to 
install packages and avoid conficting package requirements; otherwise, to run `pip`, the flag `--user` or `sudo` privileges will be necessary.]

`pip install -r requirements.txt`
or
`conda env create -f mvts_transformer.yml`.

# Datasets

Dataset sources can be found in the manuscript. Preprocess each dataset into training, validation and test partitions by running `data\load_filtered_eeg.py`.

# Example commands

To see all command options with explanations, run: `python src/main.py --help`

You should replace `TUH` below with the name of the desired dataset.

Besides the console output  and the logfile `output.log`, you can monitor the evolution of performance (after installing tensorboard: `pip install tensorboard`) with:
```bash
tensorboard dev upload --name my_exp --logdir path/to/output_dir
```

## Unsupervised Learning via the Proposed Method

### Training:
```bash
python src/main.py --output_dir experiments --comment "anomaly detection" --name TUH_anomaly --records_file TUH_anomaly_records.xls --data_dir TUH/ --data_class pdts --pattern train --val_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam  --pos_encoding learnable  --task anomaly_detection  --subsample_factor 10 --fs 250
```

### Testing:
```bash
python src/main.py --output_dir experiments --comment "anomaly detection" --name TUH_anomaly_evenTest --records_file TUH_anomaly_records.xls --data_dir TUH/ --data_class pdts --pattern all --val_ratio 0.2 --test_ratio 0.2 --test_only testset --load_model experiments/TUH_anomaly_2021-11-19_14-07-16_hPt/checkpoints/model_best.pth --lr 0.001 --optimizer RAdam  --pos_encoding learnable  --task anomaly_detection  --subsample_factor 10 --fs 250
```

## Supervised Learning Baseline

### Training:
```bash
python src/main.py --output_dir experiments --comment "classification" --name TUH_classification_AUPRC_augmented --records_file TUH_classification_records.xls --data_dir TUH/ --data_class pdts --pattern all --val_ratio 0.2 --test_ratio 0.2 --oversample --epochs 100 --lr 0.001 --optimizer RAdam --pos_encoding learnable --task classification --change_output --subsample_factor 10 --fs 250 --key_metric AUPRC
```bash

### Testing:
```bash
python src/main.py --output_dir experiments --comment "classification" --name TUH_classification_AUPRC_augmented --records_file TUH_classification_records.xls --data_dir TUH/ --data_class pdts --pattern all --val_ratio 0.2 --test_ratio 0.2 --test_only testset --load_model experiments/TUH_classification_AUPRC_augmented_2021-11-18_12-23-31_c0Y/checkpoints/model_best.pth --lr 0.001 --optimizer RAdam --pos_encoding learnable --task classification --change_output --subsample_factor 10 --fs 250 --key_metric AUPRC
```bash

## Pre-trained Supervised Learning Baseline

### Training:
```bash
python src/main.py --output_dir experiments --comment "classification" --name TUH_classification_AUPRC_finetuned --records_file TUH_classification_records.xls --data_dir TUH/ --data_class pdts --pattern all --val_ratio 0.2 --test_ratio 0.2 --oversample --load_model experiments/TUH_anomaly_.../checkpoints/model_best.pth --epochs 100 --lr 0.001 --optimizer RAdam --pos_encoding learnable --task classification --change_output --subsample_factor 10 --fs 250 --key_metric AUPRC
```bash

### Testing:
```bash
python src/main.py --output_dir experiments --comment "classification" --name TUH_classification_AUPRC_finetuned --records_file TUH_classification_records.xls --data_dir TUH/ --data_class pdts --pattern all --val_ratio 0.2 --test_ratio 0.2 --test_only testset --load_model experiments/TUH_classification_AUPRC_finetuned_2021-11-17_07-22-21_1YB/checkpoints/model_best.pth --lr 0.001 --optimizer RAdam --pos_encoding learnable --task classification --change_output --subsample_factor 10 --fs 250 --key_metric AUPRC
```bash
