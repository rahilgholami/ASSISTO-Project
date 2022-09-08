# SCC Project

## 1. Data Preparation

Let's start by preparing data. 

By runing `datapreparation/data_alignment.py`, we align the time series with the real time and split them to 1-hour intervals.

To read the data and split them to training and test sets, run the following code, `datapreparation/train_test_split.ipynb`.

## 2. Training

```train
python train.py --epochs 500 --batch_size 128 --hdim 128 --input_channel 512 --temperature 0.07 --window_size 1000
```
