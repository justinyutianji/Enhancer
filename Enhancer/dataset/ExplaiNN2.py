# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader
import csv
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from Bio import SeqIO
from utils import process_data, dna_one_hot, EnhancerDataset, split_dataset, evaluate_model,train_model, evaluate_regression_model,regression_model_plot
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.modules.activation as activation
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
import pickle
sys.path.append('../model')  
from model import ExplaiNN2

# Import or define the functions and classes
# Make sure these are available or define them if not
# from your_module import split_dataset, EnhancerDataset, ConvNetDeep, train_model, evaluate_regression_model

#-------------------------------------
#*********Train ConvNetDeep************
#************Predict GFP+**************
#-------------------------------------


# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv')

#seeds = [random.randint(0, 2**32 - 1) for _ in range(3)]
# Initialize the R_square list
seed_list = []
batch_list = []
lr_list = []
dropout_list = []
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_corr_list = []
spearman_corr_list = []
cnns = []

#seeds = [44,234,90,500,293]
#batches = [24,48,96,168,288]
batches = [96]
seeds = [98]
num_cnns = list(range(10, 251, 10))
dropout = 0.3
# Split the dataset
for seed in seeds: 
    for batch in batches:
        train, test = split_dataset(df, split_type='random', cutoff=0.8, seed=seed)

        # Process datasets
        train = EnhancerDataset(train, label_mode='G+', scale_mode='none')
        test = EnhancerDataset(test, label_mode='G+', scale_mode='none')

        # DataLoader setup
        train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
        test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

        # Hyperparameter search
        for num_cnn in num_cnns:
            # Model setup
            input_model = ExplaiNN2(num_cnns = num_cnn, input_length = 608, num_classes = 1, 
                 filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                 fc_filter1 = 20, fc_filter2 = 1, drop_out = 0.3, weight_path = None)# Training

            #for learning_rate in [6e-5, 1e-4, 4e-4]:
            for learning_rate in [1e-4]:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")
                _, _, model, train_losses_by_batch, test_losses_by_batch, results, device  = train_model(
                    input_model, train_loader, test_loader, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= False, dir_path=None)

                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pathway = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN2_{num_cnn}cnns_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}'
                mse, rmse, mae, r2, pearson_corr, spearman_corr = regression_model_plot(
                    model, test_loader, train_losses_by_batch, test_losses_by_batch, device, results, 
                    save_plot = True, dir_path = pathway
                    )
                print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                print(f"R^2: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                mse_list.append(mse)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)
                pearson_corr_list.append(pearson_corr)
                spearman_corr_list.append(spearman_corr)
                cnns.append(num_cnns)

# Save the R_square results to a CSV file
filename = '/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN2_R_Square.csv'

results_df = pd.DataFrame({
    "num_cnns": cnns,
    "batch": batch_list,
    "lr": lr_list,
    "drop_out": dropout_list,
    "seed": seed_list,
    "mse":mse_list,
    "rmse":rmse_list,
    "mae":mae_list,
    "r2":r2_list,
    "pearson_corr":pearson_corr_list,
    "spearman_corr":spearman_corr_list
})
results_df.to_csv(filename, index=False)
print(f"R_square values saved to {filename}")