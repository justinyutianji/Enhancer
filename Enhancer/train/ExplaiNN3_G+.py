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
from model import ExplaiNN3
import shutil

# Import or define the functions and classes
# Make sure these are available or define them if not
# from your_module import split_dataset, EnhancerDataset, ConvNetDeep, train_model, evaluate_regression_model

#-------------------------------------
#*********Train ConvNetDeep************
#************Predict GFP+**************
#-------------------------------------


# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv')
# Initialize the R_square list
seed_list = []
batch_list = []
lr_list = []
dropout_list = []

mse_list_p = []
rmse_list_p = []
mae_list_p = []
r2_list_p = []
pearson_corr_list_p = []
spearman_corr_list_p = []

mse_list_r = []
rmse_list_r = []
mae_list_r = []
r2_list_r = []
pearson_corr_list_r = []
spearman_corr_list_r = []

best_pearson_epochs = []
best_r2_epochs = []

cnns = []

seeds = [random.randint(1, 1000) for _ in range(3)]
batches = [96,168]
num_cnns = list(range(10, 251, 10))
num_cnns.append(256)
dropout = 0.3
lrs = [1e-4, 3e-4]
# Split the dataset
for seed in seeds: 
    for batch in batches:
        for cnn in num_cnns:
            train, test = split_dataset(df, split_type='random', cutoff=0.8, seed=seed)

            # Process datasets
            train = EnhancerDataset(train, label_mode='G+', scale_mode='none')
            test = EnhancerDataset(test, label_mode='G+', scale_mode='none')

            # DataLoader setup
            train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
            test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

            # Hyperparameter search
            input_model = ExplaiNN3(num_cnns = cnn, input_length = 608, num_classes = 1, 
                filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                drop_out = 0.3, weight_path = None)# Training

            for learning_rate in lrs:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")
                os.makedirs(f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}', exist_ok=True)
                pathway = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/models'
                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch, device  = train_model(
                    input_model, train_loader, test_loader, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= True, dir_path=pathway)
                
                # Saving all metrics for best r2 model and pearson model respectively
                index = best_pearson_epoch - 1
                mse_list_p.append(results['mse'][index])
                rmse_list_p.append(results['rmse'][index])
                mae_list_p.append(results['mae'][index])
                r2_list_p.append(results['r2'][index])
                pearson_corr_list_p.append(results['pearson_corr'][index])
                spearman_corr_list_p.append(results['spearman_corr'][index])
                
                index = best_r2_epoch - 1
                mse_list_r.append(results['mse'][index])
                rmse_list_r.append(results['rmse'][index])
                mae_list_r.append(results['mae'][index])
                r2_list_r.append(results['r2'][index])
                pearson_corr_list_r.append(results['pearson_corr'][index])
                spearman_corr_list_r.append(results['spearman_corr'][index])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                cnns.append(cnn)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)

                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Generating Plots
                pathway = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/best_pearson'
                model_path = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/models/model_epoch_{best_pearson_epoch}.pth'
                mse, rmse, mae, r2, pearson_corr, spearman_corr = regression_model_plot(
                    input_model, test_loader, train_losses_by_batch, test_losses_by_batch, 
                    device, results, label_mode = "G+",
                    save_plot = False, dir_path = pathway, model_path = model_path)
                
                pathway = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/best_r_square'
                model_path = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/models/model_epoch_{best_r2_epoch}.pth'
                mse, rmse, mae, r2, pearson_corr, spearman_corr = regression_model_plot(
                    input_model, test_loader, train_losses_by_batch, test_losses_by_batch, 
                    device, results, label_mode = "G+",
                    save_plot = False, dir_path = pathway, model_path = model_path)
                print("Deleting Models")
                if os.path.exists(f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/models'):
                    # Remove the directory and all its contents
                    shutil.rmtree(f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_cnn{cnn}_dp{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}/models')
                    print("Finish Deleting Models")

                print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                print(f"R^2: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")

    # Save the R_square results to a CSV file
    filename = f'/pmglocal/ty2514/Enhancer/Enhancer/results/ExplaiNN3_G+_Metrics_seed{seed}.csv'

    results_df = pd.DataFrame({
        "num_cnns": cnns,
        "batch": batch_list,
        "lr": lr_list,
        "drop_out": dropout_list,
        "seed": seed_list,
        "mse_p":mse_list_p,
        "rmse_p":rmse_list_p,
        "mae_p":mae_list_p,
        "r2_p":r2_list_p,
        "pearson_corr_p":pearson_corr_list_p,
        "spearman_corr_p":spearman_corr_list_p,
        "mse_r":mse_list_r,
        "rmse_r":rmse_list_r,
        "mae_r":mae_list_r,
        "r2_r":r2_list_r,
        "pearson_corr_r":pearson_corr_list_r,
        "spearman_corr_r":spearman_corr_list_r,
        "best_pearson_epoch": best_pearson_epochs,
        "best_r2_epoch": best_r2_epochs
    })
    results_df.to_csv(filename, index=False)
    print(f"R_square values saved to {filename}")