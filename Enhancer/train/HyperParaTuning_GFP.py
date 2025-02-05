# Import necessary libraries
import pandas as pd
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader
from utils import EnhancerDatasetWithID, split_dataset,train_model
import pandas as pd

import sys
import os

sys.path.append('../model')  
from model import  DanQ, ConvNetDeep, DeepSTARR, ExplaiNN3

# Import or define the functions and classes
# Make sure these are available or define them if not
# from your_module import split_dataset, EnhancerDatasetWithID, ConvNetDeep, train_model, evaluate_regression_model

#-------------------------------------
#*********Train ExplaiNN************
#************Predict GFP**************
#-------------------------------------
# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_merged_data.csv')

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

seeds = [random.randint(1, 1000) for _ in range(5)]
seeds_cnn = seeds[:3]
batches = [96,168]
num_cnns = list(range(5, 131, 5))
dropout = 0.3
lrs = [1e-4, 5e-4, 1e-3]
target_labels = ['GFP']

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN3_GFP'
os.makedirs(output_dir, exist_ok=True)
# Save the R_square results to a CSV file
filename = os.path.join(output_dir, 'ExplaiNN3_GFP_Metrics.csv')
# Split the dataset
for seed in seeds_cnn: 
    for batch in batches:
        for cnn in num_cnns:
            train_df, val_df, test_df = split_dataset(df, split_type='random', split_pattern=[0.7, 0.15, 0.15], seed=seed)

            train = EnhancerDatasetWithID(train_df, feature_list=['GFP'], scale_mode='none')
            val = EnhancerDatasetWithID(val_df, feature_list=['GFP'], scale_mode='none')
            test = EnhancerDatasetWithID(test_df, feature_list=['GFP'], scale_mode='none')

            # DataLoader setup
            train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
            val_loader = DataLoader(dataset=val, batch_size=batch, shuffle=False)
            test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=False)

            # Hyperparameter search
            input_model = ExplaiNN3(num_cnns = cnn, input_length = 608, num_classes = 1, 
                filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                drop_out = 0.3, weight_path = None)# Training

            for learning_rate in lrs:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")
                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch,  pearson_metrics, r2_metrics, device  = train_model(
                    input_model, train_loader, val_loader, test_loader,target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= False, dir_path=output_dir)
                
                # Saving all metrics for best r2 model and pearson model respectively
                mse_list_p.append(pearson_metrics['mse'][-1])
                rmse_list_p.append(pearson_metrics['rmse'][-1])
                mae_list_p.append(pearson_metrics['mae'][-1])
                r2_list_p.append(pearson_metrics['r2'][-1])
                pearson_corr_list_p.append(pearson_metrics['pearson_corr'][-1])
                spearman_corr_list_p.append(pearson_metrics['spearman_corr'][-1])
                
                mse_list_r.append(r2_metrics['mse'][-1])
                rmse_list_r.append(r2_metrics['rmse'][-1])
                mae_list_r.append(r2_metrics['mae'][-1])
                r2_list_r.append(r2_metrics['r2'][-1])
                pearson_corr_list_r.append(r2_metrics['pearson_corr'][-1])
                spearman_corr_list_r.append(r2_metrics['spearman_corr'][-1])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)
                cnns.append(cnn)

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


#-------------------------------------
#*********Train ConvNetDeep************
#************Predict GFP**************
#-------------------------------------


# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_merged_data.csv')

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

target_labels = ['GFP']
batches = [96,168,322]
learning_rates = [1e-4, 5e-4,1e-3,5e-3]

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ConvNetDeep_GFP'
os.makedirs(output_dir, exist_ok=True)
# Save the R_square results to a CSV file
filename = os.path.join(output_dir, 'ConvNetDeep_GFP_Metrics.csv')

# Split the dataset
for seed in seeds: 
    for batch in batches:
        train_df, val_df, test_df = split_dataset(df, split_type='random', split_pattern=[0.7, 0.15, 0.15], seed=seed)

        # Process datasets
        train = EnhancerDatasetWithID(train_df, feature_list=['GFP'], scale_mode='none')
        val = EnhancerDatasetWithID(val_df, feature_list=['GFP'], scale_mode='none')
        test = EnhancerDatasetWithID(test_df, feature_list=['GFP'], scale_mode='none')

        # DataLoader setup
        train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val, batch_size=batch, shuffle=False)
        test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=False)

        # Hyperparameter search
        for dropout in [0.3]:
            # Model setup
            input_model = ConvNetDeep(num_classes=1, drop_out=dropout)
            for learning_rate in learning_rates:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")

                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch,  pearson_metrics, r2_metrics, device  = train_model(
                    input_model, train_loader, val_loader, test_loader,target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= False, dir_path=output_dir)
                
                # Saving all metrics for best r2 model and pearson model respectively
                mse_list_p.append(pearson_metrics['mse'][-1])
                rmse_list_p.append(pearson_metrics['rmse'][-1])
                mae_list_p.append(pearson_metrics['mae'][-1])
                r2_list_p.append(pearson_metrics['r2'][-1])
                pearson_corr_list_p.append(pearson_metrics['pearson_corr'][-1])
                spearman_corr_list_p.append(pearson_metrics['spearman_corr'][-1])
                
                mse_list_r.append(r2_metrics['mse'][-1])
                rmse_list_r.append(r2_metrics['rmse'][-1])
                mae_list_r.append(r2_metrics['mae'][-1])
                r2_list_r.append(r2_metrics['r2'][-1])
                pearson_corr_list_r.append(r2_metrics['pearson_corr'][-1])
                spearman_corr_list_r.append(r2_metrics['spearman_corr'][-1])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)

results_df = pd.DataFrame({
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


#-------------------------------------
#*********Train DANQ************
#************Predict GFP**************
#-------------------------------------


#seeds = [random.randint(0, 2**32 - 1) for _ in range(3)]
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

target_labels = ['GFP']

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/DanQ_GFP'
os.makedirs(output_dir, exist_ok=True)
# Save the R_square results to a CSV file
filename = os.path.join(output_dir, 'DanQ_GFP_Metrics.csv')

# Split the dataset
for seed in seeds: 
    for batch in batches:
        train_df, val_df, test_df = split_dataset(df, split_type='random', split_pattern=[0.7, 0.15, 0.15], seed=seed)

        train = EnhancerDatasetWithID(train_df, feature_list=['GFP'], scale_mode='none')
        val = EnhancerDatasetWithID(val_df, feature_list=['GFP'], scale_mode='none')
        test = EnhancerDatasetWithID(test_df, feature_list=['GFP'], scale_mode='none')

        # DataLoader setup
        train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val, batch_size=batch, shuffle=False)
        test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=False)

        # Hyperparameter search
        for dropout in [0.3]:
            # Model setup
            input_model = DanQ(input_length = 608, num_classes = 1)
            for learning_rate in learning_rates:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")

                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch,  pearson_metrics, r2_metrics, device  = train_model(
                    input_model, train_loader, val_loader, test_loader,target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= False, dir_path=output_dir)
                
                # Saving all metrics for best r2 model and pearson model respectively
                mse_list_p.append(pearson_metrics['mse'][-1])
                rmse_list_p.append(pearson_metrics['rmse'][-1])
                mae_list_p.append(pearson_metrics['mae'][-1])
                r2_list_p.append(pearson_metrics['r2'][-1])
                pearson_corr_list_p.append(pearson_metrics['pearson_corr'][-1])
                spearman_corr_list_p.append(pearson_metrics['spearman_corr'][-1])
                
                mse_list_r.append(r2_metrics['mse'][-1])
                rmse_list_r.append(r2_metrics['rmse'][-1])
                mae_list_r.append(r2_metrics['mae'][-1])
                r2_list_r.append(r2_metrics['r2'][-1])
                pearson_corr_list_r.append(r2_metrics['pearson_corr'][-1])
                spearman_corr_list_r.append(r2_metrics['spearman_corr'][-1])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)


# Define a dictionary with list names as keys and the actual lists as values
list_dict = {
    "seed_list": seed_list,
    "batch_list": batch_list,
    "lr_list": lr_list,
    "dropout_list": dropout_list,
    "mse_list_p": mse_list_p,
    "rmse_list_p": rmse_list_p,
    "mae_list_p": mae_list_p,
    "r2_list_p": r2_list_p,
    "pearson_corr_list_p": pearson_corr_list_p,
    "spearman_corr_list_p": spearman_corr_list_p,
    "mse_list_r": mse_list_r,
    "rmse_list_r": rmse_list_r,
    "mae_list_r": mae_list_r,
    "r2_list_r": r2_list_r,
    "pearson_corr_list_r": pearson_corr_list_r,
    "spearman_corr_list_r": spearman_corr_list_r,
    "best_pearson_epochs": best_pearson_epochs,
    "best_r2_epochs": best_r2_epochs
}

# Iterate over the dictionary and print the length of each list
for name, lst in list_dict.items():
    print(f"Length of {name}: {len(lst)}")


results_df = pd.DataFrame({
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
print(f"Metric values saved to {filename}")


#-------------------------------------
#*********Train DeepSTARR************
#************Predict GFP**************
#-------------------------------------

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


target_labels = ['GFP']

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/DeepSTARR_GFP'
os.makedirs(output_dir, exist_ok=True)
# Save the R_square results to a CSV file
filename = os.path.join(output_dir, 'DeepSTARR_GFP_Metrics.csv')

# Split the dataset
for seed in seeds: 
    for batch in batches:
        train_df, val_df, test_df = split_dataset(df, split_type='random', split_pattern=[0.7, 0.15, 0.15], seed=seed)

        train = EnhancerDatasetWithID(train_df, feature_list=['GFP'], scale_mode='none')
        val = EnhancerDatasetWithID(val_df, feature_list=['GFP'], scale_mode='none')
        test = EnhancerDatasetWithID(test_df, feature_list=['GFP'], scale_mode='none')

        # DataLoader setup
        train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val, batch_size=batch, shuffle=False)
        test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=False)

        # Hyperparameter search
        for dropout in [0.3]:
            # Model setup
            input_model = DeepSTARR(num_classes = 1)
            for learning_rate in learning_rates:
                formatted_lr = "{:.5f}".format(learning_rate)
                print(f"dropout{dropout}_ba{batch}_lr{formatted_lr}_seed{seed}")

                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch,  pearson_metrics, r2_metrics, device  = train_model(
                    input_model, train_loader, val_loader, test_loader,target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=15, seed = seed, save_model= False, dir_path=output_dir)
                
                # Saving all metrics for best r2 model and pearson model respectively
                mse_list_p.append(pearson_metrics['mse'][-1])
                rmse_list_p.append(pearson_metrics['rmse'][-1])
                mae_list_p.append(pearson_metrics['mae'][-1])
                r2_list_p.append(pearson_metrics['r2'][-1])
                pearson_corr_list_p.append(pearson_metrics['pearson_corr'][-1])
                spearman_corr_list_p.append(pearson_metrics['spearman_corr'][-1])
                
                mse_list_r.append(r2_metrics['mse'][-1])
                rmse_list_r.append(r2_metrics['rmse'][-1])
                mae_list_r.append(r2_metrics['mae'][-1])
                r2_list_r.append(r2_metrics['r2'][-1])
                pearson_corr_list_r.append(r2_metrics['pearson_corr'][-1])
                spearman_corr_list_r.append(r2_metrics['spearman_corr'][-1])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)

# Define a dictionary with list names as keys and the actual lists as values
list_dict = {
    "seed_list": seed_list,
    "batch_list": batch_list,
    "lr_list": lr_list,
    "dropout_list": dropout_list,
    "mse_list_p": mse_list_p,
    "rmse_list_p": rmse_list_p,
    "mae_list_p": mae_list_p,
    "r2_list_p": r2_list_p,
    "pearson_corr_list_p": pearson_corr_list_p,
    "spearman_corr_list_p": spearman_corr_list_p,
    "mse_list_r": mse_list_r,
    "rmse_list_r": rmse_list_r,
    "mae_list_r": mae_list_r,
    "r2_list_r": r2_list_r,
    "pearson_corr_list_r": pearson_corr_list_r,
    "spearman_corr_list_r": spearman_corr_list_r,
    "best_pearson_epochs": best_pearson_epochs,
    "best_r2_epochs": best_r2_epochs
}

# Iterate over the dictionary and print the length of each list
for name, lst in list_dict.items():
    print(f"Length of {name}: {len(lst)}")


results_df = pd.DataFrame({
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
