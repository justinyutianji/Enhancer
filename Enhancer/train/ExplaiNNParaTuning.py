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

# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_input_data.csv')

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

seeds_cnn = [995, 582,566]
batches = [96,168]
num_cnns = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]

dropout = 0.3
lrs = [1e-4, 5e-4, 1e-3]
target_labels = ['GFP']

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN3_GFP2'
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