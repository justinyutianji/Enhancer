# Import necessary libraries
import pandas as pd
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader
from utils import EnhancerDataset, split_dataset,train_model
import pandas as pd

import sys
import os
sys.path.append('../model')  
from model import ExplaiNN3

# Import or define the functions and classes
# Make sure these are available or define them if not
# from your_module import split_dataset, EnhancerDataset, ConvNetDeep, train_model, evaluate_regression_model

#-------------------------------------
#*********Train ConvNetDeep************
#************Predict GFP**************
#-------------------------------------


# Load the dataset
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv')
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

cnns = []

seeds = [random.randint(1, 1000) for _ in range(3)]
batches = [96,168]
num_cnns = list(range(10, 251, 10))
dropout = 0.3
lrs = [5e-5, 1e-4, 5e-4]
target_labels = ['GFP']

output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN3_GFP'
os.makedirs(output_dir, exist_ok=True)
# Save the R_square results to a CSV file
filename = os.path.join(output_dir, 'ExplaiNN3_GFP_Metrics.csv')
# Split the dataset
for seed in seeds: 
    for batch in batches:
        for cnn in num_cnns:
            train, test = split_dataset(df, split_type='random', cutoff=0.8, seed=seed)

            # Process datasets
            train = EnhancerDataset(train, feature_list=['GFP'], scale_mode='none')
            test = EnhancerDataset(test, feature_list=['GFP'], scale_mode='none')

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
                _, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch, device  = train_model(
                    input_model, train_loader, test_loader, target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, 
                    criteria='mse',optimizer_type = "adam", patience=10, seed = seed, save_model= False, dir_path=output_dir)
                
                # Saving all metrics for best r2 model and pearson model respectively
                index = best_pearson_epoch
                mse_list_p.append(results['GFP']['mse'][index])
                rmse_list_p.append(results['GFP']['rmse'][index])
                mae_list_p.append(results['GFP']['mae'][index])
                r2_list_p.append(results['GFP']['r2'][index])
                pearson_corr_list_p.append(results['GFP']['pearson_corr'][index])
                spearman_corr_list_p.append(results['GFP']['spearman_corr'][index])
                
                index = best_r2_epoch
                mse_list_r.append(results['GFP']['mse'][index])
                rmse_list_r.append(results['GFP']['rmse'][index])
                mae_list_r.append(results['GFP']['mae'][index])
                r2_list_r.append(results['GFP']['r2'][index])
                pearson_corr_list_r.append(results['GFP']['pearson_corr'][index])
                spearman_corr_list_r.append(results['GFP']['spearman_corr'][index])

                seed_list.append(seed)
                batch_list.append(batch)
                lr_list.append(formatted_lr)
                dropout_list.append(dropout)
                cnns.append(cnn)
                best_pearson_epochs.append(best_pearson_epoch)
                best_r2_epochs.append(best_r2_epoch)

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
