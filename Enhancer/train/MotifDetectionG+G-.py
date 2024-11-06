from torch.utils.data import DataLoader
from utils import split_dataset, train_model,EnhancerDatasetWithID
import pandas as pd
import glob

import sys
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os
import subprocess
import json
import shutil
import random


sys.path.append('../model')  
from model import ConvNetDeep, DanQ, ExplaiNN,ConvNetDeep2, ExplaiNN2, ExplaiNN3,DeepSTARR

params = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN3_G+G-/ExplaiNN3_G+G-_Metrics.csv')
cnns = list(range(10, 101, 5))
target_labels = ["GFP+", "GFP-"]
feature_list = ["G+", "G-"]

df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_input_data.csv')
# Data frame that stores the final results
final_data = {label: [] for label in target_labels}
seeds = [random.randint(1, 1000) for _ in range(4)]

# Loop over each CNN configuration
for cnn in cnns:
    
    temp_df = params[params['num_cnns'] == cnn]
    group_means = temp_df.groupby(['lr', 'drop_out', 'batch'])['pearson_corr_p'].mean().reset_index()
    best_group = group_means.loc[group_means['pearson_corr_p'].idxmax()]
    best_lr = best_group['lr']
    best_dropout = best_group['drop_out']
    best_batch = best_group['batch']

    best_group_df = temp_df[(temp_df['lr'] == best_lr) & (temp_df['drop_out'] == best_dropout) & (temp_df['batch'] == best_batch)]
    best_seed_row = best_group_df.loc[best_group_df['pearson_corr_p'].idxmax()]
    best_seed = int(best_seed_row['seed'])
    best_batch = int(best_batch)

    num_cnns = cnn
    new_seeds = seeds + [best_seed]
    for seed in new_seeds:

        output_dir = f'/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_G+G-_Pred/CNN{num_cnns}_Seed{seed}'

        print(f'\ncnn: {num_cnns}, lr: {best_lr}, drop_out: {best_dropout}, batch: {best_batch}, seed: {seed}\n')

        # Model Training
        batch = best_batch
        learning_rate = best_lr
        train_df, val_df, test_df = split_dataset(df, split_type='random', split_pattern=[0.7, 0.15, 0.15], seed=seed)
        
        train = EnhancerDatasetWithID(train_df, feature_list=feature_list, scale_mode='none')
        test = EnhancerDatasetWithID(test_df, feature_list=feature_list, scale_mode='none')
        validation = EnhancerDatasetWithID(val_df, feature_list=feature_list, scale_mode='none')

        train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
        test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=False)
        val_loader = DataLoader(dataset=validation, batch_size=batch, shuffle=False)

        input_model = ExplaiNN3(num_cnns=num_cnns, input_length=608, num_classes=2, filter_size=19, num_fc=2, pool_size=7, pool_stride=7, drop_out=0.3, weight_path=None)
        _, _, model, *_ = train_model(input_model, train_loader, val_loader, test_loader, target_labels=target_labels, num_epochs=200, batch_size=batch, learning_rate=learning_rate, criteria='mse', optimizer_type="adam", patience=15, seed=seed, save_model=True, dir_path=output_dir)

        # Model Interpreting
        command = [
            "python", "Interpret.py", 
            "--num_class", "2", 
            "--seed", str(seed), 
            "--batch", str(best_batch), 
            "--num_cnns", str(num_cnns), 
            "--learning_rate", str(best_lr), 
            "--result_dir", output_dir,
            "--feature_list", json.dumps(['G+', 'G-']),  
            "--target_labels", json.dumps(['GFP+', 'GFP-']) 
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print("STDERR:", result.stderr)

        # Saving Results
        for label in target_labels:
            tf_file = os.path.join(output_dir, 'importance', f'{label}_sorted_tf_order.csv')
            if os.path.exists(tf_file):
                tf_filters = pd.read_csv(tf_file)
                tf_filters = tf_filters.iloc[:,:10]

                tf_names = tf_filters['tf_name'].tolist()
                importance_scores = tf_filters['importance_score'].tolist()
                row_data = {'num_cnn': num_cnns, 'seed': seed}
                for i in range(10):
                    row_data[f'tf{i+1}'] = tf_names[i]
                    row_data[f'score{i+1}'] = importance_scores[i]
                final_data[label].append(row_data)

        # Cleanup step to retain only CSV files (tf_files) and delete all other files, including .pth model files
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            
            # If the item is a file and not a CSV, delete it
            if os.path.isfile(item_path) and not item_path.endswith('.csv'):
                os.remove(item_path)
            
            # If the item is a directory
            elif os.path.isdir(item_path):
                if item == 'importance':
                    # Move the CSV files from the 'importance' subdirectory to the main output_dir
                    for label in target_labels:
                        tf_file_path = os.path.join(item_path, f'{label}_sorted_tf_order.csv')
                        if os.path.exists(tf_file_path):
                            shutil.move(tf_file_path, output_dir)
                    # Remove the 'importance' subdirectory after moving CSV files
                    shutil.rmtree(item_path)
                else:
                    # Remove any other subdirectories entirely
                    shutil.rmtree(item_path)


# Convert each label's data list into a DataFrame and save as CSV
output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_G+G-_Pred'
for label, data in final_data.items():
    label_df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, f'{label}_combined_tf_importance.csv')
    label_df.to_csv(output_file, index=False)
    print(f"Saved combined TF importance data for {label} to {output_file}")