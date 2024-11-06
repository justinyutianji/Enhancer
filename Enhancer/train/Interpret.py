import torch
from torch.utils.data import DataLoader
from utils import EnhancerDataset, find_tsv_file_path
from tools import plot_filter_weight,plot_unit_importance
import numpy as np
import pandas as pd
import subprocess
import torch.nn as nn
import torch.nn.modules.activation as activation
import sys
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
import interpretation
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import tools
import logomaker
import os
import pickle
import shutil
import json
import argparse


sys.path.append('../../Enhancer')  
from model.model import ConvNetDeep, DanQ, ExplaiNN,ConvNetDeep2, ExplaiNN2, ExplaiNN3
import scripts

# Argument parser setup
parser = argparse.ArgumentParser(description="Script for running ExplaiNN model with specified parameters.")

# Required arguments without defaults
parser.add_argument('--num_class', type=int, required=True, help="Number of output classes from the model")
parser.add_argument('--seed', type=int, required=True, help="Random seed")
parser.add_argument('--batch', type=int, required=True, help="Batch size")
parser.add_argument('--num_cnns', type=int, required=True, help="Number of CNNs in the model")
parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for the model")
parser.add_argument('--result_dir', type=str, required=True, help="Directory to save results")
parser.add_argument('--feature_list', type=json.loads, required=True, help="List of features model will predict")
parser.add_argument('--target_labels', type=json.loads, required=True, help="Name for features model will predict")


# Optional arguments with default values
parser.add_argument('--input_data_dir', type=str, default='/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_input_data.csv', help="Input data directory")
parser.add_argument('--filter_size', type=int, default=19, help="Filter size for the model")
parser.add_argument('--upper_bound', type=float, default=0.25, help="Upper bound for cutoff")
parser.add_argument('--save_cutoff_plot', type=bool, default=True, help="Whether to save the cutoff plot")
parser.add_argument('--meme_package_dir', type=str, default='/pmglocal/ty2514/meme', help="Directory for the MEME suite")
parser.add_argument('--jaspar_meme_file_dir', type=str, default='/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_meme.meme', help="Path to JASPAR meme file")
parser.add_argument('--tf_cluster_db_dir', type=str, default='/pmglocal/ty2514/Enhancer/motif-clustering/JASPAR2024_mus_musculus_non-redundant_results/metadata.tsv', help="Directory for TF cluster database")
#parser.add_argument('--tomtom_result_dir', type=str, default=os.path.join('/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_both_results', 'tomtom_results'), help="Directory for Tomtom results")

# Parse arguments
args = parser.parse_args()
args.tomtom_result_dir = os.path.join(args.result_dir, 'tomtom_results')

# Update the PATH environment variable
os.environ['PATH'] = os.path.join(args.meme_package_dir, 'bin') + ':' + os.path.join(args.meme_package_dir, 'libexec', 'meme-5.5.5') + ':' + os.environ['PATH']
# Check if 'tomtom' is in the PATH using 'which tomtom'
try:
    result = subprocess.run(['which', 'tomtom'], capture_output=True, text=True, check=True)
    print(f"Tomtom found at: {result.stdout.strip()}")
except subprocess.CalledProcessError:
    print("Error: 'tomtom' not found in the PATH.")
    sys.exit(1)  # Stop further execution if tomtom is not found

print('--------------------------------------------------------')
print('***********************   1/8   ************************')
print('--------------------------------------------------------')
print('Attaching device to the gpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model without moving it to the device yet
explainn = ExplaiNN3(num_cnns=args.num_cnns, input_length=608, num_classes=args.num_class,
                     filter_size=args.filter_size, num_fc=2, pool_size=7, pool_stride=7,
                     drop_out=0.3, weight_path=None)  # Training

file_list = glob.glob(f'{args.result_dir}/best_pearson*.pth')
print('Loading weight from following weight file to the model: ')
print(file_list)

if len(file_list) > 0:
    weight_file = file_list[0]
else:
    raise FileExistsError("Best pearson Model file not exist")

print('\n')
# Load the model weights conditionally based on GPU availability
state_dict = torch.load(weight_file, map_location=device)  # Load to the appropriate device

# Check if the state_dict contains keys prefixed with 'module.'
if any(key.startswith('module.') for key in state_dict.keys()):
    # If the state_dict was saved from a DataParallel model, remove the 'module.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    state_dict = new_state_dict

# Load the modified state_dict into the model
explainn.load_state_dict(state_dict)

# Move the model to the appropriate device after loading the weights
explainn.to(device)
explainn.eval()
print('\n')


print('--------------------------------------------------------')
print('***********************   2/8   ************************')
print('--------------------------------------------------------')
print('Reading input data for retreiving predictions')
# Load dataset as a pandas dataframe
df = pd.read_csv(args.input_data_dir)
# Prepare features and labels:
# --(Features): Transform all sequences into one-hot encodings
# --(Labels): Use GFP+ and GFP- as labels
dataset = EnhancerDataset(df, feature_list=args.feature_list, scale_mode = 'none')
# Prepare dataloader
dataset = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)

# Running get_explainn_predictions function to get predictions and true labels for all sequences in the given data loader
predictions, labels = interpretation.get_explainn_predictions(dataset, explainn, device, isSigmoid=False)
print("Get prediction using trained model!")
print(f'Prediction shape is {predictions.shape}')

"""Now filter out low confident predictions"""
print('\n')
print('--------------------------------------------------------')
print('***********************   3/8   ************************')
print('--------------------------------------------------------')

print("Now selecting high confident predictions!")
# Calculate absolute residuals
residuals = np.abs(labels - predictions)

# Define the upper bound of residuals
print(f'Using Bound = {args.upper_bound} as a cutoff to select high confident predictions.')

# Create a mask for filtering out samples with low confident precition (abs(residual) > upper_bound)
mask = (residuals <= args.upper_bound).all(axis=1)

# Get sequences and labels from dataset
data_inp = []
data_out = []
# Iterate over the DataLoader
for batch_features, batch_labels in dataset:
    data_inp.append(batch_features)
    data_out.append(batch_labels)
# Concatenate all the batches into single tensors
data_inp = torch.cat(data_inp, dim=0)
data_out = torch.cat(data_out, dim=0)

# Use the mask to filter the predictions and labels
print(f'Total number of input samples: {len(data_inp)}')
data_inp = data_inp[mask]
data_out = data_out[mask]

print(f'Number of input samples with high confident prediction: {len(data_inp)}')

# Create new dataloader with filtered high confident samples
dataset = torch.utils.data.TensorDataset(data_inp, data_out)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=args.batch, shuffle=False,)


if args.save_cutoff_plot:
    print("Plot histogram of residuals with adjusted cutoff lines")
    # Plot histogram of residuals with adjusted cutoff lines
    # 1. Compute residuals (not absolute)
    residuals = labels - predictions

    # 2. Define bounds for plotting
    lower_bound = -args.upper_bound
    print(f'Lower bound: {lower_bound}')

    # 3. Plotting the histograms for each feature
    num_labels = predictions.shape[1] if len(predictions.shape) > 1 else 1
    fig, axes = plt.subplots(nrows=num_labels, ncols=1, figsize=(5, 4*num_labels))
    
    # Ensure `axes` is iterable (even if it's a single Axes object)
    if num_labels == 1:
        axes = [axes]
    
    fig.tight_layout(pad=3.0)

    # Title for the entire figure
    fig.suptitle('Histogram of Residuals for Each Feature with Confidence Cutoffs', fontsize=16, y=1.02)

    for i in range(num_labels):
        print(f'Plot for label {i}')
        ax = axes[i]
        if num_labels > 1:
            residual_col = residuals[:, i]
        else:
            residual_col = residuals

        color = 'skyblue' 
        ax.hist(residual_col, bins=50, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(x=lower_bound, color='red', linestyle='--', label=f'Lower Cutoff ({lower_bound})')
        ax.axvline(x=args.upper_bound, color='green', linestyle='--', label=f'Upper Cutoff ({args.upper_bound})')
        ax.set_title(f'Feature {i+1} Residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)

    plt.savefig(os.path.join(args.result_dir, 'sample_cutoff_plot.png'), dpi=300, bbox_inches='tight')
    print("Sample cutoff plot saved")
    print('\n')

print('--------------------------------------------------------')
print('***********************   4/8   ************************')
print('--------------------------------------------------------')
print('Calculating activations from each layer of the model if not calculated before!')
# Get weights from activation layer
activation_file_dir = os.path.join(args.result_dir, 'Model_Activations.npy')
pwm_file_dir = os.path.join(args.result_dir, 'Model_PWM.npy')
if os.path.exists(activation_file_dir):
    print(f"Model Activation File Exists. Reading Activation and PWM files.")
    activations = np.load(activation_file_dir)
    print(f"Activations have shape: {activations.shape}")
    pwms = np.load(pwm_file_dir)
    print(f"PWMs have shape: {pwms.shape}")
else:
    print(f"Model Activation File NOT Exists. Calculating Activation and PWM files.")
    activations = interpretation.get_explainn_unit_activations(data_loader, explainn, device)
    print(f"Activations Calculation Done! Activations have shape: {activations.shape}")
    print('\n')
    # Save activations
    np.save(activation_file_dir, activations)

    # Get torch,  one-hot encoding sequences from data_loader
    sequences = []
    # Iterate over the DataLoader
    for batch in data_loader:
        batch_sequences = batch[0]  
        sequences.append(batch_sequences)
    sequences = torch.cat(sequences, dim=0)

    # Define filter size. This parameter should be consistent with filter_size used in ExplaiNN
    print("Calculating PWMs")
    pwms = interpretation.get_pwms_explainn(activations, sequences, args.filter_size)
    print(f"Calculation Done! PWMs have shape: {pwms.shape}")
    print('\n')
    # Save pwms
    np.save(pwm_file_dir, pwms)

print('Saving PWM File as a MEME file')
pwm_meme_file_dir = os.path.join(args.result_dir, 'Model_PWM.meme')
interpretation.pwm_to_meme(pwms, pwm_meme_file_dir)

print('--------------------------------------------------------')
print('***********************   5/8   ************************')
print('--------------------------------------------------------')
print('Running tomtom comparison!')
tomtom_result_dir = args.tomtom_result_dir
jaspar_meme_file_dir = args.jaspar_meme_file_dir
# Check if the output directory exists
if os.path.exists(tomtom_result_dir):
    # Optionally delete the existing directory
    shutil.rmtree(tomtom_result_dir)
    print(f"Deleted existing directory: {tomtom_result_dir}")

# Command to run tomtom
tomtom_command = [
        'tomtom', pwm_meme_file_dir, jaspar_meme_file_dir, 
        '-o', tomtom_result_dir, '--dist', 'kullback', '--min-overlap', '0', '--thresh', '0.05'
    ]

# Run the tomtom command
try:
    subprocess.run(tomtom_command, check=True)
    print(f"Tomtom successfully executed, results stored in {args.tomtom_result_dir}")
except subprocess.CalledProcessError as e:
    print(f"Error running tomtom: {e}")
    sys.exit(1)
print('\n')

print('--------------------------------------------------------')
print('***********************   6/8   ************************')
print('--------------------------------------------------------')
print('Annotate Each Filter!')
# Annotate filter by filter -> jasper TF -> TF cluster
# Find directory of tomtom results 
tomtom_comparison_result_dir = find_tsv_file_path(args.tomtom_result_dir)
# Load JASPER Clustering result (jasper TF -> TF cluster)
cluster_results = pd.read_csv(args.tf_cluster_db_dir,sep="\t",comment="#")
# Load tomtom comparison results between filter PWM and JASPER mus muscus TF database (filter -> jasper TF)
tomtom_results = pd.read_csv(tomtom_comparison_result_dir,sep="\t",comment="#")
filters_with_min_q = tomtom_results.groupby('Query_ID').min()["q-value"]
tomtom_results = tomtom_results[["Target_ID", "Query_ID", "q-value"]]
tomtom_results = tomtom_results[tomtom_results["q-value"]<0.05]
# Create dictionaries to map motif id to cluster; tf_name; family_name
motif_to_cluster = cluster_results.set_index('motif_id')['cluster'].to_dict()
motif_to_tf_name = cluster_results.set_index('motif_id')['tf_name'].to_dict()
motif_to_family_name = cluster_results.set_index('motif_id')['family_name'].to_dict()
filters = tomtom_results["Query_ID"].unique()

# Assuming `annotation` is already populated
annotation_data = []

for f in filters:
    t = tomtom_results[tomtom_results["Query_ID"] == f]
    target_id = t["Target_ID"]

    if len(target_id) > 5:
        target_id = target_id[:5]

    # Join Unique annotations by '/'
    cluster = "/".join({motif_to_cluster[i]: i for i in target_id.values})
    tf_name = "/".join({motif_to_tf_name[i]: i for i in target_id.values})
    family_name = "/".join({motif_to_family_name[i]: i for i in target_id.values})

    # Append the data to the list
    annotation_data.append({
        'filter': f,
        'cluster': cluster,
        'tf_name': tf_name,
        'family_name': family_name
    })

# Create a DataFrame from the collected data
annotation_df = pd.DataFrame(annotation_data)

weights_dir = os.path.join(args.result_dir, 'weights')
os.makedirs(weights_dir, exist_ok = True)
annotation_df.to_csv(os.path.join(weights_dir,'filter_annotation.csv'), index=False)

print('--------------------------------------------------------')
print('***********************   7/8   ************************')
print('--------------------------------------------------------')
print('Get Weight of Each Filter!')
weights = explainn.final.weight.detach().cpu().numpy()
print(f'weight_df has shape: {weights.shape} (number of labels, number of fileters)')
filters = ["f"+str(i) for i in range(args.num_cnns)]
for index,row in annotation_df.iterrows():
    filter = row['filter']
    split_string = filter.split('filter', 1)
    # change 'filter{i}' to 'f{i}'. e.g. filter20 -> f20
    new_filter_name = 'f' + split_string[1].strip()
    # Check if new_filter_name is in the filters list
    if new_filter_name in filters:
        # Find the index of the element to be replaced
        index_to_replace = filters.index(new_filter_name)
        # Replace the element in the filters list
        filters[index_to_replace] = f"{row['cluster']}({row['tf_name']})-{new_filter_name}"

weight_df = pd.DataFrame(weights, args.target_labels, columns=filters)
weight_file_dir = os.path.join(weights_dir, 'filter_weights.csv')
# Save the DataFrame to a CSV file
weight_df.to_csv(weight_file_dir, index=True)  # Set index=True if you want to save the index

print(f'Weights saved to {weight_file_dir}')
print('\n')
print('Plotting weights of different filters')
# Plotting weight for each filter using bar plot
plot_filter_weight(weight_df, weights_dir)
# Plotting weight only for filters with TF annotattions using bar plot
annotated_weight_df = weight_df.loc[:, weight_df.columns.str.contains('-')]
plot_filter_weight(annotated_weight_df, weights_dir)

print('--------------------------------------------------------')
print('***********************   8/8   ************************')
print('--------------------------------------------------------')
print('Calculating Unit Importance For Each Label')
importance_result_dir = os.path.join(args.result_dir, 'importance')
os.makedirs(importance_result_dir, exist_ok=True)
importance_dictionary_file = os.path.join(importance_result_dir, 'importance_dict.pkl')

# Check if the file exists
if os.path.exists(importance_dictionary_file):
    with open(importance_dictionary_file, 'rb') as f:
        importance_dict = pickle.load(f)
    print(f"Importance dictionary already exists. Loaded the importance dictionary from {importance_dictionary_file}.")
else:
    importance_dict = {}
    for label in args.target_labels:
        importance_dict[label] = []
    # Use tqdm to track progress over the loop
    for unit_index in tqdm(range(args.num_cnns), desc="Processing units"):
        # Calculate unit importance for the current unit
        unit_outputs = interpretation.get_explainn_unit_outputs(data_loader, explainn, device)
        importance = interpretation.get_specific_unit_importance(activations, explainn, unit_outputs, unit_index, args.target_labels)

        # Store the importance for each label
        for label in args.target_labels:
            importance_dict[label].append(importance[label])

    with open(importance_dictionary_file, 'wb') as f:
        pickle.dump(importance_dict, f)

print(f'Importance dictionary saved to {importance_dictionary_file}')
print('\n')
print('Plotting Unit Importance Plot')
unit_names = list(weight_df.columns)
for label in args.target_labels:
    _, _ = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=True, num_tf_plotted=10)
    sorted_filters, sorted_values = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=False, num_tf_plotted=False)
    # Save sorted_names to a plain text file
    #sorted_tf_order_file = os.path.join(importance_result_dir, f'{label}_sorted_tf_order.txt')
    #with open(sorted_tf_order_file, 'w') as f:
    #    for name in sorted_filters:
    #        f.write(f"{name}\n")

    # Save sorted_filters and sorted_values to a CSV file
    sorted_tf_order_file = os.path.join(importance_result_dir, f'{label}_sorted_tf_order.csv')
    sorted_df = pd.DataFrame({'tf_name': sorted_filters, 'importance_score': sorted_values})
    sorted_df.to_csv(sorted_tf_order_file, index=False)

print('Finish!')

