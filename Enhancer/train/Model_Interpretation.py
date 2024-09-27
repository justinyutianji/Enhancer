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

sys.path.append('../../Enhancer')  
from model.model import ConvNetDeep, DanQ, ExplaiNN,ConvNetDeep2, ExplaiNN2, ExplaiNN3
import scripts

seed = 42 
batch = 322
num_cnns = 90
learning_rate= 3e-4
num_cnns = 90
filter_size = 19
result_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_both_results'
input_data_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv'
upper_bound = 0.25
save_cutoff_plot = True
meme_package_dir = '/pmglocal/ty2514/meme'
jaspar_meme_file_dir = '/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_meme.meme'
tf_cluster_db_dir = '/pmglocal/ty2514/Enhancer/motif-clustering/JASPAR2024_mus_musculus_non-redundant_results/metadata.tsv'
tomtom_result_dir = os.path.join(result_dir, 'tomtom_results')
target_labels = ['GFP+','GFP-']
feature_list= ['G+','G-']
# Update the PATH environment variable
os.environ['PATH'] = os.path.join(meme_package_dir, 'bin') + ':' + os.path.join(meme_package_dir, 'libexec', 'meme-5.5.5') + ':' + os.environ['PATH']
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
explainn = ExplaiNN3(num_cnns=num_cnns, input_length=608, num_classes=2,
                     filter_size=filter_size, num_fc=2, pool_size=7, pool_stride=7,
                     drop_out=0.3, weight_path=None)  # Training

file_list = glob.glob(f'{result_dir}/best_r2*.pth')
print('Loading weight from following weight file to the model: ')
print(file_list)

if len(file_list) > 0:
    weight_file = file_list[0]
else:
    raise FileExistsError("Best r2 Model file not exist")

print('\n')
# Load the model weights conditionally based on GPU availability
if torch.cuda.is_available():
    explainn.load_state_dict(torch.load(weight_file))
    print('explainn loaded on GPU')
else:
    explainn.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    print('explainn loaded on CPU')

# Move the model to the appropriate device after loading the weights
explainn.to(device)
explainn.eval()
print('\n')


print('--------------------------------------------------------')
print('***********************   2/8   ************************')
print('--------------------------------------------------------')
print('Reading input data for retreiving predictions')
# Load dataset as a pandas dataframe
df = pd.read_csv(input_data_dir)
# Prepare features and labels:
# --(Features): Transform all sequences into one-hot encodings
# --(Labels): Use GFP+ and GFP- as labels
dataset = EnhancerDataset(df, feature_list=feature_list, scale_mode = 'none')
# Prepare dataloader
dataset = DataLoader(dataset=dataset, batch_size=batch, shuffle=False)

# Running get_explainn_predictions function to get predictions and true labels for all sequences in the given data loader
predictions, labels = interpretation.get_explainn_predictions(dataset, explainn, device, isSigmoid=False)
print("Get prediction using trained model!")

"""Now filter out low confident predictions"""
print('\n')
print('--------------------------------------------------------')
print('***********************   3/8   ************************')
print('--------------------------------------------------------')

print("Now selecting high confident predictions!")
# Calculate absolute residuals
residuals = np.abs(labels - predictions)

# Define the upper bound of residuals
print(f'Using Bound = {upper_bound} as a cutoff to select high confident predictions.')

# Create a mask for filtering out samples with low confident precition (abs(residual) > upper_bound)
mask = (residuals <= upper_bound).all(axis=1)

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
                                          batch_size=batch, shuffle=False,)


if save_cutoff_plot == True:
    print("Plot histogram of residuals with adjusted cutoff lines")
    # Plot histogram of residuals with adjusted cutoff lines
    # 1. Compute residuals (not absolute)
    residuals = labels - predictions

    # 2. Define bounds for plotting
    lower_bound = -upper_bound

    # 3.Plotting the histograms for each feature
    fig, axes = plt.subplots(nrows=predictions.shape[1], ncols=1, figsize=(5, 4*predictions.shape[1]))
    fig.tight_layout(pad=3.0)

    # Title for the entire figure
    fig.suptitle('Histogram of Residuals for Each Feature with Confidence Cutoffs', fontsize=16, y=1.02)

    for i in range(predictions.shape[1]):
        ax = axes[i]
        ax.hist(residuals[:, i], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=lower_bound, color='red', linestyle='--', label=f'Lower Cutoff (-0.25)')
        ax.axvline(x=upper_bound, color='green', linestyle='--', label=f'Upper Cutoff (+0.25)')
        ax.set_title(f'Feature {i+1} Residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)

    plt.savefig(os.path.join(result_dir, 'sample_cutoff_plot.png'), dpi=300, bbox_inches='tight')
    print("Sample cutoff plot saved")
    print('\n')

print('--------------------------------------------------------')
print('***********************   4/8   ************************')
print('--------------------------------------------------------')
print('Calculating activations from each layer of the model if not calculated before!')
# Get weights from activation layer
activation_file_dir = os.path.join(result_dir, 'Model_Activations.npy')
pwm_file_dir = os.path.join(result_dir, 'Model_PWM.npy')
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
    pwms = interpretation.get_pwms_explainn(activations, sequences, filter_size)
    print(f"Calculation Done! PWMs have shape: {pwms.shape}")
    print('\n')
    # Save pwms
    np.save(pwm_file_dir, pwms)

print('Saving PWM File as a MEME file')
pwm_meme_file_dir = os.path.join(result_dir, 'Model_PWM.meme')
interpretation.pwm_to_meme(pwms, pwm_meme_file_dir)

print('--------------------------------------------------------')
print('***********************   5/8   ************************')
print('--------------------------------------------------------')
print('Running tomtom comparison!')

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
    print(f"Tomtom successfully executed, results stored in {tomtom_result_dir}")
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
tomtom_comparison_result_dir = find_tsv_file_path(tomtom_result_dir)
# Load JASPER Clustering result (jasper TF -> TF cluster)
cluster_results = pd.read_csv(tf_cluster_db_dir,sep="\t",comment="#")
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

weights_dir = os.path.join(result_dir, 'weights')
os.makedirs(weights_dir, exist_ok = True)
annotation_df.to_csv(os.path.join(weights_dir,'filter_annotation.csv'), index=False)

print('--------------------------------------------------------')
print('***********************   7/8   ************************')
print('--------------------------------------------------------')
print('Get Weight of Each Filter!')
weights = explainn.final.weight.detach().cpu().numpy()
print(f'weight_df has shape: {weights.shape} (number of labels, number of fileters)')
filters = ["f"+str(i) for i in range(num_cnns)]
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

weight_df = pd.DataFrame(weights, target_labels, columns=filters)
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
importance_result_dir = os.path.join(result_dir, 'importance')
os.makedirs(importance_result_dir, exist_ok=True)
importance_dictionary_file = os.path.join(importance_result_dir, 'importance_dict.pkl')

# Check if the file exists
if os.path.exists(importance_dictionary_file):
    with open(importance_dictionary_file, 'rb') as f:
        importance_dict = pickle.load(f)
    print(f"Importance dictionary already exists. Loaded the importance dictionary from {importance_dict}.")
else:
    importance_dict = {}
    for label in target_labels:
        importance_dict[label] = []
    # Use tqdm to track progress over the loop
    for unit_index in tqdm(range(num_cnns), desc="Processing units"):
        # Calculate unit importance for the current unit
        unit_outputs = interpretation.get_explainn_unit_outputs(data_loader, explainn, device)
        importance = interpretation.get_specific_unit_importance(activations, explainn, unit_outputs, unit_index, target_labels)

        # Store the importance for each label
        for label in target_labels:
            importance_dict[label].append(importance[label])

    with open(importance_dictionary_file, 'wb') as f:
        pickle.dump(importance_dict, f)

print(f'Importance dictionary saved to {importance_dictionary_file}')
print('\n')
print('Plotting Unit Importance Plot')
unit_names = list(weight_df.columns)
for label in target_labels:
    _ = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=True, num_tf_plotted=10)
    sorted_filters = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=False, num_tf_plotted=False)
    # Save sorted_names to a plain text file
    sorted_tf_order_file = os.path.join(importance_result_dir, f'{label}_sorted_tf_order.txt')
    with open(sorted_tf_order_file, 'w') as f:
        for name in sorted_filters:
            f.write(f"{name}\n")

print('Finish!')

