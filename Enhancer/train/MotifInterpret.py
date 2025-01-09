import torch
from torch.utils.data import DataLoader
from utils import EnhancerDataset, find_tsv_file_path, EnhancerDatasetWithID
from tools import plot_filter_weight,plot_unit_importance,recursive_seqlets,p_value
import numpy as np
import numpy
import math
import pandas as pd
import sys
import interpretation
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import numpy
import pandas
import math
from tangermeme.annotate import annotate_seqlets, count_annotations
import torch
from utils import EnhancerDataset, dna_one_hot, split_dataset,EnhancerDatasetWithID
import pickle
from tangermeme.io import read_meme
import gzip
import argparse
import json


sys.path.append('../../Enhancer')  
from model.model import ConvNetDeep, DanQ, ExplaiNN,ConvNetDeep2, ExplaiNN2, ExplaiNN3

# Argument parser setup
parser = argparse.ArgumentParser(description="Script for Annotating ExplaiNN model with specified parameters.")

# Required arguments without defaults
parser.add_argument('--num_cnns', type=int, required=True, help="Number of CNNs in the model")
parser.add_argument('--rep', type=int, required=True, help="Replication number of Explainn Models")

# Optional arguments with default values
parser.add_argument('--batch', type=int, default=168, help="Batch size")
parser.add_argument('--filter_size', type=int, default=19, help="Filter size for the model")
parser.add_argument('--num_class', type=int, default=2,  help="Number of output classes from the model (default: 2)")
parser.add_argument('--feature_list', type=json.loads, default = '["G+","G-"]', help="List of features model will predict")
parser.add_argument('--target_labels', type=json.loads, default = '["GFP+","GFP-"]', help="Name for features model will predict")
parser.add_argument('--upper_bound', type=float, default=0.1, help="Upper bound for cutoff")
# Parse arguments
args = parser.parse_args()

result_dir = f'/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_G+G-_Merged_Pred/{args.num_cnns}NN_Rep{args.rep}'
output_dir = f'/pmglocal/ty2514/Enhancer/Enhancer/data/Activations/ExplaiNN_{args.num_cnns}CNN_Rep{args.rep}'
meme_file = "/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_with_id.meme"
input_data_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_merged_data.csv'
#parser.add_argument('--result_dir', type=str, required=True, help="Directory to save results")
#parser.add_argument('--input_data_dir', type=str, default='/pmglocal/ty2514/Enhancer/Enhancer/data/filtered_merged_data.csv', help="Input data directory")
#parser.add_argument('--jaspar_meme_file_dir', type=str, default='/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_meme.meme', help="Path to JASPAR meme file")

print('--------------------------------------------------------')
print('***********************   1/7   ************************')
print('--------------------------------------------------------')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Attaching device to the {device}')

# Initialize the model without moving it to the device yet
explainn = ExplaiNN3(num_cnns=args.num_cnns, input_length=608, num_classes=args.num_class,
                     filter_size=args.filter_size, num_fc=2, pool_size=7, pool_stride=7,
                     drop_out=0.3, weight_path=None)  # Training

file_list = glob.glob(f'{result_dir}/best_pearson*.pth')
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
print('***********************   2/7   ************************')
print('--------------------------------------------------------')
print('Reading input data for retreiving predictions')
# Load dataset as a pandas dataframe
df = pd.read_csv(input_data_dir)
# Prepare features and labels:
# --(Features): Transform all sequences into one-hot encodings
# --(Labels): Use GFP+ and GFP- as labels
dataset = EnhancerDatasetWithID(df, feature_list=args.feature_list, scale_mode = 'none')
# Prepare dataloader
dataset = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)

# Running get_explainn_predictions function to get predictions and true labels for all sequences in the given data loader
predictions, labels = interpretation.get_explainn_predictions(dataset, explainn, device, isSigmoid=False)
print("Get prediction using trained model!")
print(f'Prediction shape is {predictions.shape}')

"""Now filter out low confident predictions"""
print('\n')
print('--------------------------------------------------------')
print('***********************   3/7   ************************')
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
data_ids = []
# Iterate over the DataLoader
for batch_features, batch_labels, batch_ids in dataset:
    data_inp.append(batch_features)
    data_out.append(batch_labels)
    # Convert tuple of IDs to a NumPy array
    batch_ids_array = np.array(batch_ids)
    data_ids = np.append(data_ids, batch_ids_array)

print(f'data_ids have length: {len(data_ids)}')
# Concatenate all the batches into single tensors
data_inp = torch.cat(data_inp, dim=0)
data_out = torch.cat(data_out, dim=0)

# Use the mask to filter the predictions and labels
print(f'Total number of input samples: {len(data_inp)}')
data_inp = data_inp[mask]
data_out = data_out[mask]
data_ids = [id_ for id_, m in zip(data_ids, mask) if m]

print(f'Number of input samples with high confident prediction: {len(data_inp)}')

# Create new dataloader with filtered high confident samples
high_conf_dataset = torch.utils.data.TensorDataset(data_inp, data_out)
high_conf_data_loader = torch.utils.data.DataLoader(dataset=high_conf_dataset,
                                          batch_size=args.batch, shuffle=False,)

print('--------------------------------------------------------')
print('***********************   4/7   ************************')
print('--------------------------------------------------------')
print('Calculating activations from each layer of the model if not calculated before!')
# Get weights from activation layer
activation_file_dir = os.path.join(output_dir, 'Model_Activations.npy')
os.makedirs(output_dir, exist_ok=True)

#pwm_file_dir = os.path.join(output_dir, 'Model_PWM.npy')
if os.path.exists(activation_file_dir):
    print(f"Model Activation File Exists. Reading Activation and PWM files.")
    activations = np.load(activation_file_dir)
    print(f"Activations have shape: {activations.shape}")
    #pwms = np.load(pwm_file_dir)
    #print(f"PWMs have shape: {pwms.shape}")
else:
    print(f"Model Activation File NOT Exists. Calculating Activation and PWM files.")
    activations = interpretation.get_explainn_unit_activations(high_conf_data_loader, explainn, device)
    print(f"Activations Calculation Done! Activations have shape: {activations.shape}")
    print('\n')
    # Save activations
    np.save(activation_file_dir, activations)

print('--------------------------------------------------------')
print('***********************   5/7   ************************')
print('--------------------------------------------------------')
print('Get Weight of Each Filter!')
weights = explainn.final.weight.detach().cpu().numpy()
print(f'weight_df has shape: {weights.shape} (number of labels, number of fileters)')
filters = ["f"+str(i) for i in range(args.num_cnns)]

weight_df = pd.DataFrame(weights, args.target_labels, columns=filters)
weight_file_dir = os.path.join(output_dir, 'filter_weights.csv')
# Save the DataFrame to a CSV file
weight_df.to_csv(weight_file_dir, index=True)  # Set index=True if you want to save the index

print(f'Weights saved to {weight_file_dir}')
print('\n')

print('--------------------------------------------------------')
print('***********************   6/7   ************************')
print('--------------------------------------------------------')
print('Finding Activated Seqlets for EACH FILTER and Annotate each filter')
# Get torch,  one-hot encoding sequences from data_loader
sequences = []

# Iterate over the DataLoader
for batch in high_conf_data_loader:
    batch_sequences = batch[0]  
    sequences.append(batch_sequences)
sequences = torch.cat(sequences, dim=0)
print(f'Sequences used for calculating activations have shape {sequences.shape}')

# activated_indexes would be used to store indexes of activated sequence in each filter
activated_indexes = {}
# motif_count_matrix would store N arrays of motif counts. Each array would have length 234. N is numbe of filters
motif_count_matrix = []
# filter_names_dict would map filter (int) to an array of significant (p_value < 0.01) motif annotations
filter_names_dict = {}

X = sequences
motifs = read_meme(meme_file)
motif_names = np.array(list(motifs.keys()))
motif_names = [name.split(" ", 1)[1] if " " in name else name for name in motif_names]
motif_names = np.array(motif_names)
#motif_names = [name.split(" ", 1)[1].rsplit(".", 1)[-1] for name in motif_names]

for f in tqdm(range(args.num_cnns), desc="Annotating Filters"):
    print(f"Annotating Filter {f}")
    seqlets = recursive_seqlets(activations[:,f,:], seqlet_len=19, threshold=0.01)
    # Get indexes of sequences contain at least one significant seqlet for importance score calculation
    activated_sample_ids = np.array(seqlets['example_idx'])
    activated_sample_ids = np.unique(activated_sample_ids)
    activated_indexes[f] = (list(activated_sample_ids))

    # Annotate each seqlet by the most likely motif id
    motif_idxs = annotate_seqlets(X, seqlets, meme_file)[0][:, 0]
    # Count number of seqlets in each sample
    y = count_annotations((seqlets['example_idx'], motif_idxs),shape = (X.shape[0],224))
    y_sum = y.sum(dim=0)
    motif_count_matrix.append(np.array(y_sum))

    # Calculate the CDF
    values, counts = np.unique(y_sum, return_counts=True)
    cdf = np.cumsum(counts) / len(y_sum)
    p_values = p_value(y_sum, values, cdf)
    sig_index = np.where(p_values <= 0.01)[0]
    if sig_index.size == 0:
        filter_names = []
        filter_names_dict[f] = filter_names
    else:
        #print("Original sig_index:", sig_index)
        sig_p_values = p_values[sig_index]
        #print("Corresponding p-values:", sig_p_values)
        sorted_order = np.argsort(sig_p_values)
        #print("Sorting order:", sorted_order)
        sorted_sig_index = sig_index[sorted_order]
        #print("Sorted sig_index:", sorted_sig_index)

        names = motif_names[sorted_sig_index]
        print(f'Filter {f} can be annotated as {names}')
        filter_names_dict[f] = names

### Save filter motif count matrix into a pandas data frame and a csv file
filter_numbers = range(len(motif_count_matrix))  # Filter numbers as a column

# Convert motif_count_matrix to a DataFrame
filter_motif_count_df = pd.DataFrame(
    data=motif_count_matrix,  # Rows
    columns=motif_names  # Columns
)

# Add Filter_Number as a column
filter_motif_count_df["Filter_Number"] = filter_numbers
filter_motif_count_file_dir = os.path.join(output_dir, 'filter_motif_counts.csv')
filter_motif_count_df.to_csv(filter_motif_count_file_dir)

### Save filter to motif name dictionary to a pickle file
filter2motif_dict_dir = os.path.join(output_dir, 'filter2motif_dictionary.pkl')
# Save the dictionary to a pickle file
with open(filter2motif_dict_dir, "wb") as pickle_file:
    pickle.dump(filter_names_dict, pickle_file)

print('--------------------------------------------------------')
print('***********************   7/7   ************************')
print('--------------------------------------------------------')
print('Calculating Unit Importance For Each Label')
importance_result_dir = os.path.join(output_dir, 'importance')
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
    unit_outputs = interpretation.get_explainn_unit_outputs(high_conf_data_loader, explainn, device)
    for unit_index in tqdm(range(args.num_cnns), desc="Processing units"):
        # Calculate unit importance for the current unit
        importance = interpretation.get_specific_unit_importance_seqlet(activated_indexes, explainn, unit_outputs, unit_index, args.target_labels)

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
    #_, _, _ = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=True, num_tf_plotted=10)
    sorted_filters, sorted_values, sorted_unit_samples = plot_unit_importance(importance_dict[label], unit_names, label, dir_save_plot = importance_result_dir, annotated_filter_only=False, num_tf_plotted=False)
    # Save sorted_names to a plain text file
    #sorted_tf_order_file = os.path.join(importance_result_dir, f'{label}_sorted_tf_order.txt')
    #with open(sorted_tf_order_file, 'w') as f:
    #    for name in sorted_filters:
    #        f.write(f"{name}\n")

    # Save sorted_filters and sorted_values to a CSV file
    sorted_tf_order_file = os.path.join(importance_result_dir, f'{label}_sorted_tf_order.csv')
    sorted_df = pd.DataFrame({'tf_name': sorted_filters, 'importance_score': sorted_values,'activated_samples': sorted_unit_samples})
    sorted_df.to_csv(sorted_tf_order_file, index=False)

print('Finish!')


