import torch
from torch.utils.data import DataLoader
from utils import EnhancerDataset, split_dataset, train_model, regression_model_plot, plot_filter_weight, find_tsv_file_path
import numpy as np
import pandas as pd
import subprocess
import torch.nn as nn
import torch.nn.modules.activation as activation
import sys
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import interpretation
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import tools
import logomaker
import os
import pickle

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
tomtom_result_dir = os.path.join(result_dir, 'tomtom_results')

# Update the PATH environment variable
os.environ['PATH'] = os.path.join(meme_package_dir, 'bin') + ':' + os.path.join(meme_package_dir, 'libexec', 'meme-5.5.5') + ':' + os.environ['PATH']
# Check if 'tomtom' is in the PATH using 'which tomtom'
try:
    result = subprocess.run(['which', 'tomtom'], capture_output=True, text=True, check=True)
    print(f"Tomtom found at: {result.stdout.strip()}")
except subprocess.CalledProcessError:
    print("Error: 'tomtom' not found in the PATH.")
    sys.exit(1)  # Stop further execution if tomtom is not found

# Create the tomtom_output directory if it doesn't exist
os.makedirs(tomtom_result_dir, exist_ok=True)

print('--------------------------------------------------------')
print('***********************   1/5   ************************')
print('--------------------------------------------------------')
print('Attaching device to the gpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
explainn = ExplaiNN3(num_cnns = num_cnns, input_length = 608, num_classes = 2, 
                 filter_size = filter_size, num_fc=2, pool_size=7, pool_stride=7, 
                 drop_out = 0.3, weight_path = None)# Training
explainn.to(device)  

file_list = glob.glob(f'{result_dir}/best_r2*.pth')
print('Loading weight from following weight file to the model: ')
print(file_list)

if len(file_list) > 0:
    weight_file = file_list[0]
else:
    raise FileExistsError("Best r2 Model file not exist")
print('\n')
explainn.load_state_dict(torch.load(weight_file))
explainn.eval()
print('\n')

print('--------------------------------------------------------')
print('***********************   2/5   ************************')
print('--------------------------------------------------------')
print('Reading input data for retreiving predictions')
# Load dataset as a pandas dataframe
df = pd.read_csv(input_data_dir)
# Prepare features and labels:
# --(Features): Transform all sequences into one-hot encodings
# --(Labels): Use GFP+ and GFP- as labels
dataset = EnhancerDataset(df, feature_list=['G+','G-'], scale_mode = 'none')
# Prepare dataloader
dataset = DataLoader(dataset=dataset, batch_size=batch, shuffle=False)

# Running get_explainn_predictions function to get predictions and true labels for all sequences in the given data loader
predictions, labels = interpretation.get_explainn_predictions(dataset, explainn, device, isSigmoid=False)
print("Get prediction using trained model!")

"""Now filter out low confident predictions"""
print('\n')
print('--------------------------------------------------------')
print('***********************   3/5   ************************')
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
print(data_inp.shape)
print(data_out.shape)

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
print('***********************   4/5   ************************')
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
print('***********************   5/5   ************************')
print('--------------------------------------------------------')
print('Running tomtom comparison!')

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
print('***********************   6/5   ************************')
print('--------------------------------------------------------')
print('Annotate Each Filter!')
tomtom_comparison_result_dir = find_tsv_file_path(tomtom_result_dir)
tomtom_results = pd.read_csv(tomtom_comparison_result_dir,
                                        sep="\t",comment="#")
filters_with_min_q = tomtom_results.groupby('Query_ID').min()["q-value"]
tomtom_results = tomtom_results[["Target_ID", "Query_ID", "q-value"]]
tomtom_results = tomtom_results[tomtom_results["q-value"]<0.05]
cisbp_motifs = {}

# Get the .txt file directory by changing the extension of jaspar_meme_file_dir from .meme to .txt 
jaspar_txt_file_dir = os.path.splitext(jaspar_meme_file_dir)[0] + '.txt'
with open(jaspar_txt_file_dir) as f:
    for line in f:
        if "MOTIF" in line:
            motif = line.strip().split()[-1]
            name_m = line.strip().split()[-2]
            cisbp_motifs[name_m] = motif

filters = tomtom_results["Query_ID"].unique()
annotation = {}
for f in filters:
    t = tomtom_results[tomtom_results["Query_ID"] == f]
    target_id = t["Target_ID"]
    if len(target_id) > 5:
        target_id = target_id[:5]
    # Join Unique annotations by '/'
    ann = "/".join({cisbp_motifs[i]: i for i in target_id.values})
    annotation[f] = ann
print("Annotation is generated!")
annotation = pd.Series(annotation)
print(annotation)

print('--------------------------------------------------------')
print('***********************   7/5   ************************')
print('--------------------------------------------------------')
print('Get Weight of Each Filter!')