import os
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
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
from utils import EnhancerDataset, split_dataset, train_model, regression_model_plot, plot_filter_weight


sys.path.append('../../Enhancer')  
from model.model import ConvNetDeep, DanQ, ExplaiNN,ConvNetDeep2, ExplaiNN2, ExplaiNN3
import scripts

# Specify the meme_package_dir variable
meme_package_dir = '/pmglocal/ty2514/meme'
path = os.path.join(meme_package_dir, 'bin') + ':' + os.path.join(meme_package_dir, 'libexec', 'meme-5.5.5') + ':' + os.environ['PATH']
print(f'Updating Environment by {path}')
# Update the PATH environment variable
os.environ['PATH'] = os.path.join(meme_package_dir, 'bin') + ':' + os.path.join(meme_package_dir, 'libexec', 'meme-5.5.5') + ':' + os.environ['PATH']

print('Check if tomtom exists')
# Check if 'tomtom' is in the PATH using 'which tomtom'
try:
    result = subprocess.run(['which', 'tomtom'], capture_output=True, text=True, check=True)
    print(f"Tomtom found at: {result.stdout.strip()}")
except subprocess.CalledProcessError:
    print("Error: 'tomtom' not found in the PATH.")
    sys.exit(1)  # Stop further execution if tomtom is not found

# If 'tomtom' is found, define directories and run the tomtom command
pwm_meme_file_dir = '/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_meme.meme'  # Replace with the actual path to the PWM MEME file
jaspar_meme_file_dir = '/pmglocal/ty2514/Enhancer/motif-clustering/databases/jaspar2024/JASPAR2024_CORE_vertebrates_mus_musculus_non-redundant_pfms_meme.meme'
result_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/ExplaiNN_both_results'
tomtom_result_dir = os.path.join(result_dir, 'tomtom_results')

print('Run tomtom')
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
