import csv
from typing import Dict, Tuple
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pickle
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import ks_2samp, pearsonr, spearmanr
import random

def process_data(fasta_dir, frag_seq_dir, stan_values_dir, save_df):
    """
    Processes genetic sequence data to map sequence identifiers to their respective sequences and
    integrates this information with experimental data (GFP intensities) from a CSV file.

    Parameters:
    - fasta_dir (str): Path to the FASTA file containing the sequences.
    - frag_seq_dir (str): Path to the CSV file containing fragment sequence information.
    - stan_values_dir (str): Path to the CSV file containing experimental stan values (GFP intensities).
    - save_df (bool): If True, the result is saved to a CSV file; if False, the DataFrame is returned.

    Returns:
    - DataFrame or None: Returns the processed DataFrame if save_df is False, otherwise saves to file and returns None.
    """
   

    # Get dictionary of fragment annotations
    id_to_annot_seq = get_fragment_info_dictionary(frag_seq_dir)
    
    # Read sequences from the fasta file and filter based on conditions
    id_to_seq = {}
    with open(fasta_dir, 'r') as fasta_file:
        for fasta in SeqIO.parse(fasta_file, 'fasta'):
            name, sequence = fasta.id, str(fasta.seq)
            ids = name.split("_")
            if ids[1] == "17" or ids[2] == "16":
                continue
            if (id_to_annot_seq[ids[0]][1] + 'CTGA' + id_to_annot_seq[ids[1]][1] + 'ACCA' + id_to_annot_seq[ids[2]][1]) != sequence:
                continue
            id_to_seq[name] = sequence

    # Load stan values and add sequence info
    df = pd.read_csv(stan_values_dir, delimiter=',', index_col=0)
    fragment_ids = []
    sequences = []

    for index, row in df.iterrows():
        pos1, pos2, pos3 = f"{int(row['Pos1']):02}", f"{int(row['Pos2']):02}", f"{int(row['Pos3']):02}"
        key = f"{pos1}_{pos2}_{pos3}"
        if key in id_to_seq:
            fragment_ids.append(key)
            sequences.append(id_to_seq[key])
        else:
            fragment_ids.append(None)
            sequences.append(None)

    df['fragment_ids'] = fragment_ids
    df['sequence'] = sequences
    df = df.dropna(subset=['fragment_ids'])

    #****************************************************************
    # Calculate mean of 'G-' for reference, assuming 'G-' itself is the mean here
    df['Z-Score'] = (df['G+'] - df['G-']) / df['G-_std']

    # Define a threshold for true positives, e.g., Z-Score > 2
    threshold = 3
    df['True Positive'] = df['Z-Score'] > threshold

    # Adding 'Expression' column based on 'True Positive'
    df['Expression'] = df['True Positive'].astype(int)

    # Check results
    print(df[['G-', 'G+','G-_std', 'Z-Score', 'True Positive', 'Expression']])
    #****************************************************************

    if save_df:
        df.to_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv', index=False)
    else:
        return df

def get_fragment_info_dictionary(csv_file_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Reads a CSV file containing fragment information and returns a dictionary 
    where 'fragment_id' serves as keys and ('fragment_anno', 'fragment_seq') serve as values.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        Dict[str, Tuple[str, str]]: A dictionary containing fragment information.
            Keys are 'fragment_id', values are tuples containing ('fragment_anno', 'fragment_seq').
    """
    # Dictionary to store the data
    data_dict = {}

    # Open the CSV file
    with open(csv_file_path, newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Skip the first row since it contains meaningless data
        next(reader)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the values for 'fragment_id', 'fragment_anno', and 'fragment_seq' columns
            fragment_id = row[0]
            fragment_anno = row[1]
            fragment_seq = row[2]
            
            # Create a dictionary entry with 'fragment_id' as the key
            data_dict[fragment_id] = (fragment_anno, fragment_seq)

    return data_dict

def dna_one_hot(seq, seq_len=None, flatten=False):
    # Reference: Novakovsky, G., Fornes, O., Saraswat, M. et al. ExplaiNN: interpretable and transparent neural networks for genomics. Genome Biol 24, 154 (2023). https://doi.org/10.1186/s13059-023-02985-y
    """
    Converts an input dna sequence to one hot encoded representation, with (A:0,C:1,G:2,T:3) alphabet

    :param seq: string, input dna sequence
    :param seq_len: int, optional, length of the string
    :param flatten: boolean, if true, makes a 1 column vector
    :return: numpy.array, one-hot encoded matrix of size (4, L), where L - the length of the input sequence
    """

    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace("A", "0")
    seq = seq.replace("C", "1")
    seq = seq.replace("G", "2")
    seq = seq.replace("T", "3")

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    # dtype="int8" fails for N's
    seq_code = np.zeros((4, seq_len), dtype="float16")
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:, i] = 0.
        else:
            try:
                seq_code[int(seq[i - seq_start]), i] = 1.
            except:
                seq_code[:, i] = 0.

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_code = seq_code.flatten()[None, :]

    return seq_code

def convert_one_hot_back_to_seq(dataloader):
    # Reference: Novakovsky, G., Fornes, O., Saraswat, M. et al. ExplaiNN: interpretable and transparent neural networks for genomics. Genome Biol 24, 154 (2023). https://doi.org/10.1186/s13059-023-02985-y
    """
    Converts one-hot encoded matrices back to DNA sequences
    :param dataloader: pytorch, DataLoader
    :return: list of strings, DNA sequences
    """

    sequences = []
    code = list("ACGT")
    for seqs, labels in tqdm(dataloader, total=len(dataloader)):
        x = seqs.permute(0, 1, 3, 2)
        x = x.squeeze(-1)
        for i in range(x.shape[0]):
            seq = ""
            for j in range(x.shape[-1]):
                try:
                    seq = seq + code[int(np.where(x[i, :, j] == 1)[0])]
                except:
                    print("error")
                    print(x[i, :, j])
                    print(np.where(x[i, :, j] == 1))
                    break
            sequences.append(seq)
    return sequences

def split_dataset(df, split_type='random', key=None, cutoff=0.8, seed=None):
    """
    Splits a dataset based on the specified criteria.

    Parameters:
    - df (DataFrame): The pandas DataFrame to split.
    - split_type (str): Type of split. 'random' for random split; 'fragment' for split based on sequence fragment presence.
    - key (int): The key to look for within Pos1, Pos2, Pos3 columns if split_type is 'fragment'.
    - cutoff (float): The proportion of the dataset to include in the train split (only for random split).
    - seed (int): Seed number for reproducibility.

    Returns:
    - train_set (DataFrame): The training dataset.
    - test_set (DataFrame): The testing dataset.
    """
    if split_type == 'random' and key!=None:
        print("Warning: split_type is random, 'key' parameter should be None")
    if split_type == 'fragment' and cutoff!=None:
        print("Warning: split_type is fragment, 'cutoff' parameter would not be used")
    if seed is not None:
        np.random.seed(seed)

    if split_type == 'random':
        train_set, test_set = train_test_split(df, test_size=1-cutoff, random_state=seed)
    elif split_type == 'fragment':
        if key is None:
            raise ValueError("Key must be specified for sequence fragment-based splitting.")
        if key not in df['Pos1'].values:
            raise ValueError(f"The key {key} is not present in the 'Pos1' column.")
        # Create a mask where any of Pos1, Pos2, Pos3 equals the key
        mask = (df['Pos1'] == key) | (df['Pos2'] == key) | (df['Pos3'] == key)
        test_set = df[mask]
        train_set = df[~mask]
    else:
        raise ValueError("Invalid split type specified. Use 'random' or 'key'.")

    return train_set, test_set

class EnhancerDataset(Dataset):
    def __init__(self, dataset, feature_list, scale_mode='none'):
        """
        Initialize the dataset with the mode specifying which labels to include.
        
        Args:
        label_mode (str): Specifies the label mode. Can be 'G+', 'G-','both','distance','GFP', or 'score'.
                            'G+' only includes G+ values,
                            'G-' only includes G- values,
                            'both' includes both G+ and G- values stacked together.
                            'distance' includes both G+ and G- values and motif distance scores
                            'GFP' includes G+ minus G-    
        scale_mode (str): Specifies the scaling mode. Can be 'none', '0-1', or '-1-1'.
                              'none' means no scaling, '0-1' scales labels to [0, 1],
                              and '-1-1' scales labels to [-1, 1].
        """


        # data loading
        df = dataset.copy()

        # Apply the one-hot encoding directly and safely
        df.loc[:, 'one_hot'] = df['sequence'].apply(lambda x: torch.from_numpy(dna_one_hot(x, flatten=False)))

        one_hot_list = df['one_hot'].tolist()
        self.x = torch.stack(one_hot_list).float()

        # Check if feature_list is empty:
        if len(feature_list) == 0:
            raise KeyError("feature_list should not be empty")

        # Check if all features in the feature_list exist in the data frame. If any feature is missing, raise an error.
        missing_features = [feature for feature in feature_list if feature not in df.columns]
        if missing_features:
            print(f'feature_list is: {feature_list}')
            print(f'All columns of dataset are: {list(df.columns)}')
            raise KeyError(f"The following features from 'feature_list' are not found in the dataset: {missing_features}")
        
        # Stack all features together vertically to prepare the labels
        labels = np.stack([np.array(df[feature]) for feature in feature_list], axis=1)
        # Convert labels to torch object
        self.y = torch.from_numpy(labels).float()

        # Apply scaling if specified
        if scale_mode == '0-1':
            self.y = scale_labels_0_1(self.y)
        elif scale_mode == '-1-1':
            self.y = scale_labels_minus1_1(self.y)
        elif scale_mode not in ['0-1','-1-1','none']:
            raise ValueError("Invalid scale_mode provided. Choose '0-1', '-1-1', or 'none'.")
        
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def evaluate_model(model, test_loader, batch_size, criterion, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        # Manully devide test_loss by total number of test samples by doing len(test_loader) * batch
        avg_test_loss = test_loss / (len(test_loader) * batch_size)
        avg_test_loss_by_batch = test_loss / len(test_loader)
    return avg_test_loss, avg_test_loss_by_batch

def train_model(model, train_loader, test_loader, target_labels, num_epochs=100, batch_size=10, learning_rate=1e-6, criteria = 'mse',optimizer_type = "adam",patience=10, seed = 42, save_model = False, dir_path = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if dir_path == None and save_model == True:
            print("dir_path is set to None, model will not be saved")

    # Initialize the model, loss criterion, and optimizer
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    if criteria == "bcewithlogits":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif criteria == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif criteria == "mse":
        criterion = torch.nn.MSELoss()
    elif criteria == "pearson":
        criterion = pearson_loss
    elif criteria == "poissonnll":
        criterion = torch.nn.PoissonNLLLoss()
    elif criteria == 'huber':
        criterion = torch.nn.SmoothL1Loss()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not recognized. Please choose between adam or sgd")
    # Lists to store loss history for visualization
    train_losses = []
    test_losses = []
    train_losses_by_batch = []
    test_losses_by_batch = []

    # Add a label to represent combined metrics
    target_label_names = target_labels[:]
    target_label_names.append('total')

    # Create a dictionary to store metrics from each epoch
    results = {}
    # For each label, create a dictionary of lists to store their metrics from current epoch
    for label in target_label_names:
        results[label] = {
            'mse': [], 
            'rmse': [], 
            'mae': [], 
            'r2': [], 
            'pearson_corr': [], 
            'spearman_corr': []
        }
    # Add another dictionary of lists to store combined metrics from current epoch
    #results['total'] = {
    #        'mse': [], 
    #        'rmse': [], 
    #        'mae': [], 
    #        'r2': [], 
    #        'pearson_corr': [], 
    #        'spearman_corr': []
    #    }
    
    # Add a label to represent combined metrics
    #target_label_names.append('total')

    # Early stopping specifics
    best_test_loss = float('inf')
    early_stop = False
    epochs_no_improve = 0
    best_pearson = -2.0
    best_pearson_epoch = -1
    best_r2 = -200.0
    best_r2_epoch = -1

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        n_total_steps = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            #labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 200 == 0 or i == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{n_total_steps}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_loss_by_batch = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_losses_by_batch.append(avg_train_loss_by_batch)

        # Evaluate the model
        avg_test_loss, avg_test_loss_by_batch = evaluate_model(model, test_loader, batch_size, criterion, device)
        test_losses.append(avg_test_loss)
        test_losses_by_batch.append(avg_test_loss_by_batch)

        #print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f} , Test Loss: {avg_test_loss:.4f}")
        #print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss By Batch: {avg_train_loss_by_batch:.4f} , Test Loss By Batch: {avg_test_loss_by_batch:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss_by_batch:.4f} , Test Loss: {avg_test_loss_by_batch:.4f}")

        # metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'pearson_corr': [], 'spearman_corr': []}
        metrics = evaluate_regression_model(model, test_loader, device)

        # First, check if the number of labels is the same as the number of labels in metrics
        if len(target_label_names) != len(metrics['mse']):
            raise IndexError(f"target_labels are {target_label_names}   length of metrics are {len(metrics['mse'])}")

        # for each label in target_label list, store the 6 metric scores in current epoch
        for i in range(len(target_label_names)):
            label = target_label_names[i]
            results[label]['mse'].append(metrics['mse'][i])
            results[label]['rmse'].append(metrics['rmse'][i])
            results[label]['mae'].append(metrics['mae'][i])
            results[label]['r2'].append(metrics['r2'][i])
            results[label]['pearson_corr'].append(metrics['pearson_corr'][i])
            results[label]['spearman_corr'].append(metrics['spearman_corr'][i])

        pearson_corr = metrics['pearson_corr'][-1]
        r2 = metrics['r2'][-1]

        if pearson_corr >= best_pearson:
            best_pearson = pearson_corr
            best_pearson_epoch = epoch
        if r2 >= best_r2:
            best_r2 = r2
            best_r2_epoch = epoch
        

        if save_model == True and dir_path != None:
            os.makedirs(dir_path, exist_ok=True)
            torch.save(model.state_dict(), f'{dir_path}/model_epoch_{epoch}.pth')

        # Check if test loss improved
        if avg_test_loss_by_batch < best_test_loss:
            best_test_loss = avg_test_loss_by_batch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check if early stopping is triggered
        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            early_stop = True
            break

    if save_model == True and dir_path != None:
        with open(f'{dir_path}/train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open(f'{dir_path}/test_losses.pkl', 'wb') as f:
            pickle.dump(test_losses, f)
        with open(f'{dir_path}/test_losses_by_batch.pkl', 'wb') as f:
            pickle.dump(test_losses_by_batch, f)
        with open(f'{dir_path}/train_losses_by_batch.pkl', 'wb') as f:
            pickle.dump(train_losses_by_batch, f)
        with open(f'{dir_path}/metric_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    print(f"Finished Training!")
    print(f"Best Pearson Correlation is {best_pearson} at epoch {best_pearson_epoch}")
    print(f"Best R2 Square Correlation is {best_r2} at epoch {best_r2_epoch}")

    return train_losses, test_losses, model, train_losses_by_batch, test_losses_by_batch,results, best_pearson_epoch, best_r2_epoch, device

def evaluate_regression_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            predictions.append(outputs)
            actuals.append(labels.cpu().numpy())

    # Convert to numpy arrays for easier manipulation
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Prepare to collect metrics for each label
    metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'pearson_corr': [], 'spearman_corr': []}
    
    # Loop over each label (column) and calculate individual metrics
    for i in range(predictions.shape[1]):
        pred_col = predictions[:, i]
        actual_col = actuals[:, i]
        
        # Calculate metrics
        mse = mean_squared_error(actual_col, pred_col)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_col, pred_col)
        r2 = r2_score(actual_col, pred_col)
        pearson_corr, _ = pearsonr(actual_col, pred_col)
        spearman_corr, _ = spearmanr(actual_col, pred_col)

        # Append results for this label
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        metrics['pearson_corr'].append(pearson_corr)
        metrics['spearman_corr'].append(spearman_corr)

    # Calculate metrics for the flattened results and append them
    flat_predictions = predictions.flatten()
    flat_actuals = actuals.flatten()
    metrics['mse'].append(mean_squared_error(flat_actuals, flat_predictions))
    metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))
    metrics['mae'].append(mean_absolute_error(flat_actuals, flat_predictions))
    metrics['r2'].append(r2_score(flat_actuals, flat_predictions))
    pearson_corr, _ = pearsonr(flat_actuals, flat_predictions)
    metrics['pearson_corr'].append(pearson_corr)
    spearman_corr, _ = spearmanr(flat_actuals, flat_predictions)
    metrics['spearman_corr'].append(spearman_corr)

    # Print results in a structured format
    print("------------------------Evaluation------------------------")
    for label_index in range(len(metrics['mse']) - 1):  # -1 because the last is the overall metric
        print(f"Label {label_index + 1}: MSE={metrics['mse'][label_index]:.4f}, "
              f"RMSE={metrics['rmse'][label_index]:.4f}, MAE={metrics['mae'][label_index]:.4f}, "
              f"R^2={metrics['r2'][label_index]:.4f}, Pearson={metrics['pearson_corr'][label_index]:.4f}, "
              f"Spearman={metrics['spearman_corr'][label_index]:.4f}")
    # Print overall metrics for flattened results
    print(f"Overall (Flattened): MSE={metrics['mse'][-1]:.4f}, "
          f"RMSE={metrics['rmse'][-1]:.4f}, MAE={metrics['mae'][-1]:.4f}, "
          f"R^2={metrics['r2'][-1]:.4f}, Pearson={metrics['pearson_corr'][-1]:.4f}, "
          f"Spearman={metrics['spearman_corr'][-1]:.4f}")
    print("----------------------------------------------------------")
    
    return metrics

def regression_model_plot(model, test_loader, train_losses_by_batch, test_losses_by_batch, device, results, label_mode = "G+", save_plot = False, dir_path = None, model_path = None, best_model = -1):
    # Directory setup
    if save_plot == True:
        os.makedirs(dir_path, exist_ok=True)
        with open(f'{dir_path}/best_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    if label_mode == "G+" or label_mode == "G-" or label_mode == 'score' or label_mode == 'GFP':
        actuals = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy().flatten()
                predictions.extend(outputs)
                actuals.extend(labels.cpu().numpy().flatten())
        
        # Convert to numpy arrays for easier manipulation
        predictions = np.array(predictions)
        actuals = np.array(actuals)
    elif label_mode == "both":
        actuals_g_plus = []
        actuals_g_minus = []
        predictions_g_plus = []
        predictions_g_minus = [] 
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                
                # Separate the predictions for G+ and G-
                predictions_g_plus.extend(outputs[:, 0])
                predictions_g_minus.extend(outputs[:, 1])
                
                # Separate the actuals for G+ and G-
                actuals_g_plus.extend(labels.cpu().numpy()[:, 0])
                actuals_g_minus.extend(labels.cpu().numpy()[:, 1])
        
        # Convert lists to numpy arrays for easier manipulation
        predictions_g_plus = np.array(predictions_g_plus)
        predictions_g_minus = np.array(predictions_g_minus)
        actuals_g_plus = np.array(actuals_g_plus)
        actuals_g_minus = np.array(actuals_g_minus)
    elif label_mode == "distance":
        actuals_g_plus = []
        actuals_g_minus = []
        actuals_dist = []
        predictions_g_plus = []
        predictions_g_minus = [] 
        predictions_dist = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                
                # Separate the predictions for G+ and G-
                predictions_g_plus.extend(outputs[:, 0])
                predictions_g_minus.extend(outputs[:, 1])
                predictions_dist.extend(outputs[:,2])
                
                # Separate the actuals for G+ and G-
                actuals_g_plus.extend(labels.cpu().numpy()[:, 0])
                actuals_g_minus.extend(labels.cpu().numpy()[:, 1])
                actuals_dist.extend(labels.cpu().numpy()[:, 2])
        
        # Convert lists to numpy arrays for easier manipulation
        predictions_g_plus = np.array(predictions_g_plus)
        predictions_g_minus = np.array(predictions_g_minus)
        predictions_dist = np.array(predictions_dist)
        actuals_g_plus = np.array(actuals_g_plus)
        actuals_g_minus = np.array(actuals_g_minus)
        actuals_dist = np.array(actuals_dist)
    else:
        TypeError("label_mode needs to be G+, G-, both, or distance")

    # If we want to include the first epoch's loss and start ticks from 1
    #epochs = range(2, len(train_losses_by_batch) + 1)
    #ticks = range(2, len(test_losses_by_batch) + 1, 10)

    ############Plot1############
    # Plotting train_losses and test_losses
    plt.figure(figsize=(10, 5))
    #plt.plot(epochs, train_losses_by_batch[1:], label='Training Loss')
    #plt.plot(epochs, test_losses_by_batch[1:], label='Testing Loss')
    plt.plot(train_losses_by_batch, label='Training Loss')
    plt.plot(test_losses_by_batch, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    #plt.xticks(ticks)
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'training_testing_loss_by_batch.png'))
    plt.show()

    ############Plot2############
    if label_mode == "G+" or label_mode == "G-" or label_mode == 'score' or label_mode == 'GFP':
        # Plotting predictions vs actuals
        plt.figure(figsize=(10, 5))
        plt.scatter(actuals, predictions, alpha=0.5, s= 2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. Actuals')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)  # Diagonal line
        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        # Perform K-S test
        stat, p_value = ks_2samp(actuals, predictions)

        # Create the box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot([actuals, predictions], labels=['Actual Values', 'Predicted Values'],flierprops={'marker':'o', 'color':'black', 'markersize':5})
        plt.title('Comparison of Actual and Predicted Values')
        plt.ylabel('Value')

        # Annotate with the K-S test results
        x1, x2 = 1, 2  # Columns numbers on your plot
        y, h, col = max(max(actuals), max(predictions)) + 1, 1, 'k'  # y is the height, h is the height of the line, col is the color
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f'p-value = {p_value:.3e}', ha='center', va='bottom', color=col)
        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_bar_plot.png'))
        plt.show()

        ############Plot4############
        # Histogram of true and predicted values
        bins = np.histogram_bin_edges(predictions, bins=100)

        # Plot the histograms using the same bins
        plt.figure(figsize=(10, 5))
        plt.hist(actuals, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual Values')
        plt.hist(predictions, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Actual and Predicted Values')
        x_min = math.floor(min(predictions.min(), actuals.min()))
        x_max = math.ceil(max(predictions.max(), actuals.max()))
        plt.xlim(x_min, x_max)
        plt.legend()
        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_histogram.png'))
        plt.show()
    elif label_mode == "both":
        """When label_mode is both, output would be G+ and G-"""
        ############Plot2############
        # Plotting predictions vs actuals
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot for G+ predictions vs actuals
        axs[0].scatter(actuals_g_plus, predictions_g_plus, alpha=0.5, s=2, label='G+')
        axs[0].set_xlabel('Actual G+ Values')
        axs[0].set_ylabel('Predicted G+ Values')
        axs[0].set_title('G+ Predictions vs. Actuals')
        axs[0].plot([actuals_g_plus.min(), actuals_g_plus.max()], [actuals_g_plus.min(), actuals_g_plus.max()], 'k--', lw=2)  # Diagonal line
        axs[0].legend()

        # Subplot for G- predictions vs actuals
        axs[1].scatter(actuals_g_minus, predictions_g_minus, alpha=0.5, s=2, color='red', label='G-')
        axs[1].set_xlabel('Actual G- Values')
        axs[1].set_ylabel('Predicted G- Values')
        axs[1].set_title('G- Predictions vs. Actuals')
        axs[1].plot([actuals_g_minus.min(), actuals_g_minus.max()], [actuals_g_minus.min(), actuals_g_minus.max()], 'k--', lw=2)  # Diagonal line
        axs[1].legend()

        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        
        # Perform K-S test
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        stat_g_plus, p_value_g_plus = ks_2samp(actuals_g_plus, predictions_g_plus)
        stat_g_minus, p_value_g_minus = ks_2samp(actuals_g_minus, predictions_g_minus)

        # Box plot for G+
        axs[0].boxplot([actuals_g_plus, predictions_g_plus], labels=['Actual G+', 'Predicted G+'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[0].set_title('Comparison of Actual and Predicted G+ Values')
        axs[0].set_ylabel('Value')
        # Annotate with the K-S test results for G+
        y = max(max(actuals_g_plus), max(predictions_g_plus)) + 1
        axs[0].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[0].text(1.5, y+1, f'p-value = {p_value_g_plus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for G-
        axs[1].boxplot([actuals_g_minus, predictions_g_minus], labels=['Actual G-', 'Predicted G-'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[1].set_title('Comparison of Actual and Predicted G- Values')
        axs[1].set_ylabel('Value')
        # Annotate with the K-S test results for G-
        y = max(max(actuals_g_minus), max(predictions_g_minus)) + 1
        axs[1].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[1].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_box_plot.png'))
        plt.show()

        ############Plot4############
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Histogram for G+
        bins = np.histogram_bin_edges(actuals_g_plus, bins=50)
        axs[0].hist(actuals_g_plus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G+')
        axs[0].hist(predictions_g_plus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G+')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Histogram of Actual and Predicted G+ Values')
        axs[0].legend()

        # Histogram for G-
        bins = np.histogram_bin_edges(actuals_g_minus, bins=50)
        axs[1].hist(actuals_g_minus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G-')
        axs[1].hist(predictions_g_minus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G-')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Actual and Predicted G- Values')
        axs[1].legend()

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_histogram.png'))
        plt.show()
    elif label_mode == "distance":
        """When label_mode is distance, output would be G+, G-, and motif distance score"""
        ############Plot2############
        # Plotting predictions vs actuals
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot for G+ predictions vs actuals
        axs[0].scatter(actuals_g_plus, predictions_g_plus, alpha=0.5, s=2, label='G+')
        axs[0].set_xlabel('Actual G+ Values')
        axs[0].set_ylabel('Predicted G+ Values')
        axs[0].set_title('G+ Predictions vs. Actuals')
        axs[0].plot([actuals_g_plus.min(), actuals_g_plus.max()], [actuals_g_plus.min(), actuals_g_plus.max()], 'k--', lw=2)  # Diagonal line
        axs[0].legend()

        # Subplot for G- predictions vs actuals
        axs[1].scatter(actuals_g_minus, predictions_g_minus, alpha=0.5, s=2, color='red', label='G-')
        axs[1].set_xlabel('Actual G- Values')
        axs[1].set_ylabel('Predicted G- Values')
        axs[1].set_title('G- Predictions vs. Actuals')
        axs[1].plot([actuals_g_minus.min(), actuals_g_minus.max()], [actuals_g_minus.min(), actuals_g_minus.max()], 'k--', lw=2)  # Diagonal line
        axs[1].legend()

        # Subplot for distance predictions vs actuals
        axs[2].scatter(actuals_dist, predictions_dist, alpha=0.5, s=2, color='green', label='G-')
        axs[2].set_xlabel('Actual Motif Distance Score')
        axs[2].set_ylabel('Predicted Motif Distance Score')
        axs[2].set_title('Motif Distance Score Predictions vs. Actuals')
        axs[2].plot([predictions_dist.min(), predictions_dist.max()], [predictions_dist.min(), predictions_dist.max()], 'k--', lw=2)  # Diagonal line
        axs[2].legend()

        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        
        # Perform K-S test
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        stat_g_plus, p_value_g_plus = ks_2samp(actuals_g_plus, predictions_g_plus)
        stat_g_minus, p_value_g_minus = ks_2samp(actuals_g_minus, predictions_g_minus)

        # Box plot for G+
        axs[0].boxplot([actuals_g_plus, predictions_g_plus], labels=['Actual G+', 'Predicted G+'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[0].set_title('Comparison of Actual and Predicted G+ Values')
        axs[0].set_ylabel('Value')
        # Annotate with the K-S test results for G+
        y = max(max(actuals_g_plus), max(predictions_g_plus)) + 1
        axs[0].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[0].text(1.5, y+1, f'p-value = {p_value_g_plus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for G-
        axs[1].boxplot([actuals_g_minus, predictions_g_minus], labels=['Actual G-', 'Predicted G-'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[1].set_title('Comparison of Actual and Predicted G- Values')
        axs[1].set_ylabel('Value')
        # Annotate with the K-S test results for G-
        y = max(max(actuals_g_minus), max(predictions_g_minus)) + 1
        axs[1].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[1].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for Motif Distance Score
        axs[2].boxplot([actuals_dist, predictions_dist], labels=['Actual Motif Distance Score', 'Predicted Motif Distance Score'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[2].set_title('Comparison of Actual and Predicted Motif Distance Score')
        axs[2].set_ylabel('Motif Distance Score')
        # Annotate with the K-S test results for Motif Distance Score
        y = max(max(actuals_dist), max(predictions_dist)) + 1
        axs[2].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[2].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_and_Motif_Distance_box_plot.png'))
        plt.show()

        ############Plot4############
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Histogram for G+
        bins = np.histogram_bin_edges(actuals_g_plus, bins=50)
        axs[0].hist(actuals_g_plus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G+')
        axs[0].hist(predictions_g_plus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G+')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Histogram of Actual and Predicted G+ Values')
        axs[0].legend()

        # Histogram for G-
        bins = np.histogram_bin_edges(actuals_g_minus, bins=50)
        axs[1].hist(actuals_g_minus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G-')
        axs[1].hist(predictions_g_minus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G-')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Actual and Predicted G- Values')
        axs[1].legend()

        # Histogram for Motif Distance Score
        bins = np.histogram_bin_edges(predictions_dist, bins=50)
        axs[2].hist(actuals_dist, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual Motif Distance Score')
        axs[2].hist(predictions_dist, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted Motif Distance Score')
        axs[2].set_xlabel('Motif Distance Score')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Histogram of Actual and Predicted Motif Distance Score')
        # Calculate the min and max values for xlim
        x_min = 0
        x_max = max(predictions_dist.max(), actuals_dist.max())

        axs[2].set_xlim(x_min, x_max)
        axs[2].legend()

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_and_Motif_Distance_histogram.png'))
        plt.show()
    else:
        TypeError("label_mode needs to be G+, G-, both, or distance")

    for label, metrics in results.items():
        # Plotting MSE, RMSE, MAE
        fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
        axs1[0].plot(metrics['mse'], label='MSE')
        axs1[0].set_title(f'Mean Squared Error over Epochs: {label}')
        axs1[0].set_xlabel('Epoch')
        axs1[0].set_ylabel('MSE')
        axs1[0].legend()

        axs1[1].plot(metrics['rmse'], label='RMSE')
        axs1[1].set_title(f'Root Mean Squared Error over Epochs: {label}')
        axs1[1].set_xlabel('Epoch')
        axs1[1].set_ylabel('RMSE')
        axs1[1].legend()

        axs1[2].plot(metrics['mae'], label='MAE')
        axs1[2].set_title(f'Mean Absolute Error over Epochs: {label}')
        axs1[2].set_xlabel('Epoch')
        axs1[2].set_ylabel('MAE')
        axs1[2].legend()

        fig1.tight_layout(pad=3.0)
        if save_plot:
            plt.savefig(os.path.join(dir_path, f'mse_rmse_mae_{label}.png'))
        plt.show()

        # Plotting R^2, Pearson, Spearman
        # Calculate max ylim for R^2
        max_r2 = max(metrics['r2'])
        ylim_r2 = max(max_r2 + 0.1, 1)

        fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
        axs2[0].plot(metrics['r2'], label='R^2 Score')
        axs2[0].set_title(f'R^2 Score over Epochs: {label}')
        axs2[0].set_xlabel('Epoch')
        axs2[0].set_ylabel('R^2')
        axs2[0].set_ylim(0, ylim_r2)
        axs2[0].legend()

        axs2[1].plot(metrics['pearson_corr'], label='Pearson Correlation')
        axs2[1].set_title(f'Pearson Correlation over Epochs: {label}')
        axs2[1].set_xlabel('Epoch')
        axs2[1].set_ylabel('Pearson')
        axs2[1].legend()

        axs2[2].plot(metrics['spearman_corr'], label='Spearman Correlation')
        axs2[2].set_title(f'Spearman Correlation over Epochs: {label}')
        axs2[2].set_xlabel('Epoch')
        axs2[2].set_ylabel('Spearman')
        axs2[2].legend()

        fig2.tight_layout(pad=3.0)
        if save_plot:
            plt.savefig(os.path.join(dir_path, f'r2_pearson_spearman_{label}.png'))
        plt.show()

    # Plot MSE, RMSE, and MAE for all labels on the same plot
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    for label, metrics in results.items():
        axs1[0].plot(metrics['mse'], label=f'{label}')
        axs1[1].plot(metrics['rmse'], label=f'{label}')
        axs1[2].plot(metrics['mae'], label=f'{label}')

    # Setting titles, labels, and legends for the first plot
    for ax, metric in zip(axs1, ['MSE', 'RMSE', 'MAE']):
        ax.set_title(f'{metric} over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()

    fig1.tight_layout()
    if save_plot:
        fig1.savefig(os.path.join(dir_path, 'all_labels_metrics_plot.png'))
    plt.show()

    # Plot R^2, Pearson Correlation, and Spearman Correlation for all labels on the same plot
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    for label, metrics in results.items():
        axs2[0].plot(metrics['r2'], label=f'{label}')
        axs2[1].plot(metrics['pearson_corr'], label=f'{label}')
        axs2[2].plot(metrics['spearman_corr'], label=f'{label}')

    # Setting titles, labels, and legends for the second plot
    for ax, metric in zip(axs2, ['R^2 Score', 'Pearson Correlation', 'Spearman Correlation']):
        ax.set_title(f'{metric} over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        if metric == 'R^2 Score':
            ax.set_ylim(0, 1)
    
    fig2.tight_layout()
    if save_plot:
        fig2.savefig(os.path.join(dir_path, 'all_labels_correlations_plot.png'))
    plt.show()

    return results['total']['mse'][-1], results['total']['rmse'][-1], results['total']['mae'][-1], results['total']['r2'][-1], results['total']['pearson_corr'][-1], results['total']['spearman_corr'][-1]

def plot_filter_weight(weight_df, top_n=10):
    """
    Function to plot horizontal bar plots for each row's values in weight_df.
    Each plot will display the top 'top_n' values sorted by their absolute values.

    Args:
    - weight_df (pd.DataFrame): DataFrame containing the weights to be plotted. 
    There might be a row 'filter' specifying which filter each weight belongs to. 
    When this row exists, this row would be excluded from plotting
    - top_n (int): Number of top bars to display. Default is 10.

    Returns:
    - None
    """
    # Ignore the filter row
    weight_df = weight_df.drop("filters", errors='ignore')

    # Loop through each row to create a separate plot
    for index, row in weight_df.iterrows():
        # Convert row to a numpy array for numeric operations
        row_values = row.to_numpy()

        # Get indices of the sorted absolute values (largest to smallest)
        sorted_indices = np.argsort(np.abs(row_values))[::-1]

        # Use these indices to sort values from the row and the column names
        sorted_values = row_values[sorted_indices][:top_n]  # Only take the top_n values
        sorted_names = row.index[sorted_indices][:top_n]  # Only take the top_n names

        # Use these sorted indices to get the non-absolute (original) values in sorted order
        sorted_original_values = row_values[sorted_indices][:top_n]

        # Define colors for the bars based on whether 'filter' is in the column names
        colors = ['red' if 'filter' in name.lower() else 'royalblue' for name in sorted_names]

        # Reverse the order for plotting from top to bottom
        sorted_names = sorted_names[::-1]
        sorted_original_values = sorted_original_values[::-1]
        colors = colors[::-1]

        # Create a new figure for each row
        plt.figure(figsize=(max(10, len(sorted_names) * 1.0), max(5, len(sorted_names) * 0.5)))  # Adjusted width and height for better fit

        # Create the horizontal bar plot
        plt.barh(range(len(sorted_original_values)), sorted_original_values, color=colors)

        # Set title and labels
        plt.title(f'Ranked Weight Distribution for {index}')
        plt.xlabel('Weight Values')
        plt.ylabel('Column Names')

        # Set y-tick labels
        plt.yticks(range(len(sorted_names)), sorted_names)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

































def regression_model_plot_old(model, test_loader, train_losses_by_batch, test_losses_by_batch, device, results, label_mode = "G+", save_plot = False, dir_path = None, model_path = None, best_model = -1):
    # Directory setup
    if save_plot == True:
        os.makedirs(dir_path, exist_ok=True)
        with open(f'{dir_path}/results.pkl', 'wb') as f:
            pickle.dump(results, f)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    if label_mode == "G+" or label_mode == "G-":
        actuals = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy().flatten()
                predictions.extend(outputs)
                actuals.extend(labels.cpu().numpy().flatten())
        
        # Convert to numpy arrays for easier manipulation
        predictions = np.array(predictions)
        actuals = np.array(actuals)
    elif label_mode == "both":
        actuals_g_plus = []
        actuals_g_minus = []
        predictions_g_plus = []
        predictions_g_minus = [] 
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                
                # Separate the predictions for G+ and G-
                predictions_g_plus.extend(outputs[:, 0])
                predictions_g_minus.extend(outputs[:, 1])
                
                # Separate the actuals for G+ and G-
                actuals_g_plus.extend(labels.cpu().numpy()[:, 0])
                actuals_g_minus.extend(labels.cpu().numpy()[:, 1])
        
        # Convert lists to numpy arrays for easier manipulation
        predictions_g_plus = np.array(predictions_g_plus)
        predictions_g_minus = np.array(predictions_g_minus)
        actuals_g_plus = np.array(actuals_g_plus)
        actuals_g_minus = np.array(actuals_g_minus)
    elif label_mode == "distance":
        actuals_g_plus = []
        actuals_g_minus = []
        actuals_dist = []
        predictions_g_plus = []
        predictions_g_minus = [] 
        predictions_dist = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                
                # Separate the predictions for G+ and G-
                predictions_g_plus.extend(outputs[:, 0])
                predictions_g_minus.extend(outputs[:, 1])
                predictions_dist.extend(outputs[:,2])
                
                # Separate the actuals for G+ and G-
                actuals_g_plus.extend(labels.cpu().numpy()[:, 0])
                actuals_g_minus.extend(labels.cpu().numpy()[:, 1])
                actuals_dist.extend(labels.cpu().numpy()[:, 2])
        
        # Convert lists to numpy arrays for easier manipulation
        predictions_g_plus = np.array(predictions_g_plus)
        predictions_g_minus = np.array(predictions_g_minus)
        predictions_dist = np.array(predictions_dist)
        actuals_g_plus = np.array(actuals_g_plus)
        actuals_g_minus = np.array(actuals_g_minus)
        actuals_dist = np.array(actuals_dist)
    else:
        TypeError("label_mode needs to be G+, G-, both, or distance")

    # If we want to include the first epoch's loss and start ticks from 1
    #epochs = range(2, len(train_losses_by_batch) + 1)
    #ticks = range(2, len(test_losses_by_batch) + 1, 10)

    ############Plot1############
    # Plotting train_losses and test_losses
    plt.figure(figsize=(10, 5))
    #plt.plot(epochs, train_losses_by_batch[1:], label='Training Loss')
    #plt.plot(epochs, test_losses_by_batch[1:], label='Testing Loss')
    plt.plot(train_losses_by_batch, label='Training Loss')
    plt.plot(test_losses_by_batch, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    #plt.xticks(ticks)
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'training_testing_loss_by_batch.png'))
    plt.show()

    ############Plot2############
    if label_mode == "G+" or label_mode == "G-":
        # Plotting predictions vs actuals
        plt.figure(figsize=(10, 5))
        plt.scatter(actuals, predictions, alpha=0.5, s= 2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. Actuals')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)  # Diagonal line
        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        # Perform K-S test
        stat, p_value = ks_2samp(actuals, predictions)

        # Create the box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot([actuals, predictions], labels=['Actual Values', 'Predicted Values'],flierprops={'marker':'o', 'color':'black', 'markersize':5})
        plt.title('Comparison of Actual and Predicted Values')
        plt.ylabel('Value')

        # Annotate with the K-S test results
        x1, x2 = 1, 2  # Columns numbers on your plot
        y, h, col = max(max(actuals), max(predictions)) + 1, 1, 'k'  # y is the height, h is the height of the line, col is the color
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f'p-value = {p_value:.3e}', ha='center', va='bottom', color=col)
        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_bar_plot.png'))
        plt.show()

        ############Plot4############
        # Histogram of true and predicted values
        bins = np.histogram_bin_edges(actuals, bins=50)

        # Plot the histograms using the same bins
        plt.figure(figsize=(10, 5))
        plt.hist(actuals, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual Values')
        plt.hist(predictions, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Actual and Predicted Values')
        plt.legend()
        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_histogram.png'))
        plt.show()
    elif label_mode == "both":
        """When label_mode is both, output would be G+ and G-"""
        ############Plot2############
        # Plotting predictions vs actuals
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot for G+ predictions vs actuals
        axs[0].scatter(actuals_g_plus, predictions_g_plus, alpha=0.5, s=2, label='G+')
        axs[0].set_xlabel('Actual G+ Values')
        axs[0].set_ylabel('Predicted G+ Values')
        axs[0].set_title('G+ Predictions vs. Actuals')
        axs[0].plot([actuals_g_plus.min(), actuals_g_plus.max()], [actuals_g_plus.min(), actuals_g_plus.max()], 'k--', lw=2)  # Diagonal line
        axs[0].legend()

        # Subplot for G- predictions vs actuals
        axs[1].scatter(actuals_g_minus, predictions_g_minus, alpha=0.5, s=2, color='red', label='G-')
        axs[1].set_xlabel('Actual G- Values')
        axs[1].set_ylabel('Predicted G- Values')
        axs[1].set_title('G- Predictions vs. Actuals')
        axs[1].plot([actuals_g_minus.min(), actuals_g_minus.max()], [actuals_g_minus.min(), actuals_g_minus.max()], 'k--', lw=2)  # Diagonal line
        axs[1].legend()

        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        
        # Perform K-S test
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        stat_g_plus, p_value_g_plus = ks_2samp(actuals_g_plus, predictions_g_plus)
        stat_g_minus, p_value_g_minus = ks_2samp(actuals_g_minus, predictions_g_minus)

        # Box plot for G+
        axs[0].boxplot([actuals_g_plus, predictions_g_plus], labels=['Actual G+', 'Predicted G+'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[0].set_title('Comparison of Actual and Predicted G+ Values')
        axs[0].set_ylabel('Value')
        # Annotate with the K-S test results for G+
        y = max(max(actuals_g_plus), max(predictions_g_plus)) + 1
        axs[0].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[0].text(1.5, y+1, f'p-value = {p_value_g_plus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for G-
        axs[1].boxplot([actuals_g_minus, predictions_g_minus], labels=['Actual G-', 'Predicted G-'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[1].set_title('Comparison of Actual and Predicted G- Values')
        axs[1].set_ylabel('Value')
        # Annotate with the K-S test results for G-
        y = max(max(actuals_g_minus), max(predictions_g_minus)) + 1
        axs[1].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[1].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_box_plot.png'))
        plt.show()

        ############Plot4############
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Histogram for G+
        bins = np.histogram_bin_edges(actuals_g_plus, bins=50)
        axs[0].hist(actuals_g_plus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G+')
        axs[0].hist(predictions_g_plus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G+')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Histogram of Actual and Predicted G+ Values')
        axs[0].legend()

        # Histogram for G-
        bins = np.histogram_bin_edges(actuals_g_minus, bins=50)
        axs[1].hist(actuals_g_minus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G-')
        axs[1].hist(predictions_g_minus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G-')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Actual and Predicted G- Values')
        axs[1].legend()

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_histogram.png'))
        plt.show()
    elif label_mode == "distance":
        """When label_mode is distance, output would be G+, G-, and motif distance score"""
        ############Plot2############
        # Plotting predictions vs actuals
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot for G+ predictions vs actuals
        axs[0].scatter(actuals_g_plus, predictions_g_plus, alpha=0.5, s=2, label='G+')
        axs[0].set_xlabel('Actual G+ Values')
        axs[0].set_ylabel('Predicted G+ Values')
        axs[0].set_title('G+ Predictions vs. Actuals')
        axs[0].plot([actuals_g_plus.min(), actuals_g_plus.max()], [actuals_g_plus.min(), actuals_g_plus.max()], 'k--', lw=2)  # Diagonal line
        axs[0].legend()

        # Subplot for G- predictions vs actuals
        axs[1].scatter(actuals_g_minus, predictions_g_minus, alpha=0.5, s=2, color='red', label='G-')
        axs[1].set_xlabel('Actual G- Values')
        axs[1].set_ylabel('Predicted G- Values')
        axs[1].set_title('G- Predictions vs. Actuals')
        axs[1].plot([actuals_g_minus.min(), actuals_g_minus.max()], [actuals_g_minus.min(), actuals_g_minus.max()], 'k--', lw=2)  # Diagonal line
        axs[1].legend()

        # Subplot for distance predictions vs actuals
        axs[2].scatter(actuals_dist, predictions_dist, alpha=0.5, s=2, color='green', label='G-')
        axs[2].set_xlabel('Actual Motif Distance Score')
        axs[2].set_ylabel('Predicted Motif Distance Score')
        axs[2].set_title('Motif Distance Score Predictions vs. Actuals')
        axs[2].plot([actuals_dist.min(), actuals_dist.max()], [actuals_dist.min(), actuals_dist.max()], 'k--', lw=2)  # Diagonal line
        axs[2].legend()

        if save_plot == True:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
        plt.show()

        ############Plot3############
        # Plotting bar plots of true and predicted values
        
        # Perform K-S test
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        stat_g_plus, p_value_g_plus = ks_2samp(actuals_g_plus, predictions_g_plus)
        stat_g_minus, p_value_g_minus = ks_2samp(actuals_g_minus, predictions_g_minus)

        # Box plot for G+
        axs[0].boxplot([actuals_g_plus, predictions_g_plus], labels=['Actual G+', 'Predicted G+'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[0].set_title('Comparison of Actual and Predicted G+ Values')
        axs[0].set_ylabel('Value')
        # Annotate with the K-S test results for G+
        y = max(max(actuals_g_plus), max(predictions_g_plus)) + 1
        axs[0].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[0].text(1.5, y+1, f'p-value = {p_value_g_plus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for G-
        axs[1].boxplot([actuals_g_minus, predictions_g_minus], labels=['Actual G-', 'Predicted G-'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[1].set_title('Comparison of Actual and Predicted G- Values')
        axs[1].set_ylabel('Value')
        # Annotate with the K-S test results for G-
        y = max(max(actuals_g_minus), max(predictions_g_minus)) + 1
        axs[1].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[1].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        # Box plot for Motif Distance Score
        axs[2].boxplot([actuals_dist, predictions_dist], labels=['Actual Motif Distance Score', 'Predicted Motif Distance Score'], flierprops={'marker':'o', 'color':'black', 'markersize':5})
        axs[2].set_title('Comparison of Actual and Predicted Motif Distance Score')
        axs[2].set_ylabel('Motif Distance Score')
        # Annotate with the K-S test results for Motif Distance Score
        y = max(max(actuals_dist), max(predictions_dist)) + 1
        axs[2].plot([1, 1, 2, 2], [y, y+1, y+1, y], lw=1.5, c='k')
        axs[2].text(1.5, y+1, f'p-value = {p_value_g_minus:.3e}', ha='center', va='bottom', color='k')

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_and_Motif_Distance_box_plot.png'))
        plt.show()

        ############Plot4############
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Histogram for G+
        bins = np.histogram_bin_edges(actuals_g_plus, bins=50)
        axs[0].hist(actuals_g_plus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G+')
        axs[0].hist(predictions_g_plus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G+')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Histogram of Actual and Predicted G+ Values')
        axs[0].legend()

        # Histogram for G-
        bins = np.histogram_bin_edges(actuals_g_minus, bins=50)
        axs[1].hist(actuals_g_minus, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual G-')
        axs[1].hist(predictions_g_minus, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted G-')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Actual and Predicted G- Values')
        axs[1].legend()

        # Histogram for Motif Distance Score
        bins = np.histogram_bin_edges(actuals_dist, bins=50)
        axs[2].hist(actuals_dist, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Actual Motif Distance Score')
        axs[2].hist(predictions_dist, bins=bins, alpha=0.5, color='green', edgecolor='black', label='Predicted Motif Distance Score')
        axs[2].set_xlabel('Motif Distance Score')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Histogram of Actual and Predicted Motif Distance Score')
        axs[2].legend()

        if save_plot:
            plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_G_Plus_Minus_and_Motif_Distance_histogram.png'))
        plt.show()
    else:
        TypeError("label_mode needs to be G+, G-, both, or distance")

    ############ Figure 5: MSE, RMSE, MAE ############
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 15))
    axs1[0].plot(results['mse'], label='MSE')
    axs1[0].set_title('Mean Squared Error over Epochs')
    axs1[0].set_xlabel('Epoch')
    axs1[0].set_ylabel('MSE')
    axs1[0].legend()

    axs1[1].plot(results['rmse'], label='RMSE')
    axs1[1].set_title('Root Mean Squared Error over Epochs')
    axs1[1].set_xlabel('Epoch')
    axs1[1].set_ylabel('RMSE')
    axs1[1].legend()

    axs1[2].plot(results['mae'], label='MAE')
    axs1[2].set_title('Mean Absolute Error over Epochs')
    axs1[2].set_xlabel('Epoch')
    axs1[2].set_ylabel('MAE')
    axs1[2].legend()

    fig1.tight_layout(pad=3.0)  # Adjust layout to make room for titles etc.
    if save_plot:
        plt.savefig(os.path.join(dir_path, 'mse_rmse_mae.png'))
    plt.show()

    ############ Figure 6: R^2, Pearson, Spearman ############
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 15))
    axs2[0].plot(results['r2'], label='R^2 Score')
    axs2[0].set_title('R^2 Score over Epochs')
    axs2[0].set_xlabel('Epoch')
    axs2[0].set_ylabel('R^2')
    axs2[0].set_ylim(-1, 1)
    axs2[0].legend()

    axs2[1].plot(results['pearson_corr'], label='Pearson Correlation')
    axs2[1].set_title('Pearson Correlation over Epochs')
    axs2[1].set_xlabel('Epoch')
    axs2[1].set_ylabel('Pearson')
    axs2[1].legend()

    axs2[2].plot(results['spearman_corr'], label='Spearman Correlation')
    axs2[2].set_title('Spearman Correlation over Epochs')
    axs2[2].set_xlabel('Epoch')
    axs2[2].set_ylabel('Spearman')
    axs2[2].legend()

    fig2.tight_layout(pad=3.0)  # Adjust layout to make room for titles etc.
    if save_plot:
        plt.savefig(os.path.join(dir_path, 'r2_pearson_spearman.png'))
    plt.show()

    return results['mse'][best_model], results['rmse'][best_model], results['mae'][best_model], results['r2'][best_model], results['pearson_corr'][best_model], results['spearman_corr'][best_model]

def scale_labels_0_1(labels):
    """ Scales labels to the range [0, 1]. """
    min_val = labels.min(0, keepdim=True)[0]
    max_val = labels.max(0, keepdim=True)[0]
    return (labels - min_val) / (max_val - min_val)

def scale_labels_minus1_1(labels):
    """ Scales labels to the range [-1, 1]. """
    min_val = labels.min(0, keepdim=True)[0]
    max_val = labels.max(0, keepdim=True)[0]
    return 2 * (labels - min_val) / (max_val - min_val) - 1

def pearson_loss(x, y):
    """
    Compute Pearson correlation-based loss.
    :param x: torch.Tensor, predicted values.
    :param y: torch.Tensor, true values.
    :return: torch.Tensor, one minus Pearson correlation coefficient.
    """
    x = x.squeeze()  # Ensure x is the correct shape.
    y = y.squeeze()  # Ensure y is the correct shape.
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    return 1 - r  # Minimize 1 - r to maximize r.

def pearson_correlation(y_true, y_pred):
    """
    Calculate the Pearson correlation coefficient between the ground truth (y_true)
    and predictions (y_pred) using numpy.
    """
    # Ensure arrays are in float for precision
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # Calculate means
    mean_y_true = np.mean(y_true, axis=0)
    mean_y_pred = np.mean(y_pred, axis=0)
    
    # Calculate deviations from means
    y_true_centered = y_true - mean_y_true
    y_pred_centered = y_pred - mean_y_pred
    
    # Covariance between y_true and y_pred
    covariance = np.sum(y_true_centered * y_pred_centered)
    
    # Variances of y_true and y_pred
    y_true_variance = np.sum(y_true_centered ** 2)
    y_pred_variance = np.sum(y_pred_centered ** 2)
    
    # Pearson correlation coefficient calculation
    denominator = np.sqrt(y_true_variance * y_pred_variance)
    if denominator == 0:
        return 0  # Avoid division by zero; return 0 correlation
    correlation_coefficient = covariance / denominator
    
    return correlation_coefficient

def evaluate_regression_model_old(model, test_loader, train_losses, test_losses, train_losses_by_batch, test_losses_by_batch, device, save_plot = False, dir_path = None):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy().flatten()
            predictions.extend(outputs)
            actuals.extend(labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays for easier manipulation
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    print(f"shape of predictions: {predictions.shape}, shape of actuals: {actuals.shape}")
    # Calculating errors and R-squared
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    # Calculating Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(actuals, predictions)
    spearman_corr, _ = spearmanr(actuals, predictions)

    # Directory setup
    if save_plot == True:
        os.makedirs(dir_path, exist_ok=True)

    # Create a list of epoch numbers starting from 2 (since we are skipping the first index, index 1)
    # If we want to include the first epoch's loss and start ticks from 1
    epochs = range(2, len(train_losses) + 1)

    # Create ticks every 10 epochs starting from 1
    ticks = range(2, len(train_losses) + 1, 10)

    # Plotting train_losses and test_losses starting from the second epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses[1:], label='Training Loss')  # Skip index 0 by slicing from index 1
    plt.plot(epochs, test_losses[1:], label='Testing Loss')  # Skip index 0 by slicing from index 1
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs (Starting from Epoch 2)')
    plt.legend()
    plt.xticks(ticks)  # Set x-ticks to show proper epoch numbers
    if save_plot:
        plt.savefig(os.path.join(dir_path, 'training_testing_loss.png'))
    plt.show()

    # Plotting train_losses and test_losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses_by_batch[1:], label='Training Loss')
    plt.plot(epochs, test_losses_by_batch[1:], label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing (By Batch) Loss Over Epochs')
    plt.legend()
    plt.xticks(ticks)
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'training_testing_loss_by_batch.png'))
    plt.show()
    
    # Plotting predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.scatter(actuals, predictions, alpha=0.5, s= 2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. Actuals')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)  # Diagonal line
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
    plt.show()

    # Plotting bar plots of true and predicted values
    # Perform K-S test
    stat, p_value = ks_2samp(actuals, predictions)

    # Create the box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot([actuals, predictions], labels=['Actual Values', 'Predicted Values'],flierprops={'marker':'o', 'color':'black', 'markersize':5})
    plt.title('Comparison of Actual and Predicted Values')
    plt.ylabel('Value')

    # Annotate with the K-S test results
    x1, x2 = 1, 2  # Columns numbers on your plot
    y, h, col = max(max(actuals), max(predictions)) + 1, 1, 'k'  # y is the height, h is the height of the line, col is the color
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, f'p-value = {p_value:.3e}', ha='center', va='bottom', color=col)
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_bar_plot.png'))
    plt.show()

    residuals = actuals - predictions
    # Plotting residuals
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predictions')
    plt.axhline(y=0, color='r', linestyle='--')
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'Residual_Prediction_plot.png'))
    plt.show()

    # Histogram of true and predicted values
    plt.figure(figsize=(10, 5))
    plt.hist(actuals, bins=50, alpha=0.7, color='red', edgecolor='black', label='Actual Values')
    plt.hist(predictions, bins=50, alpha=0.5, color='green', edgecolor='black', label='Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Actual and Predicted Values')
    plt.legend()
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_histgram.png'))
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.title('Histogram of Residuals')
    if save_plot == True:
        plt.savefig(os.path.join(dir_path, 'Residual_histgram.png'))
    plt.show()

    return mse, rmse, mae, r2, pearson_corr, spearman_corr

def evaluate_regression_model_old(model, test_loader, device):
    model.eval()
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy().flatten()
            predictions.extend(outputs)
            actuals.extend(labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays for easier manipulation
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    # Calculating errors and R-squared
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    # Calculating Pearson and Spearman correlations
    pearson_corr = pearson_correlation(actuals, predictions)
    spearman_corr, _ = spearmanr(actuals, predictions)
    print("------------------------Evaluation------------------------")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")
    print("----------------------------------------------------------")
    return mse, rmse, mae, r2, pearson_corr, spearman_corr