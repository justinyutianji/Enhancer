import csv
import shutil
from typing import Dict, Tuple
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import random
from matplotlib import rcParams
import glob

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

def find_tsv_file_path(dir_path):
    search_pattern = os.path.join(dir_path, "*.tsv")
    tsv_files = glob.glob(search_pattern)
    
    if len(tsv_files) == 1:
        return tsv_files[0]  # Return the full path of the .tsv file
    elif len(tsv_files) > 1:
        raise ValueError("More than one .tsv file found.")
    else:
        raise FileNotFoundError("No .tsv file found.")

def pearson_loss(x, y):
    """
    Loss that is based on Pearson correlation/objective function
    :param x: torch, input data
    :param y: torch, output labels
    :return: torch, pearson loss per sample
    """

    mx = torch.mean(x, dim=1, keepdim=True)
    my = torch.mean(y, dim=1, keepdim=True)
    xm, ym = x - mx, y - my

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1-cos(xm,ym))
    return loss

def split_dataset_old(df, split_type='random', key=None, cutoff=0.8, seed=None):
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

def split_dataset(df, split_type='random', split_pattern=None, keys=None, seed=None):
    """
    Splits a dataset based on the specified criteria.

    Parameters:
    - df (DataFrame): The pandas DataFrame to split.
    - split_type (str): Type of split. 'random' for random split; 'fragment' for split based on sequence fragment presence.
    - split_pattern (list of floats): A list of three elements specifying the proportions for train, validation, and test sets (only for random split).
    - keys (tuple of ints): Two keys to look for within Pos1, Pos2, Pos3 columns for 'fragment' split. The first key holds validation, the second key holds testing.
    - seed (int): Seed number for reproducibility.

    Returns:
    - train_set (DataFrame): The training dataset.
    - val_set (DataFrame): The validation dataset (or None if not applicable).
    - test_set (DataFrame): The testing dataset.
    """
    if seed is not None:
        np.random.seed(seed)
        
    if split_type == 'random':
        if split_pattern is None or len(split_pattern) != 3 or not np.isclose(sum(split_pattern), 1):
            raise ValueError("split_pattern must be a list of three elements that add up to 1 for train, validation, and test.")
        train_ratio, val_ratio, test_ratio = split_pattern

        # Handle cases where validation or test is zero
        if val_ratio == 0:
            train_set, test_set = train_test_split(df, test_size=test_ratio, random_state=seed)
            val_set = None
        elif test_ratio == 0:
            train_set, val_set = train_test_split(df, test_size=val_ratio, random_state=seed)
            test_set = None
        else:
            train_set, temp_set = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=seed)
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_set, test_set = train_test_split(temp_set, test_size=(1 - val_ratio_adjusted), random_state=seed)

    elif split_type == 'fragment':
        if keys is None or len(keys) != 2:
            raise ValueError("For 'fragment' split_type, keys must be a tuple with two elements for validation and test keys.")
        key_val, key_test = keys

        # Create masks based on keys for validation and test sets
        val_mask = (df['Pos1'] == key_val) | (df['Pos2'] == key_val) | (df['Pos3'] == key_val)
        test_mask = (df['Pos1'] == key_test) | (df['Pos2'] == key_test) | (df['Pos3'] == key_test)
        
        val_set = df[val_mask]
        test_set = df[test_mask & ~val_mask]  # Ensure no overlap between validation and test
        train_set = df[~(val_mask | test_mask)]  # Exclude validation and test samples from train

    else:
        raise ValueError("Invalid split type specified. Use 'random' or 'fragment'.")

    return train_set, val_set, test_set

class EnhancerDatasetWithID(Dataset):
    def __init__(self, dataset, feature_list, scale_mode='none'):
        """
        Initialize the dataset with the mode specifying which labels to include.
        
        Args:
        feature_list (List): Specifies a list of feature names (column names in the input dataset) 
                            used to train the model. All the feature names need to exist 
                            in the input dataset column names.
        scale_mode (str): Specifies the scaling mode. Can be 'none', '0-1', or '-1-1'.
                              'none' means no scaling, '0-1' scales labels to [0, 1],
                              and '-1-1' scales labels to [-1, 1].
        """

        # Data loading
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

        # Store the fragment IDs
        self.fragment_ids = df['fragment_ids'].values

        # Apply scaling if specified
        if scale_mode == '0-1':
            self.y = scale_labels_0_1(self.y)
        elif scale_mode == '-1-1':
            self.y = scale_labels_minus1_1(self.y)
        elif scale_mode not in ['0-1','-1-1','none']:
            raise ValueError("Invalid scale_mode provided. Choose '0-1', '-1-1', or 'none'.")
        
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.fragment_ids[index]
    
    def __len__(self):
        return self.n_samples


class EnhancerDataset(Dataset):
    def __init__(self, dataset, feature_list, scale_mode='none'):
        """
        Initialize the dataset with the mode specifying which labels to include.
        
        Args:
        feature_list (List): Specifies a list of feature names (column names in the input dataset) 
                            used to train the model. All the feature names need to exist 
                            in the input dataset column names.
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

def evaluate_model(model, test_loader, batch_size, criterion, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0
        for inputs, labels, fragment_ids in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        # Manully devide test_loss by total number of test samples by doing len(test_loader) * batch
        avg_test_loss = test_loss / (len(test_loader) * batch_size)
        avg_test_loss_by_batch = test_loss / len(test_loader)
    return avg_test_loss, avg_test_loss_by_batch

def train_model(model, train_loader, val_loader, test_loader, target_labels, num_epochs=100, batch_size=10, learning_rate=1e-6, criteria = 'mse',optimizer_type = "adam",patience=10, seed = 42, save_data = False, save_model = False, dir_path = None):
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
    val_losses = []
    train_losses_by_batch = []
    val_losses_by_batch = []

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


    # Early stopping specifics
    best_val_loss = float('inf')
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
        for i, (inputs, labels, fragment_ids) in enumerate(train_loader):
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
        avg_val_loss, avg_val_loss_by_batch = evaluate_model(model, val_loader, batch_size, criterion, device)
        val_losses.append(avg_val_loss)
        val_losses_by_batch.append(avg_val_loss_by_batch)

        #print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f} , Test Loss: {avg_val_loss:.4f}")
        #print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss By Batch: {avg_train_loss_by_batch:.4f} , Test Loss By Batch: {avg_val_loss_by_batch:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} -- Train Loss: {avg_train_loss_by_batch:.4f} , Validation Loss: {avg_val_loss_by_batch:.4f}")

        # metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'pearson_corr': [], 'spearman_corr': []}
        metrics = evaluate_regression_model(model, val_loader, device)

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

        # Get pearson correlation and r2 from flattened predictions (not calculated label by label) and labels
        pearson_corr = metrics['pearson_corr'][-1]
        r2 = metrics['r2'][-1]

        if pearson_corr >= best_pearson:
            best_pearson = pearson_corr
            best_pearson_epoch = epoch
        if r2 >= best_r2:
            best_r2 = r2
            best_r2_epoch = epoch
        

        os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), f'{dir_path}/model_epoch_{epoch}.pth')

        # Check if test loss improved
        if avg_val_loss_by_batch < best_val_loss:
            best_val_loss = avg_val_loss_by_batch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check if early stopping is triggered
        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            early_stop = True
            break

    if save_data == True and dir_path != None:
        with open(f'{dir_path}/train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open(f'{dir_path}/test_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f)
        with open(f'{dir_path}/test_losses_by_batch.pkl', 'wb') as f:
            pickle.dump(val_losses_by_batch, f)
        with open(f'{dir_path}/train_losses_by_batch.pkl', 'wb') as f:
            pickle.dump(train_losses_by_batch, f)
        with open(f'{dir_path}/metric_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    print(f"Finished Training!")
    print(f"Best Pearson Correlation is {best_pearson} at epoch {best_pearson_epoch}")
    print(f"Best R2 Square Correlation is {best_r2} at epoch {best_r2_epoch}")

    #############
    # Deleting models in the output directory, except the model at best_pearson and best_r2 epoch respectively
    # Get list of all .pth files in the directory
    model_files = glob.glob(f'{dir_path}/*.pth')
    pearson_metrics, r2_metrics = None, None
    print('\n')
    print('*** Deleting model paths ***')
    # Loop through all model files and process based on best epochs
    for model_file in model_files:
        file_name = os.path.basename(model_file)
        epoch = int(file_name.split('_')[-1].split('.')[0])

        if epoch in [best_pearson_epoch, best_r2_epoch]:
            # Determine if this is the best Pearson or R2 model
            label = "best_pearson" if epoch == best_pearson_epoch else "best_r2"
            print(f"Evaluating model at {label} epoch: {epoch}")
            
            # Load model and evaluate
            model.load_state_dict(torch.load(model_file))
            model.to(device)
            model.eval()
            metrics = evaluate_regression_model(model, test_loader, device)

            # Rename based on whether it's best Pearson, R2, or both
            new_name = f'{dir_path}/{label}_model_epoch_{epoch}.pth'
            os.rename(model_file, new_name)
            print(f"Model at {label} epoch {epoch} is saved as {label}_model")
            
            # Assign metrics
            if epoch == best_pearson_epoch:
                pearson_metrics = metrics
            if epoch == best_r2_epoch:
                r2_metrics = metrics

        else:
            os.remove(model_file)
        #############
    # Clear out directory if save_model is False
    if not save_model:
        for item in os.listdir(dir_path):
            path = os.path.join(dir_path, item)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        print(f"All contents in {dir_path} have been deleted as save_model is set to False.")

    return train_losses, val_losses, model, train_losses_by_batch, val_losses_by_batch,results, best_pearson_epoch, best_r2_epoch, pearson_metrics, r2_metrics, device


def train_model_old(model, train_loader, test_loader, target_labels, num_epochs=100, batch_size=10, learning_rate=1e-6, criteria = 'mse',optimizer_type = "adam",patience=10, seed = 42, save_model = False, dir_path = None):
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
        for i, (inputs, labels, fragment_ids) in enumerate(train_loader):
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

    #############
    # Deleting models in the output directory, except the model at best_pearson and best_r2 epoch respectively
    # Get list of all .pth files in the directory
    model_files = glob.glob(f'{dir_path}/*.pth')

    # Create a flag to check if best_pearson_epoch and best_r2_epoch are the same epoch
    same_epoch = best_pearson_epoch == best_r2_epoch

    print('\n')
    print('*** Deleting model pathes ***')
    # Loop through all model files
    for model_file in model_files:
        # Extract the epoch number from the file name
        file_name = os.path.basename(model_file)
        epoch = int(file_name.split('_')[-1].split('.')[0])

        if epoch == best_pearson_epoch:
            # Rename to 'best_pearson_model_epoch_{epoch}.pth'
            print(f'model at best_pearson_epoch: {best_pearson_epoch} is saved')
            new_name = f'{dir_path}/best_pearson_model_epoch_{epoch}.pth'
            os.rename(model_file, new_name)
            
            # If the best Pearson and R2 epoch are the same, save the same model twice with different names
            if same_epoch:
                print(f'model at best_r2_epoch: {best_r2_epoch} is saved')
                new_name_r2 = f'{dir_path}/best_r2_best_pearson_model_epoch_{epoch}.pth'
                os.rename(new_name, new_name_r2)  # Copy the model file with R2 naming
            continue
        
        if epoch == best_r2_epoch:
            # Rename to 'best_r2_model_epoch_{epoch}.pth' only if it's not already handled by Pearson
            print(f'model at best_r2_epoch: {best_r2_epoch} is saved')
            if not same_epoch:
                new_name = f'{dir_path}/best_r2_model_epoch_{epoch}.pth'
                os.rename(model_file, new_name)
            continue
        
        # If it's neither of the best epochs, delete the file
        os.remove(model_file)
        #############

    return train_losses, test_losses, model, train_losses_by_batch, test_losses_by_batch,results, best_pearson_epoch, best_r2_epoch, device

def evaluate_regression_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, labels, fragment_ids in test_loader:
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

def regression_model_plot(model, test_loader, train_losses_by_batch, test_losses_by_batch, device, results, target_labels, save_plot=False, dir_path=None, model_path=None):
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 12
    # Directory setup
    if save_plot:
        os.makedirs(dir_path, exist_ok=True)
        with open(f'{dir_path}/best_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    actual_value_dict = {target: [] for target in target_labels}  # Initialize dictionaries
    predict_value_dict = {target: [] for target in target_labels}
    
    with torch.no_grad():
        for inputs, labels, fragment_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()  # Get predictions
            
            for i in range(len(target_labels)):
                target = target_labels[i]
                
                # For single target case, flatten the outputs and labels
                if len(target_labels) == 1:
                    predictions = outputs.flatten()
                    actuals = labels.cpu().numpy().flatten()
                else:
                    predictions = outputs[:, i]
                    actuals = labels.cpu().numpy()[:, i]
                
                # Append predictions and actuals to the corresponding target lists
                actual_value_dict[target].extend(actuals)
                predict_value_dict[target].extend(predictions)

    # Convert lists to numpy arrays for easier processing later
    for key in actual_value_dict:
        actual_value_dict[key] = np.array(actual_value_dict[key])
        predict_value_dict[key] = np.array(predict_value_dict[key])

    # Data storage for Excel file
    data_for_excel = {}

    # Plot 1: Training and Testing Loss Over Epochs
    plt.figure(figsize=(6, 6))
    plt.plot(train_losses_by_batch, label='Training Loss')
    plt.plot(test_losses_by_batch, label='Testing Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss Over Epochs', fontsize=14)
    plt.legend()
    if save_plot:
        plt.savefig(os.path.join(dir_path, 'training_testing_loss_by_batch.png'))
    plt.show()

    # Store the data for plot 1 into a dictionary
    data_for_excel['Training and Testing Loss'] = pd.DataFrame({
        'Training Loss': train_losses_by_batch,
        'Testing Loss': test_losses_by_batch
    })

    # Plot 2: Predictions vs Actuals
    num_targets = len(target_labels)
    fig, axs = plt.subplots(1, num_targets, figsize=(num_targets * 6, 6))

    # Ensure axs is always iterable (in case of single plot)
    if num_targets == 1:
        axs = [axs]

    for i in range(num_targets):
        label = target_labels[i]
        actual_list = actual_value_dict[label]
        prediction_list = predict_value_dict[label]
        axs[i].scatter(actual_list, prediction_list, alpha=0.5, s=2)
        axs[i].set_xlabel(f'Actual {label} Values', fontsize=12)
        axs[i].set_ylabel(f'Predicted {label} Values', fontsize=12)
        #axs[i].set_title(f'{label} Predictions vs. Actuals', fontsize=14)
        axs[i].plot([actual_list.min(), actual_list.max()], [actual_list.min(), actual_list.max()], 'k--', lw=2)  # Diagonal line

    if save_plot:
        plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Correlation_Plot.png'))
    plt.show()

    # Store the data for plot 2 into a dictionary
    for label in target_labels:
        data_for_excel[f'{label} Predictions vs Actuals'] = pd.DataFrame({
            f'Actual {label}': actual_value_dict[label],
            f'Predicted {label}': predict_value_dict[label]
        })

    # Plot 3: Histograms of Actual and Predicted Values
    fig, axs = plt.subplots(1, num_targets, figsize=(num_targets * 6, 6))

    # Ensure axs is always iterable (in case of single plot)
    if num_targets == 1:
        axs = [axs]

    for i in range(num_targets):
        label = target_labels[i]
        actual_list = actual_value_dict[label]
        prediction_list = predict_value_dict[label]

        bins = np.histogram_bin_edges(actual_list, bins=50)
        axs[i].hist(actual_list, bins=bins, alpha=0.7, color='red', edgecolor='black', label=f'Actual {label}')
        axs[i].hist(prediction_list, bins=bins, alpha=0.5, color='green', edgecolor='black', label=f'Predicted {label}')
        axs[i].set_xlabel('Value', fontsize=12)
        axs[i].set_ylabel('Frequency', fontsize=12)
        axs[i].set_title(f'Histogram of Actual and Predicted {label} Values', fontsize=14)
        axs[i].legend()

    if save_plot:
        plt.savefig(os.path.join(dir_path, 'Prediction_Actuals_Histogram.png'))
    plt.show()

    # Store the data for plot 3 into a dictionary
    for label in target_labels:
        data_for_excel[f'Histogram of {label}'] = pd.DataFrame({
            f'Actual {label}': actual_value_dict[label],
            f'Predicted {label}': predict_value_dict[label]
        })

    # Plotting Metrics Over Epochs (MSE, RMSE, MAE, R², Pearson, Spearman)
    for label, metrics in results.items():
        # Plotting MSE, RMSE, MAE
        fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
        axs1[0].plot(metrics['mse'], label='MSE')
        axs1[0].set_title(f'Mean Squared Error: {label}', fontsize=14)
        axs1[0].set_xlabel('Epoch', fontsize=12)
        axs1[0].set_ylabel('MSE', fontsize=12)
        axs1[0].legend()

        axs1[1].plot(metrics['rmse'], label='RMSE')
        axs1[1].set_title(f'Root Mean Squared Error: {label}', fontsize=14)
        axs1[1].set_xlabel('Epoch', fontsize=12)
        axs1[1].set_ylabel('RMSE', fontsize=12)
        axs1[1].legend()

        axs1[2].plot(metrics['mae'], label='MAE')
        axs1[2].set_title(f'Mean Absolute Error: {label}', fontsize=14)
        axs1[2].set_xlabel('Epoch', fontsize=12)
        axs1[2].set_ylabel('MAE', fontsize=12)
        axs1[2].legend()

        fig1.tight_layout(pad=3.0)
        if save_plot:
            plt.savefig(os.path.join(dir_path, f'mse_rmse_mae_{label}.png'))
        plt.show()

        # Store MSE, RMSE, MAE data for Excel
        data_for_excel[f'Metrics over Epochs: {label}'] = pd.DataFrame({
            'Epoch': list(range(len(metrics['mse']))),
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })

        # Plotting R², Pearson, and Spearman
        fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
        axs2[0].plot(metrics['r2'], label='R² Score')
        axs2[0].set_title(f'R² Score: {label}', fontsize=14)
        axs2[0].set_xlabel('Epoch', fontsize=12)
        axs2[0].set_ylabel('R²', fontsize=12)
        axs2[0].legend()

        axs2[1].plot(metrics['pearson_corr'], label='Pearson Correlation')
        axs2[1].set_title(f'Pearson Correlation: {label}', fontsize=14)
        axs2[1].set_xlabel('Epoch', fontsize=12)
        axs2[1].set_ylabel('Pearson', fontsize=12)
        axs2[1].legend()

        axs2[2].plot(metrics['spearman_corr'], label='Spearman Correlation')
        axs2[2].set_title(f'Spearman Correlation: {label}', fontsize=14)
        axs2[2].set_xlabel('Epoch', fontsize=12)
        axs2[2].set_ylabel('Spearman', fontsize=12)
        axs2[2].legend()

        fig2.tight_layout(pad=3.0)
        if save_plot:
            plt.savefig(os.path.join(dir_path, f'r2_pearson_spearman_{label}.png'))
        plt.show()

        # Store R², Pearson, Spearman data for Excel
        data_for_excel[f'Correlations over Epochs: {label}'] = pd.DataFrame({
            'Epoch': list(range(len(metrics['r2']))),
            'R²': metrics['r2'],
            'Pearson': metrics['pearson_corr'],
            'Spearman': metrics['spearman_corr']
        })

    # Save all the data used for plotting into an Excel file
    if save_plot:
        with pd.ExcelWriter(os.path.join(dir_path, 'model_performance_data.xlsx')) as writer:
            for sheet_name, data in data_for_excel.items():
                data.to_excel(writer, sheet_name=sheet_name)

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