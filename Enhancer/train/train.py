import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from Enhancer.train.utils import EnhancerDataset, split_dataset, train_model, evaluate_regression_model
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.modules.activation as activation
import sys

import matplotlib.pyplot as plt
import os
import pickle

sys.path.append('../model')  
from model import ConvNetDeep, DanQ, ExplaiNN

def main(args):
    # Directory setup
    os.makedirs(args.dir_path, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_path)

    # Split dataset
    train, test = split_dataset(df, split_type=args.split_type, key= args.split_key, cutoff=args.split_cutoff, seed=args.seed)


    # Process datasets
    train_dataset = EnhancerDataset(train, label_mode=args.label_mode, scale_mode=args.scale_mode)
    test_dataset = EnhancerDataset(test, label_mode=args.label_mode, scale_mode=args.scale_mode)

    # DataLoader setup
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    # Model selection
    if args.model_name == 'ConvNetDeep':
        model = ConvNetDeep(input_length = args.input_length, num_classes=args.num_classes, drop_out=args.dropout_rate)
    elif args.model_name == 'DanQ':
        model = DanQ(input_length=args.input_length, num_classes=args.num_classes)
    elif args.model_name == "ExplaiNN":
        model = ExplaiNN(num_cnns = args.num_cnns, input_length = args.input_length, num_classes = args.num_classes,filter_size=19, num_fc=1)
    else:
        raise ValueError("Unsupported model type")

    # Training (assuming train_model function is defined)
    train_losses, test_losses, model, train_losses_by_batch, test_losses_by_batch = train_model(
        model, train_loader, test_loader, num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, criteria=args.criteria, optimizer_type=args.optimizer_type, patience=args.patience,
        save_model=args.save_model, dir_path=args.dir_path
    )

    # Evaluate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse, rmse, mae, r2 = evaluate_regression_model(model, test_loader, train_losses, test_losses, train_losses_by_batch, test_losses_by_batch, device, save_plot = args.save_plot, dir_path = args.dir_path)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

    # Save results
    results = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}
    with open(os.path.join(args.dir_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate Enhancer Models")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--model_name", type=str, choices=['ConvNetDeep', 'DanQ','ExplaiNN'], required=True, help="Model to use")
    parser.add_argument("--input_length", type=int, default=608, help="Input length for the DanQ model")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes or predictions")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for drop out layers")
    parser.add_argument("--num_cnns", type=int, default=100, help="Number of cnns used in ExplaiNN")
    parser.add_argument("--label_mode", type=str, default='G+', choices=['G+', 'G-','both'], help="Type of labels to have")
    parser.add_argument("--scale_mode", type=str, default='none', choices=['none', '-1-1','1-0'], help="Type of scaling of the labels to have")
    parser.add_argument("--split_type", type=str, default='random', choices=['random', 'fragment'], help="Type of dataset split")
    parser.add_argument("--split_key", type=int, default=None, help="When split_type is fragment, this parameter specifies the fragement_id to hold as test dataset")
    parser.add_argument("--split_cutoff", type=float, default=0.8, help="When split_type is random, this parameter specifies the percentage of data randomly drawn for training")
    parser.add_argument("--seed", type=int, default=46, help="Seed for random operations")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--criteria", type=str, default='mse', choices=['bcewithlogits', 'crossentropy','mse','pearson','poissonnll','huber'], help="Type of loss function")
    parser.add_argument("--optimizer_type", type=str, default='adam', choices=['adam', 'sgd'], help="Type of optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Patience for pre-stop of training")
    parser.add_argument("--save_model", default=False, help="Flag to save the model checkpoints")
    parser.add_argument("--save_plot", default=False, help="Flag to save the evaluation plots")
    args = parser.parse_args()

    main(args)




