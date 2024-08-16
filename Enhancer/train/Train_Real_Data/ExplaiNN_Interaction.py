# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

sys.path.append('../../../Enhancer')  
from train.utils import EnhancerDataset, split_dataset, train_model, regression_model_plot, plot_filter_weight
from model.model import ExplaiNN3_interact

# Import or define the functions and classes
# Make sure these are available or define them if not
# from your_module import split_dataset, EnhancerDataset, ConvNetDeep, train_model, evaluate_regression_model

#-------------------------------------
#*********Train ExplaiNN3_interact************
#************Predict GFP+**************
#-------------------------------------

# Define some hyperparameters
seed = 42
batch = 200
num_cnns = 20
learning_rate = 4e-4
target_labels = ['GFP']
output_dir = '/pmglocal/ty2514/Enhancer/Enhancer/data/GFP_Pred_Results/ExplaiNN_Interaction_results'


print("Now reading the synthetic sequence data file")
df = pd.read_csv('/pmglocal/ty2514/Enhancer/Enhancer/data/input_data.csv')

print("Success!")
print('')
print("Now splitting data and construct data loader")
train, test = split_dataset(df, split_type='random', cutoff = 0.8, seed = seed)

train = EnhancerDataset(train, feature_list=['GFP'], scale_mode = 'none')
test = EnhancerDataset(test, feature_list=['GFP'], scale_mode = 'none')

# DataLoader setup
train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)
print("Success!")
print('')
print("Now start training")

input_model = ExplaiNN3_interact(num_cnns = num_cnns, input_length = 608, num_classes = 1, 
                 filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                 drop_out = 0.3, weight_path = None)# Training

_, _, model, train_losses_by_batch, test_losses_by_batch, results, best_pearson_epoch, best_r2_epoch, device  = train_model(input_model, train_loader, test_loader, 
                                                                                                                            target_labels=target_labels,num_epochs=200, 
                                                                                                                        batch_size=batch, learning_rate=learning_rate, 
                                                                                                                        criteria='mse',optimizer_type = "adam", patience=10, 
                                                                                                                        seed = seed, save_model= True, dir_path=output_dir)

print("\n")
print("\n")
print("\n")
print("Now start plotting")

model_path = f'/pmglocal/ty2514/Enhancer/Enhancer/data/GFP_Pred_Results/ExplaiNN_Interaction_results/model_epoch_{best_r2_epoch}.pth'
dir_path = '/pmglocal/ty2514/Enhancer/Enhancer/data/GFP_Pred_Results/ExplaiNN_Interaction_results'
mse, rmse, mae, r2, pearson_corr, spearman_corr = regression_model_plot(
    model, test_loader, train_losses_by_batch, test_losses_by_batch, 
    device, results, label_mode = "score", save_plot = True, dir_path = dir_path, model_path = model_path, best_model=best_r2_epoch)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")