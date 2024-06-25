import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm
import pandas as pd
from timeit import default_timer as timer 

from model import RegressionModel
from utils import print_train_time
from sklearn.preprocessing import StandardScaler

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir', type=str, help='path to the data file')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for processing (default: 64)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs for processing (default: 80)')
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    data_dir = args.data_dir

    # Load data from CSV
    df_house = pd.read_csv(data_dir) 

    # Build dataset
    data = df_house.drop("median_house_value", axis=1)
    label = df_house["median_house_value"]
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Standardize the labels
    label_mean = label.mean()
    label_std = label.std()
    label_scaled = (label - label_mean) / label_std

    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    label_tensor = torch.tensor(label_scaled.values, dtype=torch.float32)

    dataset = TensorDataset(data_tensor, label_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # init model
    model = RegressionModel(input_size = 8, hidden_size = 80)
    model.to("cpu")
    optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum = 0.9)  
    criterion = nn.MSELoss()

    train_time_start_on_cpu = timer()
    
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")

        # ----- train -----         
        train_loss = 0.0
        for X, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y.unsqueeze(1))  # MSE loss for regression
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)
        
        # ----- test -----         
        test_loss = 0.0
        mae = 0.0
        mse = 0.0
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                test_pred = model(X)
                test_loss += criterion(test_pred, y.unsqueeze(1)).item() * X.size(0)
                mae += torch.abs(test_pred - y.unsqueeze(1)).sum().item()  # MAE
                mse += ((test_pred - y.unsqueeze(1)) ** 2).sum().item()  # MSE
        test_loss /= len(test_loader.dataset)
        mae /= len(test_loader.dataset)
        mse /= len(test_loader.dataset)
        ## Print training status
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}")

    train_time_end_on_cpu = timer()
    print_train_time(
        start=train_time_start_on_cpu, 
        end=train_time_end_on_cpu,
        device=str(next(model.parameters()).device)
    )

if __name__ == "__main__":
    main()
