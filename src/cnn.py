import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils import *

def train_cnn_model(X_train, y_train, X_test, y_test, device, X_desc=None, y_desc=None):

    ## Hyper parameter

    l_r = 0.0001
    stop_threshold = 0.005
    num_epochs = 100
    b_size = 256
    print(f'Learning Rate: {l_r}; Batch size: {b_size}; Num epochs: {num_epochs}; Stop threshold: {stop_threshold}')
    num_classes = len(y_train.unique())
    X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
    X_train = torch.tensor(X_train, dtype=torch.float32)  
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    train_data = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_data, batch_size=b_size, shuffle=True, drop_last=True)

    X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)
    test_data = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_data, batch_size=b_size, shuffle=True, drop_last=True)

    if X_desc is not None:
        X_desc = np.array(X_desc).reshape(X_desc.shape[0], 1, X_desc.shape[1])
        X_desc = torch.tensor(X_desc, dtype=torch.float32)
        y_desc = torch.tensor(np.array(y_desc), dtype=torch.long)
        desc_data = TensorDataset(X_desc, y_desc)
        desc_dl = DataLoader(desc_data, batch_size=b_size, shuffle=True, drop_last=True)
    
    class CNN(nn.Module):

        def __init__(self, input_size, num_classes):
            print('input_size', input_size)
            super(CNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            
            # Ensure correct input size for fc1
            self.fc1 = nn.Linear((input_size // 4) * 128, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x) 
            x = torch.relu(self.conv2(x))
            x = self.pool(x)  

            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNN(X_train.shape[2], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_r)
    model.train()
    acc = {key: None for key in range(5, num_epochs + 1, 1)}
    acc_desc = {key: None for key in range(5, num_epochs + 1, 1)}
    
    prev_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (batch_X, batch_y) in enumerate(train_dl):
            # print(f"Epoch {epoch}, Batch {batch_idx}, Input shape: {batch_X.shape}")
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        # Early stopping condition
        if prev_loss - avg_loss < stop_threshold:
            print(f"Early stopping at epoch {epoch} with loss: {avg_loss:.4f}")
            acc_1 = predict_dl(model, test_dl, device)
            acc[epoch+1] = round(acc_1, 4)
            if X_desc is not None:
                acc_2 = predict_dl(model, desc_dl, device)
                acc_desc[epoch+1] = round(acc_2, 4)
                print(f"Acc_task: {acc_1:.4f}, Acc_desc: {acc_2:.4f}") 
            else:
                print(f"Acc_lang: {acc_1:.4f}") 
            break
        prev_loss = avg_loss
        if (epoch + 1) % 5 == 0:
            acc_1 = predict_dl(model, test_dl, device)
            acc[epoch+1] = round(acc_1, 4)
            if X_desc is not None:
                acc_2 = predict_dl(model, desc_dl, device)
                acc_desc[epoch+1] = round(acc_2, 4)
                print(f"Acc_task: {acc_1:.4f}, Acc_desc: {acc_2:.4f}") 
            else:
                print(f"Acc_lang: {acc_1:.4f}") 
    return model, acc, acc_desc


def predict_dl(model, data_dl, device):   
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in data_dl:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            y_pred = torch.argmax(outputs, dim=1)
            c = (y_pred == batch_y).sum().item()
            correct += c
            total += batch_y.size(0)
    accuracy = correct / total
    return accuracy

# def predict(model, X_test, y_test, device):   
#     model.eval()
#     with torch.no_grad():
#         if X_test.dim() == 2: 
#             X_test = X_test.unsqueeze(1)
#         outputs = model(X_test)
#         y_pred = torch.argmax(outputs, dim=1)
#         accuracy = (y_pred == y_test).float().mean().item()
#     return accuracy
