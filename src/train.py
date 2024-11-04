import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Net
from config import *


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, learning_rate=LEARNING_RATE):


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')
            val_accuracy = validate_model(model, X_val, y_val)
            print(f'Validation Accuracy: {val_accuracy*100:.2f}%')
            print('='*50)
    MODEL_PATH = os.path.join(os.getcwd(),"models")
    torch.save(model.state_dict(), MODEL_PATH + 'text_classifier.pth')
    print('MODELO SALVO EM ', MODEL_PATH)

def validate_model(model, X_val, y_val):
    model.eval()  # Muda para modo de avaliação
    with torch.no_grad():
        dataset_val = TensorDataset(X_val, y_val)
        loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0
        correct = 0
        loss_fn = nn.NLLLoss()

        for batch_x, batch_y in loader_val:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(loader_val)
        accuracy = correct / len(X_val)
        
        # Exibir a perda de validação se desejar
        print(f'Validation Loss: {avg_loss:.4f}')
        
        return accuracy


def model_test(model, X_test, y_test):
    model.eval()  # Muda para modo de avaliação
    with torch.no_grad():
        dataset_test = TensorDataset(X_test, y_test)
        loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0
        correct = 0
        loss_fn = nn.NLLLoss()

        for batch_x, batch_y in loader_test:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(loader_test)
        accuracy = correct / len(X_test)
        print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
