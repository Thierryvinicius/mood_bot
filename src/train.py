
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Net
from config import *

def train_model(model, X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    MODEL_PATH = 'src/models/'
    torch.save(model.state_dict(), MODEL_PATH + 'text_classifier.pth')
    print('MODELO SALVO EM ', MODEL_PATH)
