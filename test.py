# Model
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Set random seed and Use 'cuda' GPU

torch.manual_seed(0)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(0)

else:
    device = 'cpu'
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc5(out)
        return out

    def predict(model, input_data):
        model = Net()
        model.eval()
        with torch.no_grad():
            # Предобработка входных данных (предполагается, что input_data - это список)
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(
                device)  # Преобразование в тензор и добавление размерности батча

            # Получение предсказания от модели
            output = model(input_tensor)

            # Извлечение индекса класса с наивысшей вероятностью
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    """def predict(model, input_data):
        #model = Net()
        model.eval()
        with torch.no_grad():
            # Предобработка входных данных (предполагается, что input_data - это список)
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)  # Преобразование в тензор и добавление размерности батча

            # Получение предсказания от модели
            output = model(input_tensor)

            # Извлечение индекса класса с наивысшей вероятностью
            prediction = torch.argmax(output, dim=1).item()

        return prediction
    #with open('t', 'rb') as model_file:
        #model = pickle.load(model_file)

    model = torch.jit.load('model_scripted.pt')
    model.eval()

    input_data = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0]
    prediction = predict(model, input_data)"""