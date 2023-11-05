from typing import Any
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from lightning import LightningModule
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningModule

dataframes = []

for filename in glob.glob("/home/makt1x/chess-AI/pytorch-test/Data/ETFs/*.txt"):
    df = pd.read_csv(filename)
    dataframes.append(df)
merged_df = pd.concat(dataframes, ignore_index=True)
merged_df = merged_df.sort_values(by='Date')
final_array = merged_df.to_numpy()

Date = final_array[:, 0]
High = final_array[:, 1]
Low = final_array[:, 2]
Close = final_array[:, 3]
Volume = final_array[:, 4]
OpenInt = final_array[:, 5]

X = np.column_stack((High, Low, Close, Volume, OpenInt))
y = Close

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

class Net(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1, 1)
        loss = self.loss(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1, 1)
        loss = self.loss(y_hat, y.float())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = Net()
trainer = pl.Trainer(accelerator = "gpu", max_epochs=10)
trainer.fit(model, torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=64, num_workers=16))
