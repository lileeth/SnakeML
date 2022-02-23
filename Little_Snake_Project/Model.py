# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 23:07:17 2021

@author: ericl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Funct
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Funct.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, stato, azione, ricompensa, next_stato, game_over):
        stato = torch.tensor(stato, dtype=torch.float)
        next_stato = torch.tensor(next_stato, dtype=torch.float)
        azione = torch.tensor(azione, dtype=torch.long)
        ricompensa = torch.tensor(ricompensa, dtype=torch.float)
        # (n, x)

        if len(stato.shape) == 1:
            # (1, x)
            stato = torch.unsqueeze(stato, 0)
            next_stato = torch.unsqueeze(next_stato, 0)
            azione = torch.unsqueeze(azione, 0)
            ricompensa = torch.unsqueeze(ricompensa, 0)
            game_over = (game_over, )

        # 1: predizione lato QNET dato uno stato in input
        pred = self.model(stato)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = ricompensa[idx]
            if not game_over[idx]:
                Q_new = ricompensa[idx] + self.gamma * torch.max(self.model(next_stato[idx]))

            target[idx][torch.argmax(azione[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()