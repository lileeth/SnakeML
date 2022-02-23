# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:07:58 2021

@author: ericl
"""
import torch
import random
import numpy as np
from collections import deque
from SnakeSetup import AISnake, Direction, Point, BLOCK_SIZE
from Model import Linear_QNet, QTrainer
from Display import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class MachineLearning:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomicità
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_stato(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        stato = [
            # Danger straight
            (dir_r and game.controllo_urto(point_r)) or 
            (dir_l and game.controllo_urto(point_l)) or 
            (dir_u and game.controllo_urto(point_u)) or 
            (dir_d and game.controllo_urto(point_d)),

            # Danger right
            (dir_u and game.controllo_urto(point_r)) or 
            (dir_d and game.controllo_urto(point_l)) or 
            (dir_l and game.controllo_urto(point_u)) or 
            (dir_r and game.controllo_urto(point_d)),

            # Danger left
            (dir_d and game.controllo_urto(point_r)) or 
            (dir_u and game.controllo_urto(point_l)) or 
            (dir_r and game.controllo_urto(point_u)) or 
            (dir_l and game.controllo_urto(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # cibo location 
            game.cibo.x < game.head.x,  # cibo left
            game.cibo.x > game.head.x,  # cibo right
            game.cibo.y < game.head.y,  # cibo up
            game.cibo.y > game.head.y  # cibo down
            ]

        return np.array(stato, dtype=int)

    def remember(self, stato, azione, ricompensa, next_stato, fine):
        self.memory.append((stato, azione, ricompensa, next_stato, fine)) # se pieno toglie i dati vecchi facendo un pop left

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # lista di tuple
        else:
            mini_sample = self.memory

        stati, azioni, ricompense, futuri_stati, game_overs = zip(*mini_sample)
        self.trainer.train_step(stati, azioni, ricompense, futuri_stati, game_overs)
        #for stato, azione, ricompensa, next_stato, fine in mini_sample:
        #    self.trainer.train_step(stato, azione, ricompensa, next_stato, fine)

    def train_short_memory(self, stato, azione, ricompensa, next_stato, fine):
        self.trainer.train_step(stato, azione, ricompensa, next_stato, fine)

    def get_azione(self, stato):
        # random moves: tradeoff exploration / exploitation
        if self.n_games<=80:
            self.epsilon = 80 - self.n_games
        else:
            self.epsilon = 0;
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            stato0 = torch.tensor(stato, dtype=torch.float)
            prediction = self.model(stato0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = MachineLearning()
    game = AISnake()
    while True:
        # get prec stato
        stato_prec = agent.get_stato(game)

        # get move
        final_move = agent.get_azione(stato_prec)

        # perform move and get new stato
        ricompensa, fine, score = game.play_step(final_move)
        stato_new = agent.get_stato(game)

        # train short memory
        agent.train_short_memory(stato_prec, final_move, ricompensa, stato_new, fine)

        # remember
        agent.remember(stato_prec, final_move, ricompensa, stato_new, fine)

        if fine:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()