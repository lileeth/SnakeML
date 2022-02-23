# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# colori del gioco
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 40
SPEED = 50

class AISnake:

    def __init__(self, w=1280, h=960):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Little Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game stato
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.punteggio = 0
        self.cibo = None
        self._crea_cibo()
        self.frame_iteration = 0


    def _crea_cibo(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.cibo = Point(x, y)
        if self.cibo in self.snake:
            self._crea_cibo()


    def play_step(self, azione):    #algoritmo principale
        self.frame_iteration += 1   #numero passaggi
        # 1. acquisisci informazioni stato precedente
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. direzione
        self._mossa(azione) # update the head
        self.snake.insert(0, self.head)
        
        # 3. controlla se game over
        ricompensa = 0
        game_over = False
        if self.controllo_urto() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            ricompensa = -10
            return ricompensa, game_over, self.punteggio

        # 4. crea il nuovo cibo oppure muoviti in una direzione
        if self.head == self.cibo:
            self.punteggio += 1
            ricompensa = 10
            self._crea_cibo()
        else:
            self.snake.pop()
        
        # 5. aggiorna ui e clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. fornisci dati della ricompensa, game over e punteggio
        return ricompensa, game_over, self.punteggio


    def controllo_urto(self, pt=None):        #controllo urto
        if pt is None:
            pt = self.head
        # colpisce il perimetro
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # colpisce se stesso
        if pt in self.snake[1:]:    #se la testa è in una posizione già registrata all' interno del serpente allora si è scontrato
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)    #colore sfondo

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  #colore esterno
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+(BLOCK_SIZE/5), pt.y+(BLOCK_SIZE/5), BLOCK_SIZE-(2*BLOCK_SIZE/5), BLOCK_SIZE-(2*BLOCK_SIZE/5)))  #colore interno

        pygame.draw.rect(self.display, RED, pygame.Rect(self.cibo.x, self.cibo.y, BLOCK_SIZE, BLOCK_SIZE)) #Colore mela

        text = font.render("Punteggio: " + str(self.punteggio), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _mossa(self, azione):
        # [continua dritto, gira destra, gira sinistra]

        possibili_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = possibili_dir.index(self.direction)

        if np.array_equal(azione, [1, 0, 0]):
            new_dir = possibili_dir[idx] # vai dritto
        elif np.array_equal(azione, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = possibili_dir[next_idx] # gira a destra quindi scandisci da sinistra verso destra il vettore possibili direzioni
        else:
            next_idx = (idx - 1) % 4
            new_dir = possibili_dir[next_idx] # gira a sinistra quindi scandisci da destra verso sinistra il vettore possibili direzioni

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:  #Stranamente la y aumenta andando verso il basso perchè il punto più in alto ha y=0
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:    #qua la y diminuisce perchè si avvicina all'alto e quindi a 0
            y -= BLOCK_SIZE

        self.head = Point(x, y)