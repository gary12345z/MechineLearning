# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:19:55 2018

@author: HRB
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
BASE = int(UNIT/2)
SIZE = 3  # grid height
WINNUM = 3

strtonum = {'Black':1,'White':2}
nextcolor = {'Black':'White','White':'Black'}

class Board(tk.Tk, object):
    def __init__(self):
        super(Board, self).__init__()
        self.title('Game Board')
        self.geometry('{0}x{1}'.format(SIZE * UNIT, SIZE * UNIT))
        
        self.newgame()  #clear the board
        self.GAMEBOARD = np.zeros((SIZE,SIZE))  #board
        self.counter = 0  #if the game is draw 
        self.n_actions = SIZE*SIZE  #number of the action you can choose
        self.n_features = SIZE*SIZE #number of the feature

    def newgame(self):
        
        # background
        self.canvas = tk.Canvas(self, bg='white',
                           height=SIZE * UNIT,
                           width=SIZE * UNIT)

        # draw line
        for i in range(0, SIZE * UNIT, UNIT):
            x0, y0, x1, y1 = i, 0, i, SIZE * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
            x0, y0, x1, y1 = 0, i, SIZE * UNIT, i
            self.canvas.create_line(x0, y0, x1, y1)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        #self.canvas.delete(self.black)
        #self.canvas.delete(self.white)
        self.GAMEBOARD = np.zeros((SIZE,SIZE))
        self.counter = 0
        return self.GAMEBOARD.flatten()

    def step(self, location , color):
        
        X = location % SIZE
        Y = int(location / SIZE)
        s_ = self.GAMEBOARD
        reward = 0
        done = False
        error = False
        
        print(location , color, self.GAMEBOARD[X][Y])
        
        if(self.GAMEBOARD[X][Y]!=0):
            error = True
            return s_, reward, done , color , error
        
        self.GAMEBOARD[X][Y]=strtonum[color];
        self.counter = self.counter+1
        
        if (color=='Black'):
            self.white = self.canvas.create_oval(
                    BASE + Y*UNIT - 15, BASE + X*UNIT - 15,
                    BASE + Y*UNIT + 15, BASE + X*UNIT + 15,
                    fill='black')
        elif(color=='White'):
            self.white = self.canvas.create_oval(
                    BASE + Y*UNIT - 15, BASE + X*UNIT - 15,
                    BASE + Y*UNIT + 15, BASE + X*UNIT + 15,
                    fill='white')
        self.update()
        
        s_ = self.GAMEBOARD.flatten()
        winner = 0;
        GAMEBOARD = self.GAMEBOARD
        
        print(GAMEBOARD)
        #右方向
        for i in range(0,SIZE-WINNUM+1):
            for j in range (0,SIZE):
                if(GAMEBOARD[i][j] != 0):
                    for k in range(1,WINNUM):
                        if(GAMEBOARD[i+k][j] != GAMEBOARD[i][j]):
                            break
                    if(GAMEBOARD[i+k][j] == GAMEBOARD[i][j]):
                        winner = GAMEBOARD[i][j]
        #下方向
        for i in range(0,SIZE):
            for j in range (0,SIZE-WINNUM+1):
                if(GAMEBOARD[i][j] != 0):
                    for k in range(1,WINNUM):
                        if(GAMEBOARD[i][j+k] != GAMEBOARD[i][j]):
                            break
                    if(GAMEBOARD[i][j+k] == GAMEBOARD[i][j]):
                        winner = GAMEBOARD[i][j]
        #右下方向
        for i in range(0,SIZE-WINNUM+1):
            for j in range (0,SIZE-WINNUM+1):
                if(GAMEBOARD[i][j] != 0):
                    for k in range(1,WINNUM):
                        if(GAMEBOARD[i+k][j+k] != GAMEBOARD[i][j]):
                            break
                    if(GAMEBOARD[i+k][j+k] == GAMEBOARD[i][j]):
                        winner = GAMEBOARD[i][j]
        #右上方向
        for i in range(0,SIZE-WINNUM+1):
            for j in range (WINNUM-1,SIZE):
                if(GAMEBOARD[i][j] != 0):
                    for k in range(1,WINNUM):
                        if(GAMEBOARD[i+k][j-k] != GAMEBOARD[i][j]):
                            break
                    if(GAMEBOARD[i+k][j-k] == GAMEBOARD[i][j]):
                        winner = GAMEBOARD[i][j]
        if (self.counter == SIZE*SIZE):
            reward = 0
            done = True
        if (winner == 1):
            reward = 10
            done = True
        elif(winner == 2):
            reward = -10
            done = True
        
        return s_, reward, done , nextcolor[color] ,error


def update():
    color = "Black"
    while True:
        s_, reward, done , ncolor , error = board.step(int(np.random.uniform()*SIZE*SIZE) , color)
        color = ncolor
        if(done):
            break


if __name__ == '__main__':
    board = Board()
    board.after(10, update)
    board.mainloop()