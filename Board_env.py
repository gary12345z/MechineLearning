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
WINREWARD = 1

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
        self.SIZE = SIZE

    def newgame(self):  
        # background
        self.canvas = tk.Canvas(self, bg='white',
                           height=SIZE * UNIT,
                           width=SIZE * UNIT)

        # pack all
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(0.5)
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.GAMEBOARD = np.zeros((SIZE,SIZE))
        self.counter = 0
        return self.GAMEBOARD.flatten()

    def step(self, location , color):
        
        X = int(location / SIZE)
        Y = location % SIZE
        s_ = self.GAMEBOARD.flatten()
        reward = 0
        done = False
        error = False
        
        print(color,"Goto: (",X,",",Y,")")
        
        if(self.GAMEBOARD[X][Y]!=0):
            error = True
            return s_, reward, done , color , error
        
        self.GAMEBOARD[X][Y] = 1
        
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
        
        #print(GAMEBOARD)
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
            print(color," Win");
            reward = WINREWARD
            done = True
        self.GAMEBOARD = self.GAMEBOARD * -1;
        return s_, reward, done , nextcolor[color] ,error


def update():
    color = "Black"
    while True:
        s_, reward, done , ncolor , error = board.step(int(np.random.uniform()*SIZE*SIZE) , color)
        if(not error):
            print(s_, reward, done , ncolor , error)
        color = ncolor
        if(done):
            break


if __name__ == '__main__':
    board = Board()
    board.after(10, update)
    board.mainloop()