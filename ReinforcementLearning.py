# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:14:49 2018

@author: HRB
"""

import numpy as np
from Board_env import Board
from RL_brain import DeepQNetwork


def playgame():
    step = 0
    
    for episode in range(3000):
        # initial observation
        observation = env.reset()
        color = "Black"
        print("Game ",episode," Start~~~")
        
        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, ncolor , error = env.step(action , color)
            
            if(color != ncolor):
                color = ncolor

                RL.store_transition(observation, action, reward, observation_)

                if (step > 200) and (step % 5 == 0):
                    RL.learn()

            # swap observation
                observation = observation_*-1
                
                step += 1

            # break while loop when end of this episode
            if done:
                break
            
    # end of training
    print('Play with bot')
    
    black = 0
    white = 0
    drew = 0
    
    for episode in range(200):
        observation = env.reset()
        color = "Black"
        nochoice = False
        while True:
            if(color == 'Black'):
                action = RL._choose_action(observation,nochoice)
            else:
                action = int(np.random.uniform()*3*3)

            # RL take action and get next observation and reward
            observation_, reward, done, ncolor , error = env.step(action , color)
            
            if(color != ncolor):
                color = ncolor
                nochoice = False
            elif(color == "Black"):
                nochoice = True

            # break while loop when end of this episode
            if done:
                if( reward > 0 and ncolor == "White"):
                    black = black +1
                elif( reward > 0 and ncolor == "Black"):
                    white = white +1
                else :
                    drew = drew +1
                break
    
    print("黑方勝率:",black/2,"%")
    print("白方勝率:",white/2,"%")
    print("平手:",drew/2,"%")
    
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Board()
    
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(10, playgame)
    env.mainloop()
    RL.plot_cost()