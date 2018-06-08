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
    color = "Black"
    for episode in range(50):
        # initial observation
        observation = env.reset()
        
        print("Game ",episode," Start~~~")
        
        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            if(color == 'Black'):
                action = RL.choose_action_(observation)
            else:
                while(observation[action]!=0):
                    action = np.random.randint(0, 3*3)

            # RL take action and get next observation and reward
            observation, observation_, reward, done, ncolor , error = env.step(action , color)
            
            if(color != ncolor):
                color = ncolor
                if(color == "White"):
                    RL.store_transition(observation, action, reward, observation_)
                
                    if(reward>0):
                        RL.changereward(reward*-1)

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
    color = "Black"
    for episode in range(2):
        observation = env.reset()
        
        while True:
            if(color == 'Black'):
                action = RL.choose_action_(observation)
            else:
                while(observation[action]!=0):
                    action = np.random.randint(0, 3*3)
            # RL take action and get next observation and reward
            observation, observation_, reward, done, ncolor , error = env.step(action , color)
            
            if(color != ncolor):
                color = ncolor
                observation = observation_*-1
            else:
                print("ERROR")

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
    observation=np.array([0,0,0,0,0,0,0,0,0])
    RL.choose_action_(observation)
    observation=np.array([0,0,0,0,0,0,0,0,1])
    RL.choose_action_(observation)
    observation=np.array([0,0,0,0,0,0,0,-1,1])
    RL.choose_action_(observation)
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Board()
    
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.1,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(10, playgame)
    env.mainloop()
    RL.plot_cost()