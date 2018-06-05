# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:14:49 2018

@author: HRB
"""

from Board_env import Board
from RL_brain import DeepQNetwork


def playgame():
    step = 0
    color = "Black"
    for episode in range(500):
        # initial observation
        observation = env.reset()

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
                observation = observation_
                
                step += 1

            # break while loop when end of this episode
            if done:
                break
            

    # end of game
    print('game over')
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
                      # output_graph=True
                      )
    env.after(10, playgame)
    env.mainloop()
    RL.plot_cost()