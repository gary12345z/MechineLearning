from Board_env import Board
import numpy as np  
from DQN import DQN
import time  

EPISODE = 1500 # Episode limitation  
TEST = 200
STEP = 300 # Step limitation in an episode  

color_to_int={"Black" : -1 ,"White" : 1}

def main():  
    # initialize OpenAI Gym env and dqn agent  
    agent = DQN(env)
  
    agent.copyWeightsToTarget()  
  
    for episode in range(EPISODE):  
        # initialize task  
        state = env.reset()  
        color = "Black"
        print('episode ',episode)  
  
        # Train  
        for step in range(STEP):  
            #自己下一步棋  
            action = agent.egreedy_action(state) # e-greedy action for train  
            #if env.env.is_valid_set_coord(action[0],action[1]):  
            next_state,reward,done,next_color,_ = env.step(action,color)
            color = next_color
            # Define reward for agent
            agent.perceive(state,action,reward,next_state,done)  
            state = next_state * -1
            if done:  
                print('done step ',step)  
                break  
            
    white = 0
    black = 0
    drew = 0
    for i in range(TEST):  
        state = env.reset()  
        color = "Black"
        
        for step in range(STEP):  
            #env.render()
            if(color=="Black"):
                action = agent.action(state) # direct action for test  
            else:
                action = agent.random_action(state)
            
            next_state,reward,done,next_color,_ = env.step(action,color) 
            color = next_color
            state = next_state * -1
            
            if done:  
                #env.render()  
                if(reward>0 and next_color=="Black"):
                    white=white+1
                elif(reward>0 and next_color=="White"):
                    black=black+1
                else:
                    drew=drew+1
                print('done')  
                time.sleep(3)  
                break  
    print("黑方勝率:",black/TEST*100,"%")
    print("白方勝率:",white/TEST*100,"%")
    print("平手:",drew/TEST*100,"%")
            #if ave_reward >= 990:  
            #   break  
  
if __name__ == '__main__':
    env = Board()
    env.after(10, main)
    env.mainloop()