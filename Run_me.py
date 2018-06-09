from Board_env import Board
from DQN import DQN
import matplotlib.pyplot as plt
import numpy as np 

EPISODE = 10000 # Episode limitation  
TEST = 200
STEP = 9 # Step limitation in an episode
TESTCOUNT = 100

white_ls = []
black_ls = []
drew_ls = []

def main():  
    # initialize OpenAI Gym env and dqn agent  
    env = Board()
    
    agent = DQN(env)
  
    agent.copyWeightsToTarget()
    
    #play before training
    white = 0
    black = 0
    drew = 0
    for i in range(TEST):  
        state = env.reset()  
        color = "White"
        print('Game ',i)  
        for step in range(STEP):  
            #env.render()
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
                break
        if(not done):
            drew=drew+1 
    black_ls.append(black/TEST*100)
    white_ls.append(white/TEST*100)
    drew_ls.append(drew/TEST*100)
    
    for episode in range(EPISODE):  
        # initialize task  
        state = env.reset()
        print('Train ',episode)  
        color = "White"
        # Train  
        for step in range(STEP):  
            #自己下一步棋  
            if(color=="Black"):
                action = agent.egreedy_action(state) # e-greedy action for train  
            else:
                action = agent.random_action(state)
              
            if(episode <=10 and step == 1):
                action = 4
            #if env.env.is_valid_set_coord(action[0],action[1]):  
            next_state,reward,done,next_color,_ = env.step(action,color)
            color = next_color
            # Define reward for agent
            
            if done and next_color == "Black": 
                agent.modify_last_reward(-reward)
            if next_color == "White":
                agent.perceive(state,action,reward,next_state,done)  
            state = next_state * -1
            if done:  
                #print('done after ',step+1,"step(s)")  
                break  
        
        #play in training
        if(episode+1) % TESTCOUNT == 0:
            white = 0
            black = 0
            drew = 0
            for i in range(TEST):  
                state = env.reset()  
                color = "White"
                print('Game ',i)  
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
                        if(reward>0 and next_color=="Black"):
                            white=white+1
                        elif(reward>0 and next_color=="White"):
                            black=black+1
                        else:
                            drew=drew+1 
                        break
                if(not done):
                    drew=drew+1 
            black_ls.append(black/TEST*100)
            white_ls.append(white/TEST*100)
            drew_ls.append(drew/TEST*100)
    
    print("黑方勝率:",black_ls,"%")
    print("白方勝率:",white_ls,"%")
    print("平手:",drew_ls,"%")
    
    plt.title('Train with random action, white first')
    plt.xlabel('Training times(/100)')
    plt.ylabel('Percentage')
    plt.plot(black_ls,'ro-', label='Black')
    plt.plot(white_ls,'bx-', label='White')
    plt.plot(drew_ls,'g', label='drew')
    plt.legend()
    plt.show()   
            #if ave_reward >= 990:  
            #   break  
  
if __name__ == '__main__':
    main()
    #env.after(10, main)
    #env.mainloop()