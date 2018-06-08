import tensorflow as tf  
import numpy as np  
import random  
from collections import deque  
  
GAMMA = 0.9 # discount factor for target Q  
INITIAL_EPSILON = 0.1 # starting value of epsilon  
FINAL_EPSILON = 0.01 # final value of epsilon  
REPLAY_SIZE = 10000 # 经验回放缓存大小  
BATCH_SIZE = 200 # 小批量尺寸  
TARGET_Q_STEP = 100 # 目标网络同步的训练次数
LEARNING_RATE = 0.01
  
class DQN():  
    # DQN Agent  
    def __init__(self, env):  
        # init experience replay  
        self.memory = deque()  
        # init some parameters  
        self.time_step = 0  
        self.epsilon = INITIAL_EPSILON  
        self.SIZE = env.SIZE  
        self.state_dim = self.SIZE*self.SIZE
        self.action_dim = self.SIZE*self.SIZE  
        self.hide_layer_inputs = 52  
        #创建Q网络  
        self.create_Q_network()  
        #创建训练方法  
        self.create_training_method()  
  
        self.target_q_step = TARGET_Q_STEP  
        self.create_TargetQ_network()  
  
  
        # 初始会话  
        self.session = tf.InteractiveSession()  
        self.session.run(tf.initialize_all_variables())  
  
    def create_Q_network(self):  
        # network weights  
        W1 = self.weight_variable([self.state_dim,self.hide_layer_inputs])  
        b1 = self.bias_variable([self.hide_layer_inputs])  
        W2 = self.weight_variable([self.hide_layer_inputs,self.action_dim])  
        b2 = self.bias_variable([self.action_dim])  
        # input layer  
        self.state_input = tf.placeholder("float",[None,self.state_dim])  
        # hidden layers  
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)  
        # Q Value layer  
        self.Q_value = tf.matmul(h_layer,W2) + b2  
        #保存权重  
        self.Q_Weihgts = [W1,b1,W2,b2]  
  
    def create_TargetQ_network(self):  
        # network weights  
        W1 = self.weight_variable([self.state_dim,self.hide_layer_inputs])  
        b1 = self.bias_variable([self.hide_layer_inputs])  
        W2 = self.weight_variable([self.hide_layer_inputs,self.action_dim])  
        b2 = self.bias_variable([self.action_dim])  
        # input layer  
        #self.state_input = tf.placeholder("float",[None,self.state_dim])  
        # hidden layers  
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)  
        # Q Value layer  
        self.TargetQ_value = tf.matmul(h_layer,W2) + b2  
        self.TargetQ_Weights = [W1,b1,W2,b2]  
  
    def copyWeightsToTarget(self):  
        for i in range(len(self.Q_Weihgts)):  
            self.session.run(tf.assign(self.TargetQ_Weights[i],self.Q_Weihgts[i]))  
  
    def create_training_method(self):  
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation  
        self.y_input = tf.placeholder("float",[None])  
      
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1) #mul->matmul  
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))  
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)  
  
    def perceive(self,state,action,reward,next_state,done):  
        one_hot_action = np.zeros(self.action_dim)  
        one_hot_action[action] = 1  
        self.memory.append([state,one_hot_action,reward,next_state,done])  
        if len(self.memory) > REPLAY_SIZE:  
            self.memory.popleft()  
  
        if len(self.memory) > BATCH_SIZE:  
            self.train_Q_network()  
  
    def modify_last_reward(self,new_reward):  
        v = self.memory.pop()
        v[2] = new_reward  
        self.memory.append(v) 
  
    def train_Q_network(self):  
        self.time_step += 1  
        # Step 1: obtain random minibatch from replay memory  
        minibatch = random.sample(self.memory,BATCH_SIZE)  
        state_batch = [data[0] for data in minibatch]  
        action_batch = [data[1] for data in minibatch]  
        reward_batch = [data[2] for data in minibatch]  
        next_state_batch = [data[3] for data in minibatch]  
  
        # Step 2: calculate y  
        y_batch = []  
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})  
        #Q_value_batch = self.TargetQ_value.eval(feed_dict={self.state_input:next_state_batch})  
        for i in range(0,BATCH_SIZE):  
            done = minibatch[i][4]  
            if done:  
                y_batch.append(reward_batch[i])  
            else :  
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))  
  
        self.optimizer.run(feed_dict={  
            self.y_input:y_batch,  
            self.action_input:action_batch,  
            self.state_input:state_batch  
            })  
  
        #同步目标网络  
        if self.time_step % self.target_q_step == 0:  
            self.copyWeightsToTarget()  
  
    #有機率嘗試新的走法
    def egreedy_action(self,state):  
        Q_value = self.Q_value.eval(feed_dict = {  
            self.state_input:[state]  
            })[0]  
        #print(Q_value)
        min_v = Q_value[np.argmin(Q_value)]-1  
        valid_action = []  
        for i in range(len(Q_value)):  
            if state[i]==0:  
                valid_action.append(i)  
            else:  
                Q_value[i] = min_v  
  
        if random.random() <= self.epsilon:  
            return valid_action[random.randint(0,len(valid_action) - 1)]  
            #return random.randint(0,self.action_dim - 1)  
        else:  
            return np.argmax(Q_value)  
  
        #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000  
        
    #不嘗試新的走法，直接最佳解
    def action(self,state):  
        Q_value = self.Q_value.eval(feed_dict = {  
            self.state_input:[state]  
            })[0]  
  
        min_v = Q_value[np.argmin(Q_value)]-1  
        valid_action = []  
        for i in range(len(Q_value)):  
            if state[i]==0:  
                valid_action.append(i)  
            else:  
                Q_value[i] = min_v  
  
        return np.argmax(Q_value)
    
    #隨機走
    def random_action(self,state):
        valid_action = []
        for i in range(self.action_dim):  
            if state[i]==0:  
                valid_action.append(i)  
   
        return valid_action[random.randint(0,len(valid_action) - 1)]  
            #return random.randint(0,self.action_dim - 1)  

  
    def weight_variable(self,shape):  
        initial = tf.truncated_normal(shape)  
        return tf.Variable(initial)  
  
    def bias_variable(self,shape):  
        initial = tf.constant(0.01, shape = shape)  
        return tf.Variable(initial)  