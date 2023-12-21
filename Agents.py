
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from collections import deque
import torch.nn as nn
import random



class DeepQAgent3:
    def __init__(self,env,state_size=8,action_size=4,discount_factor=0.995,epsilon_greedy=1,epsilon_min=0.01,epsilon_decrement=5e-4,learning_rate=1e-3,max_memory_size=500):
        super(DeepQAgent3,self).__init__()
        self.env=env
        self.epsilon=epsilon_greedy
       
        self.learning_rate=learning_rate
        self.epsilon_decrement=epsilon_decrement
        self.state_size=state_size
        self.action_size=action_size
        self.gamma=discount_factor
        self.epsilon_min=epsilon_min
        self.build_model()
        self.actions=[i for i in range(action_size)]
        self.memory=deque(maxlen=max_memory_size)
        #self.memory=[]
    def build_model(self):
        self.model=nn.Sequential(nn.Linear(self.state_size,512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.Linear(256,self.action_size)
                                 
                                
                                )
        self.loss_fn=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),self.learning_rate)
    def remember(self,transition ):
        self.memory.append(transition)
    def choose_action(self,state):
        if np.random.rand()<=self.epsilon:

            return np.random.choice(self.actions)
        else:

            with torch.no_grad():
                q_values=self.model(torch.tensor(state,dtype=torch.float32))
                #print(q_values)
                return torch.argmax(q_values).item()
    def learn(self,batch_samples):
        s_list=[]
        action_list=[]
        reward_list=[]
        next_s_list=[]
        done_list=[]
        
        for t in batch_samples:
            s,action,reward,next_s,done=t
            if done:
                done=1
            else:
                done=0
            s_list.append(torch.tensor(s,dtype=torch.float32))
            action_list.append(action)
            reward_list.append(torch.tensor(reward,dtype=torch.float32))
            next_s_list.append(torch.tensor(next_s,dtype=torch.float32))
            done_list.append(torch.tensor(done))
        q_next=self.model(torch.stack(next_s_list))
        qmax,index=q_next.max(axis=1)
        q_target=torch.stack(reward_list)+self.gamma*qmax*(1-torch.stack(done_list))
        q_eval=self.model(torch.stack(s_list))
        target_all=q_eval.clone()
        for i,action in enumerate(action_list):
            target_all[i,action]=q_target[i]
        
        loss=self.loss_fn(q_eval,target_all)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.adjust_epsilon()
        return loss.item()
        
        
        
            
 
    def adjust_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon=self.epsilon-self.epsilon_decrement
        else:
            self.epsilon=self.epsilon_min
    def replay(self,batch_size):
        sample=random.sample(self.memory,batch_size)
        return self.learn(sample)




class DeepQAgent:
    def __init__(self,env,state_size=4,action_size=2,discount_factor=0.95,epsilon_greedy=1,epsilon_min=0.01,epsilon_decay=0.995,learning_rate=1e-3,max_memory_size=500):
        super(DeepQAgent,self).__init__()
        self.env=env
        self.epsilon=epsilon_greedy
        self.epsilon_decay=epsilon_decay
        self.learning_rate=learning_rate
        self.epsilon_decay=epsilon_decay
        self.state_size=state_size
        self.action_size=action_size
        self.gamma=discount_factor
        self.epsilon_min=epsilon_min
        self.build_model()
        self.actions=[0,1]
        self.memory=deque(maxlen=max_memory_size)
    def build_model(self):
        self.model=nn.Sequential(nn.Linear(self.state_size,256),
                                nn.ReLU(),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.ReLU(),
                                nn.Linear(64,self.action_size))
        self.loss_fn=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),self.learning_rate)
    def remember(self,transition ):
        self.memory.append(transition)
    def choose_action(self,state):
        if np.random.rand()<=self.epsilon:

            return np.random.choice(self.actions)
        else:

            with torch.no_grad():
                q_values=self.model(torch.tensor(state,dtype=torch.float32))
                #print(q_values)
                return torch.argmax(q_values).item()
    def learn(self,batch_samples):
        batch_states,batch_targets=[],[]
        for transition in batch_samples:
            s,a,r,next_s,done=transition
            with torch.no_grad():
                if done:
                    target=r
                else:
                    pred=self.model(torch.tensor(next_s,dtype=torch.float32))

                    target=r+self.gamma*pred.max()

                target_all=self.model(torch.tensor(s,dtype=torch.float32))
                target_all[a]=target
            batch_states.append(s)
            batch_targets.append(target_all)
            self.adjust_epsilon()# no se si va dentro del for o afuera
        self.optimizer.zero_grad()
        pred=self.model(torch.tensor(batch_states,dtype=torch.float32))
        loss=self.loss_fn(pred,torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def adjust_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    def replay(self,batch_size):
        sample=random.sample(self.memory,batch_size)
        return self.learn(sample)
