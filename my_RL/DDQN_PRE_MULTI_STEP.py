from random import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from NNlayer import neuralNetLayer
from NNlayer import MLP
from PER_NN import tensorflowMLP
from kerasNN import neuralNet_keras
class myDQN:
    def __init__(self,
                 env,
                 epsilon = 0.8,
                 lamda = 0.01,
                 gama = 0.97,
                 step_size = 0.1,
                 alpha = 1,#线性回归的学习率
                 maxIteration = 50000,
                 learning_iter =1500,
                 batch = 64,
                 DDQN_step = 500):
        self.env = env
        self.step_size = step_size
        self.gama = gama
        self.lamda = lamda
        self.epsilon = epsilon
        self.alpha = alpha
        self.DDQN_step = DDQN_step
        self.maxIteration = maxIteration
        self.learning_iter = learning_iter
        self.state_n = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        print("staten:{} actionn:{}".format(self.state_n,self.action_n))
        #t = input()
        self.W = np.mat(np.zeros((self.state_n+self.action_n+1,1)))
        self.neuralnet = tensorflowMLP(self.state_n, 20, out=self.env.action_space.n)
        self.point = 0
        self.Q_list = []
        self.batch = batch
        self.IsRender = False
        self.test = 0
        self.multi_step = 0#when multy_step=0 it become DDQN_PER
    def getState(self, state):
        s = np.zeros((1,state.shape[0]))
        for i in range(state.shape[0]):
            s[0, i] = state[i]
        return s
    def oa2digit(self, state):
        ans = np.zeros((1,self.state_n))
        for i in range(self.state_n):
            ans[0,i] = state[i]
        return ans

    def performPolicy(self, state):
        random_value = random()
        if random_value < self.epsilon:
            action = self.env.action_space.sample()
            return action, None
        qValue, pro = self.getQvalue(state)
        action = np.argmax(qValue, axis=0)
        return action,np.max(qValue)

    def storage(self, s0, action, s1, reward,is_done, gama):
        self.neuralnet.storage(s0, action, s1, reward, is_done, gama)
    def getQvalue(self, state):
        Qvalue = self.neuralnet.current_predict(state)
        Q = Qvalue.flatten()
        return Q,None
    def getMAXQvalue(self, state):
        qValue, _ = self.getQvalue(state)
        value = np.max(qValue)
        action = np.argmax(qValue, axis=0)
        return value,action
    def learn(self):
        iteration = 1
        reward_list = 0.0
        reward_lists = []
        step = 0
        total_iter = 0
        while iteration < self.maxIteration:
            reward_ = 0
            self.epsilon -= 1.0/(1000)
            self.epsilon = max(self.epsilon, 0.05)
            s0 = self.oa2digit(self.env.reset())
            #if iteration<200:
                #self.multi_step = 10
            #else:
                #self.multi_step -= 0.1
                #self.multi_step = max(0,self.multi_step)
            #print("step:{}".format(self.multi_step))
            inner_iter = 0
            multi_step_list = []#storage s,r
            is_done = False
            while not is_done:
                if self.IsRender:
                    self.env.render()
                step +=1
                inner_iter += 1
                total_iter += 1
                a0,q_old = self.performPolicy(s0)
                s1, r1, is_done, info = self.env.step(a0)
                s1 = self.oa2digit(s1)
                reward_ += r1
                #if is_done:
                    #r1 = -40
                multi_step_list.append([s0, a0, s1, r1])
                r1 = 0
                s0 = multi_step_list[0][0]
                a0 = multi_step_list[0][1]
                for i in range(len(multi_step_list)-1, -1, -1):
                    r1 = self.gama*r1 + multi_step_list[i][3]
                if len(multi_step_list)>self.multi_step:
                    multi_step_list = multi_step_list[1:]
                self.storage(s0, a0, s1, r1, is_done, self.gama**(self.multi_step+1))
                if inner_iter > self.learning_iter:
                    is_done = True
                s0 = s1
                if total_iter % self.DDQN_step == 0:
                    total_iter = 0
                    self.neuralnet.copy()

            iteration += 1
            reward_list += reward_
            if iteration % 100 ==0:
                reward_list = reward_list * 1.0 / 100
                reward_lists.append(reward_list)
                reward_list = 0.0
            if iteration > 49990:# and iteration %10000 ==0 and self.test==0:
                t = input()
                self.IsRender = True
            print("第 {} 循环，奖励为{}".format(iteration,reward_))
            self.neuralnet.train()
        line_x = range(1, int(self.maxIteration/100)+1, 1)
        #print(reward_lists)
        #print(line_x)
        plt.plot(line_x, reward_lists)
        plt.show()




