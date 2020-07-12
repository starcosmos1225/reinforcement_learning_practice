from random import random
import numpy as np
import matplotlib.pyplot as plt
from NNlayer import tensorflowMLP
#from matrix_NN import tensorflowMLP





class myDQN:
    def __init__(self,
                 env,
                 epsilon = 0.2,
                 lamda = 0.01,
                 gama = 0.9,
                 step_size = 0.1,
                 alpha = 1,#线性回归的学习率
                 maxIteration = 4000,
                 learning_iter =500,
                 batch = 256):
        global sess
        self.env = env
        self.step_size = step_size
        self.gama = gama
        self.lamda = lamda
        self.epsilon = epsilon
        self.alpha = alpha
        self.maxIteration = maxIteration
        self.learning_iter = learning_iter
        self.state_n = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.W = np.mat(np.zeros((self.state_n+self.action_n+1, 1)))
        self.actor =tensorflowMLP(self.state_n, 20,learn_rate=1e-3,out=self.action_n, outputType="softmax")
        self.critic = tensorflowMLP(self.state_n, 20, out=1)


        self.point = 0
        self.Q_list = []
        self.batch = batch
        self.IsRender = False
        self.test = 0
    def getState(self,state):
        s = np.zeros((1,state.shape[0]))
        for i in range(state.shape[0]):
            s[0, i] = state[i]
        return s
    def oa2digit(self,state):
        ans = np.zeros((1,self.state_n))
        for i in range(self.state_n):
            ans[0,i] = state[i]
        return ans

    def performPolicy(self, state):
        pro = self.getQvalue(state)
        #for i in range(pro.shape[1]):
            #if pro[0,i]<0.05:
                #pro[0,i]=0.0
            #elif pro[0,i]>0.95:
                #pro[0,i]=1.0
        #pro = pro/np.sum(pro)
        #print(pro)
        #t=input()
        return np.random.choice(np.arange(pro.shape[1]), p=pro.ravel()),pro

    def sigmoid(self,inX):
        return 1.0/(1+np.exp(-inX))
    def softmax(self,inX):
        exps = np.exp(inX-np.max(inX))
        return exps / np.sum(exps)
    def Russian(self,pro):
        m = len(pro)
        per_epsilon = self.epsilon * 1.0 / m
        for i in range(m):
            pro[i] = pro[i] * (1-self.epsilon) + per_epsilon
        return
    def getQvalue(self, state):
        pro = self.actor.predict(state)
        return pro
    def getMAXQvalue(self, state):
        qValue, _ = self.getQvalue(state)
        value = np.max(qValue)
        return value

    def learn(self):
        iteration = 1
        reward_list = 0.0
        reward_lists = []
        step = 0
        while iteration < self.maxIteration:
            reward_ = 0
            #s0 = self.env.reset()
            s0 = self.oa2digit(self.env.reset())

            #print(type(s0))
            #print(s0)
            #t=input()
            inner_iter = 0
            is_done = False
            while not is_done:
                if self.IsRender:
                    self.env.render()
                step +=1
                inner_iter += 1
                a0, _ = self.performPolicy(s0)

                #print(a0)
                #print(pro)
                s1, r1, is_done, info = self.env.step(a0)
                reward_ += r1
                if is_done:
                    r1 = -40
                s1 = self.oa2digit(s1)
                td_error= self.critic.learn(s0, r1, s1, td=None)
                self.actor.learn(s0, a0, state_=None, td=td_error)
                if inner_iter > self.learning_iter:
                    is_done = True
                s0 = s1

            iteration += 1
            reward_list += reward_
            if iteration % 100 ==0:
                reward_list = reward_list * 1.0 / 100
                reward_lists.append(reward_list)
                reward_list = 0.0
            if iteration > 3990:# and iteration %10000 ==0 and self.test==0:
                t = input("show time:")
                self.IsRender = True
            print("第 {} 循环，奖励为{}".format(iteration,reward_))
            #if iteration>10:
            #self.stateNet.train()
            #self.Qnet.train()
            #self.Q_list.clear()
        line_x = range(1, int(self.maxIteration/100)+1, 1)
        #print(reward_lists)
        #print(line_x)
        plt.plot(line_x, reward_lists)
        plt.show()




