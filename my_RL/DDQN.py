from random import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from NNlayer import neuralNetLayer
from NNlayer import MLP
#from Model_NN import tensorflowMLP
from Dueling_NN import tensorflowMLP
from kerasNN import neuralNet_keras
class myDQN:
    def __init__(self,
                 env,
                 epsilon = 1.0,
                 lamda = 0.01,
                 gama = 0.97,
                 step_size = 0.1,
                 alpha = 1,#线性回归的学习率
                 maxIteration = 4000,
                 learning_iter =5000,
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
        self.W = np.mat(np.zeros((self.state_n+self.action_n+1,1)))
        self.neuralnet = tensorflowMLP(self.state_n, 20, out=self.env.action_space.n)
        self.valueNet = copy(self.neuralnet)
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
        random_value = random()
        if random_value < self.epsilon:
            action = self.env.action_space.sample()
            return action, None
        qValue, pro = self.getQvalue(state)
        action = np.argmax(qValue, axis=0)
        return action,np.max(qValue)

    def storage(self,state,action,value):
        inPut = state
        outPut = self.neuralnet.predict(state)
        #print("s:{},a:{},v:{}".format(state,action,value))
        #print(outPut)
        #t=input()
        outPut[0, action] = value
        #if len(self.Q_list)<5000:
            #self.Q_list.append((state,action,value))
            #self.point+=1
        #else:
            #self.point = self.point % 5000
            #self.Q_list[self.point] = (state,action,value)
            #self.point +=1
        self.neuralnet.storage(inPut,outPut)

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
        Qvalue = self.neuralnet.predict(state)
        Q = Qvalue.flatten()
        return Q,None
    def getMAXQvalue(self, state):
        qValue, _ = self.getQvalue(state)
        value = np.max(qValue)
        action = np.argmax(qValue, axis=0)
        return value,action

    def MCSearch(self,state, depth):
        #print("MC")
        if self.IsRender:
            self.env.render()
        A = self.performPolicy(state)
        s1, r1, done, info = self.env.step(A)
        s1 = self.oa2digit(s1)
        if done:
            return r1 - 40, A, r1
        if depth > 500:
            return r1, A, r1
        Qvalue,action,r = self.MCSearch(s1, depth + 1)
        oldQ, oldP = self.getQvalue(s1)
        #print(oldQ)
        #print(oldP)
        #t = input()
        self.Russian(oldP)
        #print(oldP)
        #t = input()
        self.storage(s1, action, Qvalue)
        E = 0.0
        for i in range(self.env.action_space.n):
            if i != action:
                E += oldP[i] * oldQ[i]
            else:
                E += Qvalue * oldP[i]
        return E + r1, A, r + r1

    def fitting(self):
        mat_state = np.zeros((len(self.Q_list), self.W.shape[0]))
        mat_q = np.zeros((len(self.Q_list),1))
        for i in range(len(self.Q_list)):
            s = self.Q_list[i][0].copy()
            a = np.mat(np.zeros((1,self.action_n)))
            a[0, self.Q_list[i][1]] = 1
            s = np.c_[s, a]
            b = np.mat(np.zeros((1, 1)))
            b[0,0] = 1
            s = np.c_[s, b]
            mat_state[i, :] = s
            mat_q[i, 0] = self.Q_list[i][2]
        for i in range(self.learning_iter):
            error = mat_q - self.sigmoid(mat_state*self.W)
            self.W = self.W + self.alpha*mat_state.transpose()*error

    def sample(self):
        if (len(self.Q_list)<self.batch):
            return
        sample_index = np.random.choice(len(self.Q_list), size=self.batch)
        q_l = []
        for a in sample_index:
            q_l.append(self.Q_list[a])
        self.Q_list = q_l.copy()

    def non_linear_fitting(self):
        qv = []
        qr = []
        for i in range(len(self.Q_list)):
            for j in range(10):
                s = self.Q_list[i][0].copy()
                a = np.mat(np.zeros((self.action_n, 1)))
                a[self.Q_list[i][1],0] = 1
                s = np.r_[s, a]
                qv.append(self.neuralnet.compute(s))
                self.neuralnet.backpropagation(self.Q_list[i][2])
                qr.append(self.Q_list[i][2])

        return
    def train(self):
        if len(self.Q_list)<self.batch:
            return
        batch = min(len(self.Q_list),self.batch)
        if batch <= 0 :
            return
        a_ = range(len(self.Q_list))
        a1 = np.random.choice(a_, size=batch, replace=False, p=None)
        state_mat = np.zeros((batch, self.state_n))
        value_mat = np.zeros((batch, self.action_n))
        count = 0
        for i in a1:
            state_mat[count, :] = self.Q_list[i][0]
            value_mat[count, :] = self.neuralnet.predict(self.Q_list[i][0])
            value_mat[count, self.Q_list[i][1]] = self.Q_list[i][2]
            count += 1
        self.neuralnet.fit(state_mat,value_mat)

    def learn(self):
        iteration = 1
        reward_list = 0.0
        reward_lists = []
        step = 0
        total_iter = 0
        while iteration < self.maxIteration:
            reward_ = 0
            #self.epsilon -= 1.0/(iteration+1)
            #self.epsilon = max(self.epsilon, 0.05)
            self.epsilon = 0.01
            s0 = self.oa2digit(self.env.reset())
            inner_iter = 0
            is_done = False
            while not is_done:
                if self.IsRender:
                    self.env.render()
                step +=1
                inner_iter += 1
                total_iter += 1
                a0,q_old = self.performPolicy(s0)
                s1, r1, is_done, info = self.env.step(a0)
                #print(r1)
                reward_ += r1
                #if is_done:
                    #r1 = -40
                s1 = self.oa2digit(s1)

                #reward_ += r1
                if is_done:
                    q = r1
                else:
                    if q_old is None:
                        q_old, _ = self.getMAXQvalue(s0)
                    _, action = self.getMAXQvalue(s1)
                    vv = self.valueNet.predict(s1)
                    td_error = r1 +self.gama*vv[0, action]-q_old
                    #td_error = r1 + self.gama * vv - q_old
                    #q = r1 + self.gama * vv
                    q = q_old + self.alpha*td_error

                #elif inner_iter > 200:
                    #q = 500 + r1
                    #is_done = True
                if inner_iter > self.learning_iter:
                    # q = 500 + r1
                    is_done = True
                #if inner_iter > self.learning_iter:
                    #is_done = True
                #inPut = s0
                #outPut = self.neuralnet.predict(s0)
                #outPut[0, a0] = q
                #self.neuralnet.fit(inPut,outPut)
                self.storage(s0, a0, q)
                s0 = s1
                #self.neuralnet.train()
                if total_iter % self.DDQN_step==0:
                    total_iter =0
                    self.valueNet = copy(self.neuralnet)

            iteration += 1
            reward_list += reward_
            if iteration % 100 ==0:
                reward_list = reward_list * 1.0 / 100
                reward_lists.append(reward_list)
                reward_list = 0.0
            if iteration > 3990:# and iteration %10000 ==0 and self.test==0:
                t = input()
                self.IsRender = True
            print("第 {} 循环，奖励为{}".format(iteration,reward_))
            self.neuralnet.train()
            #self.valueNet = copy(self.neuralnet)
        line_x = range(1, int(self.maxIteration/100)+1, 1)
        #print(reward_lists)
        #print(line_x)
        plt.plot(line_x, reward_lists)
        plt.show()




