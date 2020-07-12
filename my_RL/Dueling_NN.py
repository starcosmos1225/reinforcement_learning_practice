import numpy as np
import tensorflow as tf
import math
class tensorflowMLP:

    def __init__(self,n=4, n_layer=20,out=2,
                 learn_rate=1e-2,storageSize=5000,
                 batch=256,outputType=None):
        self.n= n
        self.n_layer = n_layer
        self.learn_rate = learn_rate
        self.storageSize = storageSize
        #self.layer1_W = np.ones((n+1, n_layer))
        #self.layer2_W = np.ones((n_layer, out))
        self.epoch = 1
        self.out = out
        self.Q_list=[]
        self.batch = batch
        self.point = 0
        self.outputType = outputType
        # 定义两个占位符
        self.x = tf.placeholder(tf.float32, [None, n])  # 形状为n行1列，同x_data的shape
        self.y = tf.placeholder(tf.float32, [None, out])
        #if self.outputType == "softmax":
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        # 定义神经网络

        l1 = tf.layers.dense(
            inputs=self.x,
            units=self.n_layer,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        l2 = tf.layers.dense(
            inputs=l1,
            units=self.n_layer,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        self.L3V = tf.layers.dense(
            inputs=l2,
            units=1,  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        self.L3A = tf.layers.dense(
            inputs=l2,
            units=self.out,  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        self.L3 = self.L3V + (self.L3A-tf.reduce_mean(self.L3A,axis=1,keep_dims=True))
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.y - self.L3
            self.loss = tf.square(self.td_error)
            #self.loss = tf.square(self.y - self.L3)

    # 梯度下降最小化损失函数
        with tf.variable_scope('train_critic'):
            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

    # 全局变量初始化
        self.sess =tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, input, label):
        for i in range(self.epoch):
            self.sess.run(self.train_step,{self.x: input, self.y:label})
        return

    def learn(self, state, r, state_, td=None):
        if td is not None:
            feed_dict = {self.x: state, self.a: r, self.td_error: td}
            _, exp_v = self.sess.run([self.train_step, self.exp_v], feed_dict)
        else:
            v_ = self.sess.run(self.L3, {self.x: state_})
            td_error, _ = self.sess.run([self.td_error, self.train_step],
                                                {self.x: state, self.v_: v_, self.r: r})
            return td_error
    def predict(self,state):
        return self.sess.run(self.L3, feed_dict={self.x: state})
    def display_net(self):
        print("L1 W:")
        print(self.sess.run(self.L1_weights))
        #print(self.L1_weights[1])
        print("L1 B:")
        print(self.sess.run(self.L1_bias))
        print("L2 W:")
        print(self.sess.run(self.L2_weights))
        print("L2 B:")
        print(self.sess.run(self.L2_bias))
        print("L3 W:")
        print(self.sess.run(self.L3_weights))
        print("L3 B:")
        print(self.sess.run(self.L3_bias))
    def storage(self, input, output):
        if len(self.Q_list) < self.storageSize:
            self.Q_list.append((input, output))
            self.point += 1
        else:
            self.point = self.point % self.storageSize
            self.Q_list[self.point] = (input, output)
            self.point +=1
    def storage_priority(self, inPut, output, r,action,qold):
        if len(self.Q_list) < self.storageSize:
            #从大到小排
            self.Q_list.append((inPut, output, r, action))
            index = 0
            #print(self.Q_list[index][2])
            while index < self.point and math.fabs(self.Q_list[index][2]) > math.fabs(r):
                index += 1
            self.point = len(self.Q_list)
            #self.Q_list[index] = self.Q_list[self.point]
            for i in range(self.point-1, index, -1):
                self.Q_list[i] = self.Q_list[i-1]
            self.Q_list[index] = (inPut, output, r, action,qold)
            #print(self.Q_list)
            #t = input()
        else:
            index = 0
            while index < self.point and self.Q_list[index][2] > r:
                index += 1
            if index==self.point:
                return
            for i in range(self.point-1, index, -1):
                self.Q_list[i] = self.Q_list[i-1]
            self.Q_list[index] = (inPut, output, r, action, qold)
    def train(self):
        if len(self.Q_list)<self.batch:
            return
        batch = self.batch
        a_ = range(len(self.Q_list))
        a1 = np.random.choice(a_, size=batch, replace=False, p=None)
        input_mat = np.zeros((batch, self.n))
        output_mat = np.zeros((batch, self.out))
        count = 0
        for i in a1:
            input_mat[count, :] = self.Q_list[i][0]
            output_mat[count, :] = self.Q_list[i][1]
            count += 1
        self.fit(input_mat, output_mat)
    def train_priority(self):
        if len(self.Q_list)<self.batch:
            return
        batch = self.batch
        #a_ = range(len(self.Q_list))
        #a1 = np.random.choice(a_, size=batch, replace=False, p=None)
        a1 = self.Q_list[0:batch].copy()
        self.Q_list = self.Q_list[batch:].copy()
        input_mat = np.zeros((batch, self.n))
        output_mat = np.zeros((batch, self.out))
        count = 0
        for i in a1:
            input_mat[count, :] = i[0]
            output_mat[count, :] = i[1]
            count += 1
        self.fit(input_mat, output_mat)
        for i in a1:
            s1 = self.predict(i[0])
            qold = i[4]
            qnew = np.max(s1)
            rnew = i[2]-(qnew-qold)
            self.storage_priority(i[0], s1, rnew, i[3],qnew)

class MLP:
    def __init__(self, n=4, n_layer=10,out=2, learn_rate=1e-2):
        self.n= n
        self.n_layer = n_layer
        self.learn_rate = learn_rate
        self.layer1_W = np.ones((n+1, n_layer))
        self.layer2_W = np.ones((n_layer, out))
        self.epoch = 1000
        self.out = out

    def fit(self, state,value):
        one_mat = np.mat(np.ones((state.shape[0], 1)))
        new_mat = np.c_[state,one_mat]
        for i in range(self.epoch):
            value_ = self.forward(new_mat)
            #print(value_)
            #t=input()
            delta = value-value_
            self.backward(delta)
            print(value)
            #t=input()
            print(self.forward(new_mat))
        t = input()

    def predict(self, state):
        one_mat = np.mat(np.ones((state.shape[0], 1)))
        new_mat = np.c_[state, one_mat]
        #print(new_mat.shape)
        #t=input()
        return self.forward(new_mat)

    def forward(self, state_mat):
        self.input = state_mat
        #print(self.layer1_W.shape)
        #t=input()
        self.layer1_mat_in = state_mat*self.layer1_W
        self.layer1_mat = self.relu(self.layer1_mat_in)
        return self.layer1_mat * self.layer2_W

    def sigmoid(self, inX, back=False):
        if not back:
            return 1.0/(1+np.exp(-inX))

    def relu(self, data_, back=False):
        if not back:
            return np.maximum(0, data_)
        ans = np.zeros(data_.shape)
        for i in range(data_.shape[0]):
            for j in range(data_.shape[1]):
                if self.layer1_mat_in[i, j] > 0:
                    ans[i, j] = data_[i, j]
        return ans

    def backward(self, delta):
        #ones_mat = np.ones((delta.shape[0], 1))
        #print(delta.shape)
        #print(y.shape)
        #print(type(delta))
        #print(type(y))
        #test = np.multiply(delta,y)
        #t = input()
        #delta_2_in = np.multiply(np.multiply(delta,y),(ones_mat-y))
        delta_2_in = delta
        #print(delta)
        delta_layer2_W = self.layer1_mat.T*delta_2_in
        delta_layer1_out = delta_2_in*self.layer2_W.T
        delta_1_in = self.relu(delta_layer1_out, back=True)
        delta_layer1_W = self.input.T * delta_1_in
        self.layer1_W += self.learn_rate*delta_layer1_W
        self.layer2_W += self.learn_rate*delta_layer2_W
        #print(self.layer1_W)
        #print(self.layer2_W)

        #t=input()

class neuralNetLayer:
    def __init__(self, n=6, number_w=2, learn_rate=1e-4, Type="Relu"):
        self.next = None
        self.n = n
        self.nw = number_w
        self.data = np.mat(np.zeros((n, 1)))
        self.result = np.mat(np.zeros((n+1-number_w, 1)))
        self.W = np.mat(np.ones((n+1-number_w,number_w+1)))
        self.previous = None
        self.Type = Type
        self.learn_rate = learn_rate

    def createLayer(self):
        if self.n > 1 and self.nw < self.n:
            nextLayer = neuralNetLayer(self.n-1, self.nw, Type=self.Type)
            self.next = nextLayer
            nextLayer.previous = self
            nextLayer.createLayer()
        else:
            self.next = None

    def compute(self, data_):
        #print("compute begin")
        self.data = data_.copy()
        #print(self.data)
        self.result = np.mat(np.zeros((self.n-1,1)))
        a = np.mat(np.ones((1, 1)))
        for i in range(self.n+1-self.nw):
            dataPair = np.r_[self.data[i:i+self.nw, 0], a]
            self.result[i,0] = self.W[i]*dataPair
        #print(self.W)
        #print(self.result)
        if self.Type == "Relu" and self.next is not None:
            self.result = self.Relu(self.result)
        #print(self.data)
        #print("result")
        #print(self.result)
        #print("compute end")
        if self.next == None:
            return self.result[0,0]
        else:
            return self.next.compute(self.result)

    def backpropagation(self, result_):
        lastlayer = self.findButtom()
        data_ = np.mat(np.zeros((1,1)))
        data_[0,0] = result_
        delta = data_ -lastlayer.result
        while lastlayer is not None:
            delta = lastlayer.fitting(delta)
            lastlayer = lastlayer.previous

    def findButtom(self):
        layer = self
        while layer.next is not None:
            layer = layer.next
        return layer

    def fitting(self, delta):
        #print("fitting in nn")
        #print(data_)
        #print(self.n)
        #print(self.result)
        #delta = data_ - self.result
        b = np.mat(np.ones((1, 1)),dtype = 'float64')
        deltaW = np.mat(np.zeros(self.W.shape),dtype = 'float64')
        delta_data = np.mat(np.zeros(self.data.shape))
        #print(delta)
        if self.Type =="Relu" and self.next is not None:
            delta = self.fittingRelu(delta)
        for i in range(self.n+1-self.nw):
            tmpmat = np.r_[self.data[i:i+self.nw], b]
            tmpmat = tmpmat.transpose()
            #print(tmpmat)
            #print(delta)
            deltaW[i] =self.learn_rate*delta[i,0] *tmpmat
            #print(deltaW[i])
        for i in range(self.n):
            for j in range(self.nw):
                index = i+1+j-self.nw
                if (index<0 or index > self.n-self.nw):
                    continue
                #index = min(max(index, 0), self.n-self.nw)
                delta_data[i] += delta[index]*self.W[index, self.nw-1-j]
        #print(deltaW)
        self.W += deltaW
        #print(self.W)
        #print("end fitting in nn")
        return delta_data

    def display(self):
        print(self.W)
        if self.next is not None:
            self.next.display()

    def Relu(self, data_):
        return np.maximum(0, data_)

    def fittingRelu(self, data_):
        ans = self.result.copy()
        #print(ans.shape)
        #print(type(ans))
        #print("ans")
        #print(ans[0,0])
        #print(ans[1, 0])
        for i in range(ans.shape[0]):
            #print(ans[i,0])
            if ans[i, 0] > 0.0:
                ans[i, 0] = data_[i, 0]
            else:
                ans[i, 0] = 0.0

        return ans


