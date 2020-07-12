import numpy as np
import tensorflow as tf
import math
from sumTree import sumTree
from copy import copy
#首先要改网络，输出是atoms*action的softmax p
#然后predict的时候，计算的是support*p得到的action的期望值
#训练的时候，输入除了iw，s0，还有一个mi，a0，然后loss是milogP[a0]
#mi的计算是在tensorflowMLP中的train实现的
#因此需要修改的地方是网络结构和loss定义，还有train的方法
class tensorflowMLP:

    def __init__(self,n=4, n_layer=20,out=2,
                 learn_rate=1e-2,storageSize=1024,
                 batch=8,outputType=None,atoms = 11,Vmax=10,Vmin=-10):
        self.n= n
        self.n_layer = n_layer
        self.learn_rate = learn_rate
        self.storageSize = storageSize
        #self.layer1_W = np.ones((n+1, n_layer))
        #self.layer2_W = np.ones((n_layer, out))
        self.tree = sumTree(size=self.storageSize)
        self.tree.updateTree()
        self.PER_Beta = 0.4
        self.PER_E = 0.00001
        self.PER_A = 0.6
        self.out = [atoms, out]
        self.batch = batch
        self.outputType = outputType
        self.current_net = MLP(n, n_layer, self.out, learn_rate)
        self.value_net = copy(self.current_net)
        self.gama = 0.97
        self.support = np.zeros((1,atoms))
        self.distribution_step = (Vmax-Vmin)/(atoms-1)
        count = 0
        for i in range(atoms):
            #Vmin, Vmax+self.distribution_step, self.distribution_step):
            self.support[0, count] = Vmin + self.distribution_step *i
            count += 1
        self.atoms = atoms
        self.Vmax=Vmax
        self.Vmin = Vmin
        # 定义两个占位符

    def current_predict(self, state):
        #self.support = np.ones((1,11))
        #self.support[0, 10] = 0.0
        #print(self.support.shape)
        #print(self.current_net.predict(state)[0, :, :].shape)
        #print(np.dot(self.support, self.current_net.predict(state)[0, :, :])[0,:])
        #t = input()
        return np.dot(self.support, self.current_net.predict(state)[0, :, :])[0,:]

    def value_predict(self, state):
        return np.dot(self.support, self.value_net.predict(state)[0, :, :])[0,:]

    def storage(self, s0, action, s1, reward, is_done, gama):
        data = []
        mi_mat = np.zeros(self.atoms)
        a1 = np.argmax(self.current_predict(s1))
        #print("storage a1:{}".format(a1))
        pns = self.value_net.predict(s1)[0,:,:]
        #print(pns.shape)
        pns0 = self.current_net.predict(s0)[0,:, action]
        #print(pns0.shape)
        pns_a = pns[:, a1]
        #print(pns_a.shape)
        #t=input()
        for j in range(self.atoms):
            if is_done:
                Tz = min(max(self.Vmin, reward), self.Vmax)
            else:
                Tz = min(max(reward + gama * self.support[0, j], self.Vmin), self.Vmax)
            b = (Tz - self.Vmin) / self.distribution_step
            u = math.ceil(b)
            l = math.floor(b)
            if u == l:
                if u > 0:
                    l -= 1
            if u == l:
                if l < self.atoms - 1:
                    u += 1
            mi_mat[l] += pns_a[l] * (b - l)
            mi_mat[u] += pns_a[u] * (u - b)
        #print(np.log(pns0))
        #print(mi_mat)
        #t = input()
        loss = -np.dot(mi_mat, np.log(pns0).T)
        #print("loss:{}".format(loss))
        #t = input()
        priority = pow((abs(loss) + self.PER_E), self.PER_A)
        #v = q_old+td
        #output[0,action] = v
        data.append(s0)
        #data.append(output)
        data.append(action)
        data.append(s1)
        data.append(reward)
        data.append(is_done)
        data.append(gama)
        self.tree.insert(data, priority)
    def train(self):
        if not self.tree.Isfull():
            return
        batch = self.batch
        sum = self.tree.get_sum()
        input_mat = np.zeros((batch, self.n))
        mi_mat = np.zeros((batch, 1,self.atoms))
        iw_mat = np.zeros((batch, 1))
        #action_mat = np.zeros(batch)
        action_mat = []
        pri_seg = sum / batch
        node_list, leaf_index = self.tree.get_priority_list()
        min_list = node_list[leaf_index:]
        leaf_list = []
        data_list = []
        min_prob = np.min(min_list) / sum
        if min_prob == 0:
            min_prob = self.PER_E
        for i in range(batch):
            #print("batch {}".format(i))
            a, b = pri_seg * i, pri_seg * (i + 1)
            p_value = np.random.uniform(a, b)
            index, p, data = self.tree.get_leaf(p_value)
            leaf_list.append(index)
            data_list.append(data)
            pdf = p / sum
            tmp_mi_mat = np.zeros((1, 1, self.atoms))
            tmp_input_mat = np.zeros((1, self.n))
            tmp_input_mat[0, :] = data[0]
            input_mat[i, :] = data[0]
            action = data[1]
            #action_mat[i] = action
            action_mat.append(action)
            s1 = data[2]
            reward = data[3]
            is_done = data[4]
            gama = data[5]
            a1 = np.argmax(self.current_predict(s1))
            #print("action:{}".format(a1))
            pns = self.value_net.predict(s1)[0, :, :]
            #print(pns.shape)
            pns_a = pns[:, a1]
            #print(pns_a.shape)
            #t = input()
            for j in range(self.atoms):
                if is_done:
                    Tz = min(max(self.Vmin, reward), self.Vmax)
                else:
                    Tz = min(max(reward + gama * self.support[0, j], self.Vmin), self.Vmax)
                b = (Tz-self.Vmin)/self.distribution_step
                u = math.ceil(b)
                l = math.floor(b)
                #print(b-l)
                #print(u-b)
                #t=input()
                if u == l:
                    if u > 0:
                        l -= 1
                if u == l:
                    if l < self.atoms-1:
                        u += 1
                mi_mat[i, 0, l] += pns_a[l] * (b - l)
                mi_mat[i, 0, u] += pns_a[u] * (u - b)
                tmp_mi_mat[0, 0, l] += pns_a[l] * (b - l)
                tmp_mi_mat[0, 0, u] += pns_a[u] * (u - b)
            #output_mat[i, :] = output
            tmp_iw_mat = np.zeros((1, 1))
            tmp_iw_mat[0, 0] = pow(batch * pdf, -self.PER_Beta)
            iw_mat[i, 0] = pow(batch * pdf, -self.PER_Beta)
            loss = self.current_net.fit(tmp_input_mat, tmp_mi_mat, action, tmp_iw_mat)
            #print("loss:{}".format(loss))
            p = pow((abs(loss) + self.PER_E), self.PER_A)
            self.tree.update_leaf(index, p)
        #t = input("begin fit")
        #self.current_net.fit(input_mat, mi_mat, action_mat, iw_mat)
        #t = input("aft fit")
        self.PER_Beta += 0.001
        self.PER_Beta = min(1.0, self.PER_Beta)
        return
        for i in range(len(data_list)):
            data_ = data_list[i]
            s0 = data_[0]
            action = data_[1]
            s1 = data_[2]
            reward = data_[3]
            is_done = data_[4]
            gama = data_[5]
            a1 = np.argmax(self.current_predict(s1))
            mi_mat = np.zeros((1, self.atoms))
            pns = self.value_net.predict(s1)[0, :, :]
            #print(pns.shape)
            pns0 = self.current_net.predict(s0)[:, :, action]
            #print(pns0.shape)
            pns_a = pns[ :, a1]
            #print(pns_a.shape)
            #t=input()
            for j in range(self.atoms):
                if is_done:
                    Tz = min(max(self.Vmin, reward), self.Vmax)
                else:
                    Tz = min(max(reward + gama * self.support[0, j], self.Vmin), self.Vmax)
                b = (Tz - self.Vmin) / self.distribution_step
                u = math.ceil(b)
                l = math.floor(b)

                if u == l:
                    if u > 0:
                        l -= 1
                if u == l:
                    if l < self.atoms - 1:
                        u += 1
                mi_mat[0, l] += pns_a[l] * (b - l)
                mi_mat[0, u] += pns_a[u] * (u - b)
            loss = -np.dot(mi_mat, np.log(pns0).T)
            print("loss:{}".format(loss))
            #print(loss.shape)
            #t=input()
            p = pow((abs(loss) + self.PER_E), self.PER_A)
            index = leaf_list[i]
            self.tree.update_leaf(index, p)
    def copy(self):
        self.value_net = copy(self.current_net)

class MLP:
    def __init__(self, n=4, n_layer=20, out=2, learn_rate=1e-2):
        self.n= n
        self.n_layer = n_layer
        self.learn_rate = learn_rate
        self.out = out
        self.epoch = 2
        self.x = tf.placeholder(tf.float32, [None, n])  # 形状为n行1列，同x_data的shape
        #self.y = tf.placeholder(tf.float32, [None, out])
        self.iw = tf.placeholder(tf.float32, [None, 1])
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.abs_td = tf.placeholder(tf.float32, None, "abstd")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.m = tf.placeholder(tf.float32, [None, 1, out[0]], "distribute_m")

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
            units=self.out[0],  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        self.L3A = tf.layers.dense(
            inputs=l2,
            units=self.out[0]*self.out[1],  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )
        #self.L3V = tf.tile(self.L3V, [1, self.out[1]])
        self.L3V = tf.reshape(self.L3V, [-1, self.out[0], 1])
        self.L3A = tf.reshape(self.L3A, [-1, self.out[0], self.out[1]])
        #print(self.L3V.shape)
        #print(self.L3A.shape)
        #for i in range(self.out[1]):
            #self.L3V1 = tf.stack(self.L3V1, self.L3V)
        #self.L3A = tf.reshape(self.L3A, [self.out[1], self.out[0]])
        self.L3 = self.L3V + (self.L3A - tf.reduce_mean(self.L3A, axis=1, keep_dims=True))
        #print(self.L3.shape)
        #self.L3 = tf.reshape(self.L3, [self.out[0], self.out[1]])
        #print(self.L3.shape)
        #self.L3 = tf.transpose(self.L3)
        #print(self.L3.shape)
        self.L3 = tf.nn.softmax(self.L3, axis=1)
        #print(self.L3.shape)
        #print(self.m.shape)
        #print(self.iw.shape)
        #t = input()
        with tf.compat.v1.variable_scope('squared_TD_error'):
            #self.td_error = self.y - self.L3
            #self.abs_td = tf.abs(self.td_error)
            #t=input("input:t")
            #aa = self.a
            #tmp = tf.slice(self.L3,[0,0,self.a],[-1,-1,1])
            tmp = self.L3[:, :, self.a]
            #print(tmp.shape)
            self.loss = - self.iw * tf.matmul(self.m , tf.transpose(tf.math.log(tmp)))
            self.loss = self.loss[0,0,0]
            #print(self.loss.shape)
            #t=input()
            # self.loss = tf.square(self.y - self.L3)

        # 梯度下降最小化损失函数
        with tf.variable_scope('train_critic'):
            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

        # 全局变量初始化
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, inPut, mi, a, iw):
        for i in range(self.epoch):
            #t = input()
            #self.batch = len(a)

            #aa = tf.convert_to_tensor(a)
            _, loss = self.sess.run([self.train_step,self.loss], {self.x: inPut, self.m: mi, self.iw: iw, self.a: a})
            #print(loss)
            #t=input()
            #t = input()
        return loss

    def predict(self, state):
        return self.sess.run(self.L3, feed_dict={self.x: state})


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


