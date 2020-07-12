import math
import numpy as np
class sumTree:
    def __init__(self,size=4096):
        self.size = size
        level = math.ceil(math.log2(size))
        self.capacity = 2**level
        self.nodeSize = self.capacity+size
        self.node = [1]*(self.nodeSize)
        self.node[0] = 0
        self.data = []
        self.point = 0
    def updateTree(self):
        index = int((self.nodeSize-1)/2)
        while index > 0:
            lchild = index*2
            rchild = index*2+1
            if rchild >= self.nodeSize:
                rchild = 0
            self.node[index] = self.node[lchild] + self.node[rchild]
            index -= 1
    def get_sum(self, index=1):
        return self.node[index]

    def length(self):
        return len(self.data)

    def get_priority_list(self):
        return self.node,self.capacity

    def get_leaf(self, v):
        parent_idx = 1
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= self.nodeSize:  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.node[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.node[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx-self.capacity
        return leaf_idx, self.node[leaf_idx], self.data[data_idx]
    def update_leaf(self,tree_idx,p):
        change = p - self.node[tree_idx]
        self.node[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 1:  # this method is faster than the recursive loop in the reference code
            tree_idx = tree_idx // 2
            self.node[tree_idx] += change
    def Isfull(self):
        return len(self.data)==self.size
    def insert(self, data,p):
        if len(self.data) >= self.size:
            self.point = self.point % self.size
            self.data[self.point] = data
            tree_idx = self.point + self.capacity
            self.update_leaf(tree_idx, p)
            self.point += 1
            return
        self.data.append(data)
        tree_idx = self.point + self.capacity
        self.update_leaf(tree_idx, p)
        self.point += 1
    def get_data(self):
        return self.data
    def updataBatch(self, priority, leaf_list):
        for i in range(len(leaf_list)):
            p = priority[i]
            index = leaf_list[i]
            self.updata_leaf(index, p)




