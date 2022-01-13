# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:56:33 2020

@author: Minhyeong
"""

class Prioritized_Experience_Replay:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.priorities = np.zeros(self.capacity)
        
        if self.isPowerOfTwo(self.capacity):
            self.tree = np.zeros(2 * self.capacity - 1)
            self.tree_type = True
            
        else:
            self.tree = np.zeros(2 ** (math.ceil(math.log2(self.capacity)) + 1) - 1)
            self.tree_type = False
            
        self.buffer_idx = 0
        self.n_entries = 0
        
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        
    def isPowerOfTwo(self, x):
        return (x and (not(x & (x - 1))))
    
    def add(self, z, x, priority):
        transition = [z, x]
        self.buffer[self.buffer_idx] = transition
        self.priorities[self.buffer_idx] = priority
        
        if self.tree_type:
            self.tree_idx = self.buffer_idx + self.capacity - 1
            
        else:
            self.tree_idx = self.buffer_idx + (2 ** math.ceil(math.log2(self.capacity))) - 1
        
        self.update_priority(self.tree_idx, priority)
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
        
    def update_priority(self, index, new_priority):
        change = new_priority - self.tree[index]
        self.tree[index] = new_priority
        self._propagate(index, change)
        
    def _propagate(self, index, change):
        parent = (index - 1) // 2
        
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)
        
    def priority_sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        
        segment = self.tree[0] / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        sampling_probabilities = np.array([x / self.tree[0] for x in priorities])
            
        is_weight = np.power(self.n_entries * sampling_probabilities, -self.beta)
        max_weight = self.cal_max_weight()
        
        is_weight /= max_weight
        
        return batch, idxs, is_weight
        
    def get(self, s):
        idx = self._retrieve(0, s)
        
        if self.tree_type:
            dataIdx = idx - self.capacity + 1
        else:
            dataIdx = idx - (2 ** math.ceil(math.log2(self.capacity))) + 1
        
        return idx, self.tree[idx], self.buffer[dataIdx]
        
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def cal_max_weight(self):
        max_probability = self.priorities.min() / self.tree[0]
        max_weight = np.power(self.n_entries * max_probability, -self.beta)
        
        return max_weight