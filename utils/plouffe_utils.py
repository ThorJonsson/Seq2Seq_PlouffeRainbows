# Thor H. Jonsson
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage.interpolation import shift
import pdb

basis_len = 50

class SequenceIterator(object):
    def __init__(self,n_graphs):
            self.size = n_graphs
            self.df = make_plouffe_df(n_graphs)
            self.seq = df.PlouffeSeq.tolist()
            self.cursor = 0
            self.shuffle()
            self.epochs = 0
    
    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        random.shuffle(self.)
        self.cursor = 0

    def next_batch(self, n):
        # if any of the buckets is full go to next epoch
        if self.cursor > self.size:
            self.epochs += 1
            self.shuffle()

        batch_seq = self.seq[self.cursor:self.cursor+n]
        batch_len = [len(s) for s in batch_seq]
        self.cursor += n        
        maxlen = max(batch_len) + 1

        # Pad sequences with 0s so they are all the same length
        #### INPUT
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[1:batch_len[i]+1] = batch_seq[i]
        #### OUTPUT - simply the input shifted by 1
        y = np.roll(x,-1)
        
        return x, y, batch_len


def get_edge(row):
    N = row['Sets']
    k = row['Powers']
    input_seq = row['Input_Seq']
    output_list = []
    for zeta in input_seq:
        output_list.append((zeta*k)%N)
    return output_list    

def make_multiplication_df(N_Graphs = 100):
    N = np.random.randint(1900, size = N_Graphs)
    k = np.random.randint(300, size = N_Graphs)
    df = pd.DataFrame({'Sets':N, 'Powers':k})
    df['Input_Seq'] = df['Sets'].apply(range)
    df['Output_Seq'] = df.apply(get_edge,axis=1)
    return df

class PlouffeGraph(object):

    def __init__(self,N, k):
        self.N = N 
        self.k = k
        self.data = [(i, int(i*k)%N) for i in range(N)]
        ''' 
        We could also solve this by using class inheritance 
        but this will do for now as a placeholder for our nx.Graph object - but is class inheritance evil?
        '''
        self.graph = nx.Graph(data = self.data)
        
    def DrawGraph(self):
        plt.figure(figsize=(8,8))
        nx.draw_circular(self.graph,node_size=10, alpha=0.7,with_labels = False)
        plt.axis('equal')
        plt.show()

    '''ATH tsne!!!!!!!!'''
    def predict(self):
        input_data = (range(self.n_nodes),np.repeat(self.n_nodes,self.step), np.repeat(k,self.step))
        output_data = np.array(self.data)
        prediction = model.predict(input_data)

class PlouffeSequence(object):
    
    def __init__(self, N, k, limit, n_frames):
        self.plouffe = PlouffeGraph(N, k)
        self.n_nodes = N
        self.cursor = k
        self.step = (limit - k)/float(n_frames)
        self.pos = nx.circular_layout(self.plouffe.graph) # Set position of nodes in G
        self.fig = plt.figure(figsize=(8,8))
    
    def _update_graph(self):
        self.plouffe.graph.remove_edges_from(self.plouffe.data)
        self.plouffe.data = [(i, int(i*self.cursor)%self.n_nodes) for i in range(self.n_nodes)]
        self.plouffe.graph.add_edges_from(self.plouffe.data)
    
    def add2graph(list_of_tuples):
        self.plouffe.graph.remove_edges_from(self.plouffe.data)
        self.plouffe.data = list_of_tuples
        self.plouffe.graph.add_edges_from(self.plouffe.data)
        
    def _new_graph(self):
        self.plouffe = PlouffeGraph(self.n_nodes,self.cursor)
    
    def generate(self,cursor,n):
        if cursor < limit:
            self.step = (limit-k)float(n)
            self.cursor = cursor
        X = np.zeros([n+1,self.n_nodes, 2])
        # After the loop state of self.cursor = limit
        for i in range(1,n):
            X[i] = np.asarray(self.plouffe.data)
            self.cursor += self.step
            self._update_graph()
        self.cursor = 0
        return X

    def predict(self,limit):
        input_data = self.generate(k)
        output_data = np.roll(input_data,-1)
        prediction = model.predict(input_data)
        
    def next_frame2draw(self):
        self.fig.clf()
        self.cursor += self.step
        # update graph - remove existing edges and generate edges from the new cursor value
        self._update_graph()
        # generate new drawing elements for animation
        nodes = nx.draw_networkx_nodes(self.plouffe.graph,pos=self.pos,node_size=10,node_label=False)
        edges = nx.draw_networkx_edges(self.plouffe.graph,pos=self.pos)
        return nodes, edges


def set_plouffe_graph(row):
    N = row['Sets']
    k = row['Powers']
    return PlouffeGraph(N,k)

def set_plouffe_seq(row):
    N = row['Sets']
    k = row['Powers']
    return PlouffeSequence(N,k)

def make_plouffe_df(n_graphs = 100):
    N = np.random.randint(10,size=n_graphs)
    k = np.random.randint(100,size=n_graphs)
    df = pd.DataFrame({'Sets':N, 'Powers':k})
    df['Plouffe'] = df.apply(set_plouffe_graph,axis=1)
    df['PlouffeSeq'] = df.apply(set_plouffe_seq,axis=1)
    return df

class SequenceIterator(object):
    def __init__(self,n_graphs):
            self.size = n_graphs
            self.df = make_plouffe_df(n_graphs)
            self.seq = df.PlouffeSeq.tolist()
            self.cursor = 0
            self.shuffle()
            self.epochs = 0
    
    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        random.shuffle(self.)
        self.cursor = 0

    def next_batch(self, n):
        # if any of the buckets is full go to next epoch
        if self.cursor > self.size:
            self.epochs += 1
            self.shuffle()

        batch_seq = self.seq[self.cursor:self.cursor+n]
        batch_len = [len(s) for s in batch_seq]
        self.cursor += n        
        maxlen = max(batch_len) + 1

        # Pad sequences with 0s so they are all the same length
        #### INPUT
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[1:batch_len[i]+1] = batch_seq[i]
        #### OUTPUT - simply the input shifted by 1
        y = np.roll(x,-1)
        
        return x, y, batch_len


def get_edge(row):
    N = row['Sets']
    k = row['Powers']
    input_seq = row['Input_Seq']
    output_list = []
    for zeta in input_seq:
        output_list.append((zeta*k)%N)
    return output_list    

def make_multiplication_df(N_Graphs = 100):
    N = np.random.randint(1900, size = N_Graphs)
    k = np.random.randint(300, size = N_Graphs)
    df = pd.DataFrame({'Sets':N, 'Powers':k})
    df['Input_Seq'] = df['Sets'].apply(range)
    df['Output_Seq'] = df.apply(get_edge,axis=1)
    return df

