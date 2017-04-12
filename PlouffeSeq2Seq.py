import tensorflow as tf
import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from scipy.ndimage.interpolation import shift
import pdb
from IPython.display import HTML


def batch_up(inputs, batch_size,  max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """
    sequence_lengths = [len(seq) for seq in inputs]

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


class PlouffeGraph(object):
    '''
    Initializes a Plouffe Graph with number of nodes N and exponent multiplier k
    Example of ipython usage:
        import PlouffeSeq2Seq
        P = PlouffeSeq2Seq.PlouffeGraph(300,17)
        P.DrawGraph()
    '''
    def __init__(self,N, k):
        self.n_nodes = N
        self.k = k
        self.data = [(i, int(i*self.k) % self.n_nodes) for i in range(self.n_nodes)]
        '''
        We could also solve this by using class inheritance
        but this will do for now as a placeholder for our nx.Graph object - but is class inheritance evil?
        '''
        self.graph = nx.Graph(data = self.data)

    def DrawGraph(self):
        '''
        Simple utility function to draw the graph with matplotlib
        '''
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
    '''
    This object represents a sequence of Plouffe Graphs
    We can use this object to generate a video using the next_frame2draw method along with FuncAnimation from the
    matplotlib package
    We can also generate a sequence from a sequence of increasing values for k
    TODO generate a sequence from a sequence of increasing values for N
    We would like to perform time prediction on such sequences
    Given a PlouffeGraph at time t the corresponding output is a PlouffeGraph at time t+1
    Each PlouffeGraph is represented as a list of edges where each edge is given by 2d tuple
    Relevant dimensions:
    Input: N_nodes x 2
    Output: N_nodes x 2
    We would like to train an rnn at predicting a PlouffeGraph given the previous PlouffeGraph
    we would formulate the loss as a sum of cross entropy losses for each node in the graph this is similar to how
    charrnn is done. 
    '''
    def __init__(self, N, k, limit, n_frames):
        self.plouffe = PlouffeGraph(N, k)
        self.limit = limit
        self.n_nodes = N
        self.k = k
        self.cursor = self.k
        self.n_frames = n_frames
        self.step = (self.limit - self.cursor)/float(self.n_frames)
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

    def generate(self):
        '''
        Starting from the initial frame this method generates a sequence of n_frames between k and limit
        '''
        X = np.zeros([self.n_frames+1,self.n_nodes, 2])
        # After the loop state of self.cursor = limit
        for i in range(1,self.n_frames):
            X[i] = np.asarray(self.plouffe.data) # From list of tuples to a n_nodes x 2 array
            self.cursor += self.step
            # Update the graph before fetching the next graph
            self._update_graph()
        self.cursor = self.k
        return X

    def predict(self,limit):
        input_data = self.generate(k)
        output_data = np.roll(input_data,-1)
        prediction = model.predict(input_data)

    def next_frame2draw(self,step):
        self.fig.clf()
        self.cursor += step
        # update graph - remove existing edges and generate edges from the new cursor value
        self._update_graph()
        # generate new drawing elements for animation
        nodes = nx.draw_networkx_nodes(self.plouffe.graph,pos=self.pos,node_size=10,node_label=False)
        edges = nx.draw_networkx_edges(self.plouffe.graph,pos=self.pos)
        return nodes, edges


def animate(plouffe_seq):
    '''
    Animates the Plouffe sequence and saves if to Plouffe_n_nodes_k_limit_n_frames.mp4
    '''
    plt.cla()
    anim = FuncAnimation(plouffe_seq.fig, plouffe_seq.next_frame2draw,frames=plouffe_seq.n_frames, blit=True)
    return anim


class Plouffe_df(object):
    '''
    Generates dataframe with randomly chosen PlouffeGraphs
    '''
    def __init__(self, n_graphs):
        self.n_frames = 200
        self.n_nodes = np.random.randint(10,size=n_graphs)
        self.k = np.random.randint(100,size=n_graphs)
        self.limit = self.k + np.random.randint(400,size=n_graphs)
        self.df = self.build_df()

    def _set_plouffe_graph(self, row):
        N = row['Sets']
        k = row['Powers']
        return PlouffeGraph(N,k)

    def _set_plouffe_seq(self, row):
        N = row['Sets']
        k = row['Powers']
        limit = row['Limits']
        return PlouffeSequence(N, k, limit, self.n_frames)

    def build_df(self):
        df = pd.DataFrame({'Sets':self.n_nodes, 'Powers':self.k, 'Limits':self.limit})
        df['Plouffe'] = df.apply(self._set_plouffe_graph,axis=1)
        df['PlouffeSeq'] = df.apply(self._set_plouffe_seq,axis=1)
        return df


class Iterator(object):
    '''
    This iterator takes a dataframe of PlouffeSequences and transforms it into a list of such sequences
    It then creates batches of such sequences
    each sequence has number of steps self.n_frames with each element of the sequence having dimensions n_nodes x 2
    '''
    def __init__(self,n_graphs):
            self.size = n_graphs
            self.df = Plouffe_df(n_graphs)
            self.n_frames = self.df.n_frames
            self.seq = df.PlouffeSeq.tolist()
            self.cursor = 0
            self.shuffle()
            self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        random.shuffle(self.seq)
        self.cursor = 0

    def next_batch(self, batch_size):
        # if any of the buckets is full go to next epoch
        if self.cursor > self.size:
            self.epochs += 1
            self.shuffle()

        batch_seq = self.seq[self.cursor:self.cursor+batch_size]
        self.cursor += batch_size
        # Pad sequences with 0s so they are all the same length
        #### INPUT
        # Many dimensions can vary
        x = np.zeros([batch_size, self.n_frames, max_n_nodes], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[1:batch_len[i]+1] = batch_seq[i]
        #### OUTPUT - simply the input shifted by 1
        y = np.roll(x,-1)

        return x, y, batch_len


# Simple bucket sequence iterator
class Iterator(object):
    def __init__(self, num_nodes = 10, batch_size = 64, num_buckets = 5):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.size = int(len(self.sequences)/num_buckets)
        self.bucket_data = []
        # Put the shortest sequences in the first bucket etc
        # bucket_data is a list of 'buckets' where each bucket is a list of Sentences.
        self.num_buckets = num_buckets
        for bucket in range(self.num_buckets):
            self.bucket_data.append(self.sequences[bucket*self.size: (bucket+1)*self.size -1])
        self.cursor = np.array([0]*num_buckets)
        self.shuffle()
        self.epochs = 0


    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            random.shuffle(self.bucket_data[i])
            self.cursor[i] = 0


    def next_batch(self):
        # if any of the buckets is full go to next epoch
        if np.any(self.cursor+self.batch_size > self.size):
            self.epochs += 1
            self.shuffle() # Also resets cursor

        i = np.random.randint(0,self.num_buckets)
        all_seq = self.bucket_data[i][self.cursor[i]:self.cursor[i]+self.batch_size]

        input_seq = []
        target_seq = []
        for seq in all_seq:
            split_idx = np.random.choice(range(len(seq)))
            input_seq.append(seq[0:split_idx])
            target_seq.append(seq[split_idx:len(seq)])

        input_seq_time_major, input_seq_lengths = batch_up(input_seq, self.batch_size)

        target_seq_time_major, target_seq_lengths = batch_up(target_seq, self.batch_size)

        self.cursor[i] += self.batch_size

        return input_seq_time_major, input_seq_lengths, target_seq_time_major, target_seq_lengths
