# Arnbjorg Soffia <3 Thor Jonsson
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''
set parameters for the groups given by the alias
=====   =================
Alias   Property
=====   =================
'lw'    'linewidth'
'ls'    'linestyle'
'c'     'color'
'fc'    'facecolor'
'ec'    'edgecolor'
'mew'   'markeredgewidth'
'aa'    'antialiased'
=====   =================
'''
from matplotlib import rc
import numpy as np
import networkx as nx
import pandas as pd
import random
import pdb

class PlouffeGraph(object):

    def __init__(self,N, k):
        self.N = N
        self.k = k
        # This list represents the Plouffe Graph for exponent k
        self.data = [(i, int(i*k%N)) for i in range(N)]
        '''
        We could also solve this by using class inheritance
        but this will do for now as a placeholder for our nx.Graph object - but is class inheritance evil?
        '''
        self.graph = nx.Graph(data = self.data)

    def draw(self):
        plt.figure(figsize=(8,8))
        nx.draw_circular(self.graph,node_size=10, alpha=0.7,with_labels = False)
        plt.axis('equal')
        plt.show()
        plt.clf()


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
        self.plouffe.data = [(i, int(i*self.cursor%self.n_nodes)) for i in range(self.n_nodes)]
        self.plouffe.graph.add_edges_from(self.plouffe.data)

    def add2graph(list_of_tuples):
        self.plouffe.graph.remove_edges_from(self.plouffe.data)
        self.plouffe.data = list_of_tuples
        self.plouffe.graph.add_edges_from(self.plouffe.data)

    def _new_graph(self):
        self.plouffe = PlouffeGraph(self.n_nodes,self.cursor)

    def next_frame(self,step):
        self.fig.clf()
        self.cursor += self.step
        # update graph - remove existing edges and generate edges from the new cursor value
        self._update_graph()
        # generate new drawing elements for animation
        nodes = nx.draw_networkx_nodes(self.plouffe.graph,pos=self.pos,node_size=10,node_label=False)
        edges = nx.draw_networkx_edges(self.plouffe.graph,pos=self.pos)
        return nodes, edges


class ReconPlouffeViewer(object):

    def __init__(self, PlouffeReconstruction, seq_length, num_nodes):
        '''
        This class is used to generate animations of reconstructions of the plouffe sequence
        Args:
          PlouffeReconstruction: A seq_length x num_nodes numpy array
        '''
        self.data = np.reshape(PlouffeReconstruction, [seq_length, num_nodes]).tolist()
        self.seq_length = seq_length
        self.num_nodes = num_nodes
        self.cursor = 0
        self._init_graph()

    def _init_graph(self):
        self.curr_graph = [(i, int(self.data[self.cursor][i]%self.num_nodes)) for i in range(self.num_nodes-1)]
        self.graph = nx.Graph(data = self.curr_graph)
        self.pos = nx.circular_layout(self.graph) # Set position of nodes in G
        self.fig = plt.figure(figsize=(8,8))

    def _update_graph(self):
        self.graph.remove_edges_from(self.curr_graph)
        self.curr_graph = [(i, int(self.data[self.cursor][i]%self.num_nodes)) for i in range(self.num_nodes-1)]
        self.graph.add_edges_from(self.curr_graph)

    def next_frame(self,step):
        self.fig.clf()
        self.cursor += 1
        # update graph - remove existing edges and generate edges from the new cursor value
        self._update_graph()
        # generate new drawing elements for animation
        nodes = nx.draw_networkx_nodes(self.graph,pos=self.pos,node_size=10,node_label=False)
        edges = nx.draw_networkx_edges(self.graph,pos=self.pos)
        return nodes, edges


# TODO n_frames make consistent
def get_plouffe_seq(init, num_nodes, num_frames):
    limit = init+10
    # Really simple if you think about what the Plouffe Sequence actually is (and you like list comprehension in list
    # comprehension
    Plouffe_Seq = [[(i*j)%num_nodes for i in range(num_nodes)] for j in np.arange(init,limit, (limit-init)/float(num_frames))]
    return Plouffe_Seq


def make_dataset(_size, num_nodes, num_frames, max_int = 5000):
    x = np.random.randint(max_int, size=_size)
    df = pd.DataFrame(x)
    df['Plouffe'] = df[0].apply(get_plouffe_seq, args=(num_nodes, num_frames))
    return df


def batch_up(inputs, batch_size, num_nodes, num_frames=200):
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
    inputs_batch_major = np.zeros(shape=[batch_size, num_frames, num_nodes], dtype=np.float32) # == PAD

    for i, seq in enumerate(inputs):
        inputs_batch_major[i] = np.array(seq)*(1/float(num_nodes))

    return inputs_batch_major


# Simple sequence iterator
class Iterator(object):

    def __init__(self, Plouffe_Sequences, num_nodes, num_frames, batch_size):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_frames = num_frames
        self.data = Plouffe_Sequences
        self.size = len(self.data)
        # cursor within an epoch
        self.cursor = 0
        self.epoch = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.data)
        self.cursor = 0

    def next_batch(self):
        # if any of the buckets is full go to next epoch
        if np.any(self.cursor+self.batch_size > self.size):
            self.epoch += 1
            self.shuffle() # Also resets cursor
        #pdb.set_trace()
        input_seq = self.data[self.cursor:self.cursor+self.batch_size]
        # Takes the list of lists of lists and makes a numpy array
        input_seq_time_major = batch_up(input_seq,
                                        self.batch_size,
                                        self.num_nodes,
                                        self.num_frames)

        self.cursor += self.batch_size
        #TODO add seq_length for variable length sequences
        return input_seq_time_major


def test_animation():
    plt.cla() # Clear figure
    N = 200 # Set number of nodes
    n_frames = 200
    limit = 102
    G = PlouffeSequence(N,98,limit,n_frames) # Initialize the graph G
    anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
    anim.save('PlouffeSequence200_98_102.gif', dpi=80, writer='imagemagick')


def test_iterator():
    df = make_dataset()
    test_iterator = Iterator(df.tolist())
    return test_iterator


