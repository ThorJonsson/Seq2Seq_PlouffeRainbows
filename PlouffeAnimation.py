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

    def next_frame(self,step):
        self.fig.clf()
        self.cursor += self.step
        # update graph - remove existing edges and generate edges from the new cursor value
        self._update_graph()
        # generate new drawing elements for animation
        nodes = nx.draw_networkx_nodes(self.plouffe.graph,pos=self.pos,node_size=10,node_label=False)
        edges = nx.draw_networkx_edges(self.plouffe.graph,pos=self.pos)
        return nodes, edges


def get_plouffe_seq(init, num_nodes = 5, n_frames=2):
    limit = init+10
    Plouffe_Seq = [[int(i*j)%num_nodes for i in range(num_nodes)] for j in np.arange(init,limit, (limit-init)/float(n_frames))]
    return Plouffe_Seq


def make_dataset():
    x = np.random.randint(50, size=5)
    df = pd.DataFrame(x)
    df['Plouffe'] = df[0].apply(get_plouffe_seq)
    return df


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

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        inputs_batch_major[i] = seq

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


# Simple sequence iterator
class Iterator(object):

    def __init__(self, Plouffe_Sequences, num_nodes = 10, num_frames = 200, batch_size = 1):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
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
            self.epochs += 1
            self.shuffle() # Also resets cursor

        target_seq = self.data[self.cursor:self.cursor+self.batch_size]
        input_seq = [list(reversed(seq)) for seq in input_seq]

        input_seq_time_major = batch_up(input_seq, self.batch_size)

        target_seq_time_major = batch_up(target_seq, self.batch_size)

        self.cursor += self.batch_size

        return input_seq_time_major, input_seq_lengths, target_seq_time_major, target_seq_lengths


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


