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
        self.plouffe.data = [(i, int(i*self.cursor)%self.n_nodes)for i in range(self.n_nodes)]
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

def test():
    plt.cla() # Clear figure
    N = 200 # Set number of nodes
    n_frames = 200
    limit = 102
    G = PlouffeSequence(N,98,limit,n_frames) # Initialize the graph G
    anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
    anim.save('PlouffeSequence200_98_102.gif', dpi=80, writer='imagemagick')
