# Thor H. Jonsson
# LSTM with Plouffe Sequences
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

def make_FLAGS(): 
    ### HyperParameters ###
    flags = tf.flags
    flags.DEFINE_integer('batch_size', 100, 'mini-batch size')
    flags.DEFINE_integer('max_epoch', 600, 'maximum number of epochs')
    flags.DEFINE_integer('save_every', 10, 'Saving frequency(epoch), 0 if you do not wish to save')
    flags.DEFINE_integer('val_every', 1, 'Validate frequency(epoch), 0 if you do not wish to validate')
    flags.DEFINE_float('lr', 0.01, 'learning rate')
    flags.DEFINE_float('lr_decay', 1, 'learning rate decay')
    flags.DEFINE_float('init_mom', 0.5, 'initial momentum')
    flags.DEFINE_float('final_mom', 0.9, 'final momentum')
    
    return flags.FLAGS

basis_len = 50

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

class PlouffeIterator(object):
    '''
    NOT FINISHED
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

class SequenceClassification(object):

    def __init__(self, FLAGS=None):
        self.state_size = FLAGS.state_size,
        self.num_classes = FLAGS.num_classes
        self.batch_size = FLAGS.batch_size,
        self.num_steps = FLAGS.num_steps,
        self.num_layers = FLAGS.num_layers,
        self.learning_rate = FLAGS.learning_rate

        self.graph = self.build_graph()
    
    def build_graph(self):
        '''
        builds the tensorflow graph and returns a dictionary of placeholders and corresponding variables
        '''
        x = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='labels_placeholder')

        embeddings = tf.get_variable('embedding_matrix', [self.num_classes, self.state_size])

        # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
        
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))

        #reshape rnn_outputs and y so we can get the logits in a single matmul
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.state_size])
        y_reshaped = tf.reshape(y, [-1])

        logits = tf.matmul(rnn_outputs, W) + b

        total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        return dict(x = x,
                    y = y,
                    init_state = init_state,
                    final_state = final_state,
                    total_loss = total_loss,
                    train_step = train_step)

    def reset_graph():
        '''
        utilty function to make sure that the graph is clean before we build it
        '''
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()
    


    def train_on_plouffe(graph):
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        sequences = fetch_plouffe()

        train_seq = sequences[:int(len(sequences)*0.8)]
        valid_seq = sequences[int(len(sequences)*0.8):int(len(sequences)*0.95)]
        test_seq = sequences[int(len(sequences)*0.95):int(len(sequences))]
    
        train_iterator = PlouffeIterator(train_seq)
        valid_iterator = PlouffeIterator(valid_seq)
        test_iterator = PlouffeIterator(test_seq)

        step, mean_loss = 0,0
        train_losses, valid_losses = [], []
        current_epoch = 0
        while current_epoch < n_epochs:
            step += 1
            batch = train_iterator.next_batch(self.batch_size)
            feed = {graph['X']: batch[0], graph['Y']: batch[1],graph['seqlen']: batch[2] }

            mean_loss_batch, _ = sess.run([graph['mean_loss'],graph['updates']], feed_dict=feed)
            mean_loss += mean_loss_batch

        if train_iterator.epochs > current_epoch:
            current_epoch += 1
            train_losses.append(mean_loss / step)
            step, mean_loss = 0, 0

            # eval test set
            valid_epoch = valid_iterator.epochs
            while valid_iterator.epochs == valid_epoch:
                step += 1
                batch = valid_iterator.next_batch(self.batch_size)
                feed = {graph['X']: batch[0], graph['Y']: batch[1],graph['seqlen']: batch[2]}
                mean_loss_batch = sess.run([graph['mean_loss']], feed_dict=feed)[0]
                mean_loss += mean_loss_batch

            valid_losses.append(mean_loss / step)
            step, mean_loss = 0,0
            print('Accuracy after epoch', current_epoch, ' - train loss:', train_losses[-1], '- validation loss:', valid_losses[-1])

            if current_epoch % 2 == 0:
                batch = test_iterator.next_batch(10)
                feed = {graph['X']:batch[0], graph['Y']:batch[1], graph['seqlen']:batch[2]}
                p = sess.run([graph['Y_pred']], feed_dict = feed)
                print('Prediction:',p)
                print('Real Output:', batch[1])

        return train_losses, valid_losses

if __name__ == "__main__":
    graph = build_graph()
    train_graph(graph)
