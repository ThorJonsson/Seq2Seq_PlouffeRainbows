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

# ATH NO EMBEDDINGS
class Seq2SeqModel():

    EOS = 0
    PAD = 1
    def __init__(self, encoder_cell, decoder_cell, num_nodes, bidirectional = True, attention=False)
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_nodes = num_nodes

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._make_graph()

    @property
    def decoder_hidden_units(self):
        return self.decoder_cell.output_size

    def _init_placeholders():
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )


    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))


            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")


    def _make_graph(self):
        self._init_placeholders()
        self._init_decoder_train_connectors()
        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()


    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
                )


    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                attention_values,
                attention_score_fn,
                attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


# Ath fix for Plouffe
def train_on_fibonacci_split():
    model = Seq2SeqModel(encoder_cell = LSTMCell(10), decoder_cell=LSTMCell(20), vocab_size=10, embedding_size=10)

    sample_step = 100
    last_step = 1000

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        iterator = seq_utils.FibonacciSequenceIterator()
        loss_track = []

        for i in tqdm(range(last_step)):
            inputs_, input_length, targets_, targets_length = iterator.next_batch()
            feed_dict = {model.encoder_inputs: inputs_,
                         model.encoder_inputs_length: input_length,
                         model.decoder_targets: targets_,
                         model.decoder_targets_length: targets_length,
                        }
            _, l = session.run([model.train_op, model.loss], feed_dict)
            loss_track.append(l)

            if i == 0 or i % sample_step == 0:
                print('batch {}'.format(i))
                print(' minibatch_loss: {}'.format(session.run(model.loss, feed_dict)))
                # Transposed for batch major
                for i, (e_in, dt_pred) in enumerate(zip(
                        feed_dict[model.decoder_targets].T,
                        session.run(model.decoder_prediction_train, feed_dict).T
                    )):
                    print('  sample {}:'.format(i + 1))
                    print('    enc input           > {}'.format(e_in))
                    print('    dec train predicted > {}'.format(dt_pred))
                    if i >= 2:
                        break
                print()


train_on_fibonacci_split()
