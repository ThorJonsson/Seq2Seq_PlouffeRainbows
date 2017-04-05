# Taken from this awesome github repo:
# https://github.com/ematvey/tensorflow-seq2seq-tutorials
# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
import math
import numpy as np
import random
import tensorflow as tf
import seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from decoder import simple_decoder_fn_train, regression_decoder_fn_inference
slim = tf.contrib.slim
import seq_utils
import pdb
MAX_SEQ_LENGTH = 100
PAD = 0
EOS = 1


def random_sequence(num_dim, num_length):
    # Realize walk
    return tf.stack([tf.random_uniform(shape=(num_dim,)) for i in range(num_length)])

def _get_preprocessed_random_sequence(max_seq_length,num_dim):
    """
    Args: max_seq_length: the maximum length allowed for a sequence
          num_dim: number of dimension for the components of the sequence
    Returns: A tuple of:
                    1. random sequence of num_dim dimensional components padded to max_seq_length
                    2. the length of each sequence as a (python/numpy) integer
    """
    seq_length = np.random.choice(np.arange(1,max_seq_length+1, step=1))
    X = random_sequence(num_dim, seq_length)
    # Add EOS (vector of ones)
    X = tf.concat([tf.ones(shape=(1,num_dim)),X],axis=0)
    # Pad to max_seq_length and return sample
    return tf.pad(X, [[0,max_seq_length-seq_length],[0,0]]), seq_length + 1


def make_dataset(size,max_seq_length,num_dim):
    """
    This function makes a toy dataset of random sequences of variable length (up to max_seq_length)
    The components of the sequences are vectors.
    Args: size: the size of our dataset
           max_seq_length: the maximum length allowed for a sequence 
           num_dim: number of dimension for the components of the sequence
    Returns: A tuple of:
                    1. lists where each component of the list is a tensor representing a sequence
                    2. list of python integers representing the length of the sequences in the other list
    """
    seqs = []
    seqs_lengths = []
    for i in range(size):
        seq_i, seq_length_i = _get_preprocessed_random_sequence(max_seq_length,num_dim)
        seqs.append(seq_i)
        seqs_lengths.append(seq_length_i)
    return seqs, seqs_lengths


class SequenceIterator(object):
    """ Simple iterator that works like all our other iterators """
    def __init__(self, seqs, seqs_lengths, num_features, batch_size):
        self.size = len(seqs)
        self.batch_size = batch_size
        self.seqs_len = seqs_lengths
        self.num_features = num_features
        self.seqs = seqs
        self.epoch = 0
        self.cursor = 0
        #self.shuffle()


    #def shuffle(self):
    #    random.shuffle(self.seqs)
    #    self.cursor = 0
    def next_batch(self):
        # if any of the buckets is full go to next epoch
        if (self.cursor+self.batch_size) > self.size:
            self.epoch += 1
            # self.shuffle() # Also resets cursor
        #pdb.set_trace()
        encoder_input_batch = tf.stack(self.seqs[self.cursor:self.cursor+self.batch_size])
        print(encoder_input_batch)
        encoder_input_lengths = self.seqs_len[self.cursor:self.cursor+self.batch_size]
        decoder_target_batch = tf.reverse_sequence(encoder_input_batch, encoder_input_lengths, 1, 0)
        # We need to transpose for time major
        encoder_input_batch = tf.transpose(encoder_input_batch, [1,0,2])
        decoder_target_batch = tf.transpose(decoder_target_batch, [1,0,2])
        print(decoder_target_batch)
        print(tf.ones(self.num_features))
        decoder_input_batch = tf.concat([tf.ones(shape=(1, self.batch_size, self.num_features)), decoder_target_batch],axis=0)
        self.cursor += self.batch_size

        return encoder_input_batch, encoder_input_lengths, decoder_input_batch, decoder_target_batch


class Seq2SeqModel():
    """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
    Requires TF 1.0.0-alpha"""

    def __init__(self,
                 encoder_cell,
                 decoder_cell,
                 embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False):

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.embedding_size = embedding_size

        self.bidirectional = bidirectional
        self.attention = attention

        self.make_graph()


    def init_placeholders(self):
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None, self.embedding_size),
            dtype=tf.float32,
            name='encoder_inputs',
        )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None, self.embedding_size),
            dtype=tf.float32,
            name='decoder_targets'
        )

        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )


    def init_decoder_train_connector(self):
        """
        The connector takes the decoder targets during training and applies EOS and PAD
        Note that here it's convenient to think about num_features axis as the z axis
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_MASK = tf.ones([self.batch_size, self.num_features], dtype=tf.int32)*self.EOS
            PAD_MASK = tf.ones([self.batch_size, self.num_features], dtype=tf.int32)*self.PAD

            # Because everything is time major the EOS Mask is easy - the first one
            # Note that we reverse the output sequence so the EOS comes first
            # We then unroll the sequence until reaching the end symbol
            # Note that upon inference we have no correct values to feed to the decoder
            self.decoder_train_inputs = tf.concat([EOS_MASK, self.decoder_targets],axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            # The targets go well with the PAD_MASK
            decoder_train_targets = tf.concat([self.decoder_targets, PAD_MASK], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1,0])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")


    def init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
                )

    # TODO Verify
    def init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            # TODO Explain, why would we want to use LSTMStateTuple?
            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')


    def init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return slim.fully_connected(outputs, self.num_features, scope=scope)

            # TODO attention without embeddings
            if not self.attention:
                decoder_fn_train = simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = regression_decoder_fn_inference(self.encoder_state,
                                                                       self.seq_length,
                                                                       self.batch_size,
                                                                       self.num_features,
                                                                       output_fn=output_fn)

                decoder_train_output = seq2seq.dynamic_rnn_decoder(cell=self.decoder_cell,
                                                           decoder_fn=decoder_fn_train,
                                                           inputs=self.decoder_targets,
                                                           sequence_length=self.decoder_targets_length,
                                                           time_major=True,
                                                           scope=scope)

                self.decoder_outputs_train, self.decoder_state_train, self.decoder_context_state_train = decoder_train_output

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            # We need the values to be between 0 and 1 to be easy to parameterize with a network for regression
            self.decoder_prediction_train = self.decoder_logits_train*self.num_features

            scope.reuse_variables()

            decoder_inference_out = seq2seq.dynamic_rnn_decoder(cell=self.decoder_cell,
                                                        decoder_fn=decoder_fn_inference,
                                                        time_major=True,
                                                        scope=scope)

            self.decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference = decoder_inference_out
            self.decoder_prediction_inference = self.decoder_logits_inference*self.num_features


    def init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        # TODO fix loss
        self.loss = seq2seq.l2_loss(logits=logits, targets=targets, weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


    def make_graph(self):
        self.init_placeholders()

        if self.bidirectional:
            self.init_bidirectional_encoder()
        else:
          self.init_simple_encoder()

        self.init_decoder()

        self.init_optimizer()


def train_on_fibonacci_split():
    model = Seq2SeqModel(LSTMCell(100), LSTMCell(100),
                         100,64,10,bidirectional=False, attention=False)

    sample_step = 100
    last_step = 10000

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


# Make an encoder function
def init_simple_encoder(encoder_cell, encoder_inputs, encoder_inputs_length):
    with tf.variable_scope("Encoder") as scope:
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=encoder_inputs,
                                                           sequence_length=encoder_inputs_length,
                                                           time_major=True,
                                                           dtype=tf.float32)
        return encoder_outputs, encoder_state

# Make a decoder function
def init_decoder_train(decoder_cell, decoder_targets, decoder_targets_length, encoder_state, num_features):
    with tf.variable_scope("Decoder") as scope:
        def output_fn(outputs):
            return slim.fully_connected(outputs, num_features, scope=scope)

        # TODO Comment
        decoder_output = seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                                     decoder_fn=simple_decoder_fn_train(encoder_state=encoder_state),
                                                     inputs=decoder_targets,
                                                     sequence_length=decoder_targets_length,
                                                     time_major=True,
                                                     scope=scope)

        decoder_outputs, decoder_state, decoder_context_state = decoder_output

        decoder_logits = output_fn(decoder_outputs)
        return decoder_logits


def init_decoder_inference(decoder_cell, encoder_state, seq_length, batch_size, num_features, output_fn):
    with tf.variable_scope("Decoder") as scope:
        scope.reuse_variables()

        decoder_fn_inference = regression_decoder_fn_inference(encoder_state,
                                                               seq_length,
                                                               batch_size,
                                                               num_features,
                                                               output_fn=output_fn)

        decoder_inference_out = seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                                            decoder_fn=decoder_fn_inference,
                                                            time_major=True,
                                                            scope=scope)

        decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference = decoder_inference_out
        # We need the values to be between 0 and 1 to be easy to parameterize with a network for regression
        decoder_prediction_inference = decoder_logits_inference*num_features
    return decoder_prediction_inference

# Setup optimizer function (returns the train_op)
def init_optimizer(decoder_logits, decoder_targets):
    # Question: Why transpose?
    #logits = tf.transpose(decoder_logits_train, [1, 0, 2])
    #targets = tf.transpose(decoder_train_targets, [1, 0])
    # TODO verify that home-made loss function works
    loss = l2_loss(logits=decoder_logits, targets=decoder_targets)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return loss, train_op

def l2_loss(logits, targets, name=None):
  """l2 loss for a sequence of logits (per example).

  Args:
    logits: A 3D Tensor of shape
      [batch_size x sequence_length x num_features] and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype
      int. The target represents the true class at each timestep.
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The l2 loss divided by the batch_size,
    the number of sequence components and the number of features.

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions.
  """
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 3:
    raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
  with tf.name_scope(name, "sequence_loss", [logits, targets]):
    num_features = tf.shape(logits)[2]
    batch_size = tf.shape(logits)[1]
    seq_length = tf.shape(logits)[0]
    # Get Loss Function
    l2loss = tf.square(tf.subtract(logits, targets))

    l2loss = tf.reduce_sum(l2loss)
    total_size = tf.to_float(num_features*batch_size*seq_length)+1e-12 # to avoid division by 0 for all-0 weights
    l2loss /= total_size
  return l2loss

if __name__=="__main__":
    #from tensorflow.contrib.rnn import LSTMCell
    #m_reg = Seq2SeqModel(LSTMCell(10), LSTMCell(10), 10, 3, 7, bidirectional=False, attention=False)
    num_features = 2
    batch_size = 2
    seqs, seqs_lengths = make_dataset(6,5,num_features)
    eh = SequenceIterator(seqs, seqs_lengths, num_features, batch_size)
    encoder_input, seq_length, decoder_input, decoder_target = eh.next_batch()
    encoder_output, encoder_state = init_simple_encoder(LSTMCell(10), encoder_input, seq_length)
    decoder_logits = init_decoder_train(LSTMCell(10), decoder_target, seq_length, encoder_state, num_features)
    loss, train_op = init_optimizer(decoder_logits, decoder_target)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Now we have encoder inputs and decoder targets so we need decoder inputs
        # We need to make sure that we can replicate this for other classes (e.g. the Plouffe Fractal and Cellular
        # Automata
        for 
            my_loss, my_train_op = sess.run([loss, train_op])
        # Next we need to
    #train_on_fibonacci_split()

