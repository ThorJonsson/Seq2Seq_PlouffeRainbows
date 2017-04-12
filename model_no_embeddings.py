# Taken from this awesome github repo:
# https://github.com/ematvey/tensorflow-seq2seq-tutorials
# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
import math
import numpy as np
import random
import tensorflow as tf
import seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from decoder import simple_decoder_fn_train, regression_decoder_fn_inference, dynamic_rnn_decoder
import seq_utils
import PlouffeAnimation
import pdb
from tqdm import trange
import time
import pandas as pd
MAX_SEQ_LENGTH = 100
PAD = 0
EOS = 1

slim = tf.contrib.slim

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

                decoder_train_output = dynamic_rnn_decoder(cell=self.decoder_cell,
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


def preprocess(encoder_input, seq_length, batch_size, num_features):
    """
    This function takes a placeholder representing the encoder_input, and appends EOS to the front of it. 
    It then generates a corresponding decoder_input and decoder_target
    """
    #batch_size = tf.shape(encoder_input)[0]
    #TODO: Variable seq length
    encoder_input_lengths = tf.ones(batch_size,dtype=tf.int32)*seq_length
    decoder_target = tf.reverse_sequence(encoder_input, encoder_input_lengths, 1, 0)
    # We need to transpose for time major
    encoder_input = tf.transpose(encoder_input, [1,0,2])
    decoder_target = tf.transpose(decoder_target, [1,0,2])
    decoder_input = tf.concat([tf.ones(shape=(1, batch_size, num_features)), decoder_target],axis=0)

    return encoder_input, encoder_input_lengths, decoder_input, decoder_target


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
def decoder_teacher_forcing(decoder_cell, context_vector, encoder_state, decoder_targets, max_sequence_length, num_features):
    """
    Args:
      decoder_cell: The type of a gated RNN architecture that we use for the decoder
      context_vector: The last output of the encoder, which we condition on to reproduce the sequence
      encoder_state: The last hidden state of the encoder network
      decoder_targets: For teacher forcing! The sequence of vectors that the decoder ought to produce
      max_sequence_length: The longest sequence in the minibatch. To stop the decoder when training
      num_features: How many features to project the output of the RNN onto
    Returns:
      decoder_logits: A set of logits to be passed to our objective function
    """
    with tf.variable_scope("Decoder") as scope:
        def output_fn(outputs):
            return slim.fully_connected(outputs, num_features, scope=scope)

        decoder_fn_train = simple_decoder_fn_train(encoder_state, context_vector)

        decoder_output = dynamic_rnn_decoder(cell=decoder_cell,
                                             decoder_fn=decoder_fn_train,
                                             inputs=decoder_targets,
                                             sequence_length=max_sequence_length,
                                             time_major=True,
                                             scope=scope)

        decoder_outputs, decoder_state, decoder_context_state = decoder_output

        decoder_logits = output_fn(decoder_outputs)
    return decoder_logits


def decoder_inference(decoder_cell, context_vector, encoder_state, batch_size, max_sequence_length, num_features):
    """
    Args:
      decoder_cell: The type of a gated RNN architecture that we use for the decoder
      context_vector: The last output of the encoder, which we condition on to reproduce the sequence
      encoder_state: The last hidden state of the encoder network
      decoder_targets: For teacher forcing! The sequence of vectors that the decoder ought to produce
      max_sequence_length: The longest sequence in the minibatch. To stop the decoder when training
      num_features: How many features to project the output of the RNN onto
      batch_size: Needed to make a minibatch of GO tokens for the decoder
    Returns:
      decoder_prediction_inference: A reproduced PlouffeGraph given by the decoder
    """
    with tf.variable_scope("Decoder") as scope:
        def output_fn(outputs):
            return slim.fully_connected(outputs, num_features, scope=scope)

        scope.reuse_variables()

        decoder_fn_inference = regression_decoder_fn_inference(encoder_state,
                                                               context_vector,
                                                               max_sequence_length,
                                                               batch_size,
                                                               num_features,
                                                               output_fn)

        decoder_inference_out = dynamic_rnn_decoder(cell=decoder_cell,
                                                    decoder_fn=decoder_fn_inference,
                                                    time_major=True,
                                                    scope=scope)

        decoder_logits_inference, decoder_state_inference, decoder_context_state_inference = decoder_inference_out
    return decoder_logits_inference

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

def run_inference():
    assert not np.isnan(batch_loss)
    # TODO inference every now and then
    #print(l)
    if i == 0 or i % sample_step == 0:
        print('batch {}'.format(i))
        print(' minibatch_loss: {}'.format(session.run(loss, feed_dict)))
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

def sample_Bernoulli(p=0.5):
    """
    Args:
      p: a float32 between 0 and 1 indicating the threshold
    Returns:
      a boolean sampled from a Bernoulli distribution
    """
    x = tf.random_uniform(())
    return tf.greater_equal(x,tf.constant(p))


def test_and_display(session, encoder_input_ph, is_validation, decoder_predict):
    ''' This is what we want to reproduce with the network.'''
    N = 200 # Set number of nodes
    n_frames = 200
    limit = 102
    G = PlouffeAnimation.PlouffeSequence(N,98,limit,n_frames) # Initialize the graph G
    anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
    plt.show()
    anim.save('PlouffeSequence200_98_102.gif', dpi=80, writer='imagemagick')
    # TODO
    #feed_dict = {encoder_input_ph: }
    #my_prediction = session.run(decoder_predict)
    #print(my_prediction)

def train_on_plouffe_copy(checkpoint_name = 'Holuhraun'):
    log_dict = {'CheckpointName': checkpoint_name,'Epoch': [], 'TrainingLoss': [], 'MeanTrainingDuration': [], 'ValidationLoss': [], 'MeanValidDuration':[]}
    ########
    # Set Hyperparameters
    ########
    num_frames = 200
    num_nodes = 100
    batch_size = 10
    cell_size = 64
    # TODO inference
    dataset_size = 1000
    max_num_epoch = 100
    sample_step = 100
    num_step = 100
    p = 0.5
    ########
    # Define the Computational Graph
    ########
    encoder_input_ph = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_frames, num_nodes), name='encoder_input')
    is_validation = tf.placeholder(tf.bool, name='is_validation')

    encoder_input, seq_length, decoder_input, decoder_target = preprocess(encoder_input_ph,
                                                                          num_frames,
                                                                          batch_size,
                                                                          num_nodes)

    encoder_output, encoder_state = init_simple_encoder(LSTMCell(cell_size),
                                                        encoder_input,
                                                        seq_length)
    pdb.set_trace()
    context_vector = encoder_output[-1]


    decoder_logits_train = decoder_teacher_forcing(LSTMCell(cell_size),
                                                   context_vector,
                                                   encoder_state,
                                                   decoder_target,
                                                   seq_length,
                                                   num_nodes)

    decoder_logits_test = decoder_inference(LSTMCell(cell_size),
                                            context_vector,
                                            encoder_state,
                                            batch_size,
                                            num_frames,
                                            num_nodes)


    is_teacher_forcing = tf.logical_or(sample_Bernoulli(p), is_validation)

    decoder_logits = tf.where(is_teacher_forcing, decoder_logits_train, decoder_logits_test)

    loss, train_op = init_optimizer(decoder_logits, decoder_target)
    # We need the values to be between 0 and 1 to be easy to parameterize with a network for regression
    decoder_prediction = decoder_logits*num_nodes

    ########
    # Run Graph
    ########
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        df = PlouffeAnimation.make_dataset(dataset_size)
        data = df['Plouffe'].tolist()
        training_data = data[:int(dataset_size*0.8)]
        valid_data = data[int(dataset_size*0.8):int(dataset_size)]

        train_iterator = PlouffeAnimation.Iterator(training_data,
                                                   num_nodes,
                                                   num_frames,
                                                   batch_size)

        valid_iterator = PlouffeAnimation.Iterator(valid_data,
                                                   num_nodes,
                                                   num_frames,
                                                   batch_size)

        train_epoch_mean_loss, valid_epoch_mean_loss, step, mean_loss = 0,0,0,0
        train_losses, valid_losses = [], []
        current_epoch = 0
        while current_epoch < max_num_epoch:
            mean_train_duration = 0
            # tqdm iterator
            num_train_steps = int(dataset_size*0.8/batch_size)
            train_epoch = trange(num_train_steps, desc='Loss', leave=True)
            ### Training
            for _ in train_epoch:
                feed_dict = {encoder_input_ph: train_iterator.next_batch(), is_validation: False}
                start_time = time.time()
                train_batch_loss,_ = session.run([loss, train_op], feed_dict)
                duration = time.time() - start_time
                mean_train_duration += duration
                step_desc = ('Epoch {}: loss = {} ({:.2f} sec/step)'.format(current_epoch, train_batch_loss, duration))
                train_epoch_mean_loss += train_batch_loss
                train_epoch.set_description(step_desc)
                train_epoch.refresh()

            mean_valid_duration = 0
            # tqdm iterator
            num_valid_steps = int(dataset_size*0.15/batch_size)
            valid_epoch = trange(num_valid_steps, desc='Loss', leave=True)
            ### Validating
            for _ in valid_epoch:
                feed_dict = {encoder_input_ph: valid_iterator.next_batch(), is_validation: True}
                start_time = time.time()
                valid_batch_loss = session.run([loss], feed_dict)
                duration = time.time() - start_time
                mean_valid_duration += duration
                step_desc = ('Epoch {}: loss = {} ({:.2f} sec/step)'.format(current_epoch, valid_batch_loss, duration))
                valid_epoch_mean_loss += valid_batch_loss[0]
                valid_epoch.set_description(step_desc)
                valid_epoch.refresh()

            ### Logging
            log_dict['Epoch'] = current_epoch
            log_dict['TrainingLoss'] = train_epoch_mean_loss/num_train_steps
            log_dict['MeanTrainingDuration'] = mean_train_duration/num_train_steps
            log_dict['ValidationLoss'] = valid_epoch_mean_loss/num_valid_steps
            log_dict['MeanValidationDuration'] = mean_valid_duration/num_valid_steps

            log_df = pd.DataFrame(log_dict)
            current_epoch += 1
            ### Testing
            # Here we use the model in its current state and we try to reproduce the Plouffe Graph for an interesting
            # case.
            #if current_epoch % sample_step == 2:
            #    test_and_display(session, encoder_input_ph, is_validation, decoder_prediction)


if __name__=="__main__":
    train_on_plouffe_copy()
    '''with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Now we have encoder inputs and decoder targets so we need decoder inputs
        # We need to make sure that we can replicate this for other classes (e.g. the Plouffe Fractal and Cellular
        # Automata
        my_loss, my_train_op = sess.run([loss, train_op])
        # Next we need to
        #train_on_fibonacci_split()
    '''
