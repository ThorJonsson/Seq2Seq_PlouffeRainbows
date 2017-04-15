import os
import math
import numpy as np
import random
import pdb
from tqdm import trange
import time
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from decoder import simple_decoder_fn_train, regression_decoder_fn_inference, dynamic_rnn_decoder
slim = tf.contrib.slim

import sys
sys.path.append("../")
from plouffe import PlouffeLib


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
def init_optimizer(decoder_logits, decoder_targets, lr):
    loss = l2_loss(logits=decoder_logits, targets=decoder_targets)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
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


def sample_Bernoulli(p):
    """
    Args:
      p: a float32 between 0 and 1 indicating the threshold
    Returns:
      a boolean sampled from a Bernoulli distribution
    """
    x = tf.random_uniform(())
    return tf.greater_equal(x,tf.constant(p))


def get_Plouffe_reconstruction(decoder_logits, seq_length, num_nodes):
    """
    Args:
      decoder_logits: The logits that need to be reversed and decompressed into a Plouffe sequence
    Returns:
      A Plouffe Sequence as a batch_size x seq_length x num_nodes tensor
    """
    # 0 is the sequence axis and 1 is the batch axis
    reversed_logits = tf.reverse_sequence(decoder_logits, seq_length, 0, 1)
    # Batch major
    return tf.transpose(reversed_logits, [1,0,2])*num_nodes


def _save_df(log_df, file_name):
    """Saves dataframe to hdf"""
    print('Saving dataframe to', file_name)
    log_df.to_pickle(file_name)


def restore_checkpoint_variables(session, checkpoint_path):
    """Initializes the model in the graph of a passed session with the
    variables in the file found in `checkpoint_path`, except those excluded by
    `checkpoint_exclude_scopes`.
    """
    if checkpoint_path is None:
        return
    else:
        variables_to_restore = tf.global_variables()
        restorer = tf.train.Saver(var_list=variables_to_restore)
        restorer.restore(sess=session, save_path=checkpoint_path)


def train_on_plouffe_copy(sess_args, load_params):
    if load_params == 0:
      ########
      # Set Hyperparameters
      ########
      num_frames = sess_args['hyperparameters.numFrames']
      num_nodes = sess_args['hyperparameters.numNodes']
      batch_size = sess_args['hyperparameters.batchSize']
      cell_size = sess_args['networkOptions.cellSize']
      dataset_size = sess_args['datasetParams.datasetSize']
      max_num_epoch = sess_args['hyperparameters.maxEpoch']
      teacher_forcing_prob = sess_args['hyperparameters.teacherForcingProb']
      learning_rate = sess_args['hyperparameters.learningRate']
      checkpoint_path = os.getcwd() + sess_args['globalParams.checkpointDir']
      checkpoint_name = sess_args['globalParams.checkpointName']

    else:
      ########
      # Set Hyperparameters
      ########
      num_frames = sess_args['numFrames']
      num_nodes = sess_args['numNodes']
      #print(num_nodes)
      batch_size = sess_args['batchSize']
      cell_size = sess_args['cellSize']
      dataset_size = sess_args['datasetSize']
      max_num_epoch = sess_args['maxEpoch']
      teacher_forcing_prob = sess_args['teacherForcingProb']
      learning_rate = sess_args['learningRate']
      checkpoint_path = sess_args['checkpointDir']
      checkpoint_name = sess_args['checkpointName']

    log_dict = {'CheckpointName': checkpoint_name, 'Epoch': [], 'TrainingLoss': [], 'MeanTrainingDuration': [], 'ValidationLoss': [], 'MeanValidDuration':[]}

    try:
        os.makedirs(checkpoint_path)
    except OSError:
        if not os.path.isdir(checkpoint_path):
            raise

    sample_step = 100
    p = 0.5

    ########
    # Define the Computational Graph
    ########
    encoder_input_ph = tf.placeholder(dtype=tf.float32, shape=(None, num_frames, num_nodes), name='encoder_input')
    is_validation = tf.placeholder(tf.bool, name='is_validation')

    encoder_input, seq_length, decoder_input, decoder_target = preprocess(encoder_input_ph,
                                                                          num_frames,
                                                                          batch_size,
                                                                          num_nodes)

    encoder_output, encoder_state = init_simple_encoder(LSTMCell(cell_size),
                                                        encoder_input,
                                                        seq_length)
    context_vector = encoder_output[-1]

    decoder_logits_train = decoder_teacher_forcing(LSTMCell(cell_size),
                                                   context_vector,
                                                   encoder_state,
                                                   decoder_target,
                                                   seq_length,
                                                   num_nodes)

    decoder_logits_valid = decoder_inference(LSTMCell(cell_size),
                                             context_vector,
                                             encoder_state,
                                             batch_size,
                                             num_frames,
                                             num_nodes)

    is_teacher_forcing = tf.logical_or(sample_Bernoulli(teacher_forcing_prob), is_validation)

    decoder_logits = tf.where(is_teacher_forcing, decoder_logits_valid, decoder_logits_train)

    loss, train_op = init_optimizer(decoder_logits, decoder_target,
                                    learning_rate)
    # We need the values to be between 0 and 1 to be easy to parameterize with a network for regression
    decoder_prediction = get_Plouffe_reconstruction(decoder_logits, seq_length, num_nodes)
    saver = tf.train.Saver(var_list=tf.global_variables())
    ########
    # Run Graph
    ########
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train_df = PlouffeLib.make_dataset(dataset_size, num_nodes, num_frames)
        data = train_df['Plouffe'].tolist()
        training_data = data[:int(dataset_size*0.8)]
        valid_data = data[int(dataset_size*0.8):int(dataset_size)]

        train_iterator = PlouffeLib.Iterator(training_data,
                                             num_nodes,
                                             num_frames,
                                             batch_size)

        valid_iterator = PlouffeLib.Iterator(valid_data,
                                             num_nodes,
                                             num_frames,
                                             batch_size)

        train_epoch_mean_loss, valid_epoch_mean_loss, step, mean_loss = 0,0,0,0
        train_losses, valid_losses = [], []
        mean_train_duration_list, train_batch_loss_list = [], []
        mean_valid_duration_list, valid_batch_loss_list = [], []

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
                mean_train_duration_list.append(mean_train_duration)
                step_desc = ('Epoch {}: loss = {} ({:.2f} sec/step)'.format(current_epoch, train_batch_loss, duration))
                train_batch_loss_list.append(train_batch_loss)
                train_epoch_mean_loss += train_batch_loss
                train_epoch.set_description(step_desc)
                train_epoch.refresh()

            mean_valid_duration = 0
            # tqdm iterator
            #assert dataset_size*0.15 < batch_size
            num_valid_steps = int(dataset_size*0.15/batch_size)
            valid_epoch = trange(num_valid_steps, desc='Loss', leave=True)
            ### Validating
            for _ in valid_epoch:
                feed_dict = {encoder_input_ph: valid_iterator.next_batch(), is_validation: True}
                start_time = time.time()
                valid_batch_loss = session.run([loss], feed_dict)
                duration = time.time() - start_time
                mean_valid_duration += duration
                mean_valid_duration_list.append(mean_valid_duration)
                step_desc = ('Epoch {}: loss = {} ({:.2f} sec/step)'.format(current_epoch, valid_batch_loss, duration))
                valid_batch_loss_list.append(valid_batch_loss)
                valid_epoch_mean_loss += valid_batch_loss[0]
                valid_epoch.set_description(step_desc)
                valid_epoch.refresh()

            ### Logging
            # TODO find a better way to log hyperparameters
            log_dict['num_frames'] = num_frames
            log_dict['num_nodes'] = num_nodes
            log_dict['batch_size'] = batch_size
            log_dict['cell_size'] = cell_size
            log_dict['dataset_size'] = dataset_size
            log_dict['max_num_epoch'] = max_num_epoch
            log_dict['checkpoint_path'] = checkpoint_path
            log_dict['checkpoint_name'] = checkpoint_name
            log_dict['Epoch'].append(current_epoch)
            log_dict['TrainingLoss'].append(train_batch_loss_list)
            log_dict['MeanTrainingDuration'].append(mean_train_duration_list)
            #print(valid_epoch_mean_loss)
            #print(num_valid_steps)
            log_dict['ValidationLoss'].append(valid_batch_loss_list)
            log_dict['MeanValidDuration'].append(mean_valid_duration_list)

            log_df = pd.DataFrame(log_dict)
            _save_df(log_df, checkpoint_path+checkpoint_name + '.pcl')
            current_epoch += 1
            saver.save(sess=session, save_path=checkpoint_path+checkpoint_name)
