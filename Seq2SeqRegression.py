import math
import numpy as np
import random
import tensorflow as tf
import seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from decoder import simple_decoder_fn_train, regression_decoder_fn_inference
import seq_utils
import PlouffeAnimation
import pdb
from tqdm import trange
import time

slim = tf.contrib.slim
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


# TODO 
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

def test_and_display():
    N = 200 # Set number of nodes
    n_frames = 200
    limit = 102
    G = PlouffeSequence(N,98,limit,n_frames) # Initialize the graph G
    anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
    anim.save('PlouffeSequence200_98_102.gif', dpi=80, writer='imagemagick')


def train_on_plouffe_copy(checkpoint_name = 'Holuhraun'):
    log_dict = {'CheckpointName': checkpoint_name,'Epoch': [], 'TrainingLoss': [], 'MeanTrainingDuration': [], 'ValidationLoss': [], 'MeanValidDuration':[]}
    ########
    # Set Hyperparameters
    ########
    dataset_size = 1000
    num_nodes = 100
    batch_size = 10
    # TODO inference
    max_num_epoch = 100
    sample_step = 100
    seq_length = 200
    num_step = 100
    cell_size = 64
    ########
    # Define the Computational Graph
    ########
    encoder_input_ph = tf.placeholder(dtype=tf.float32,shape=(batch_size, seq_length, num_features), name='encoder_input')
    encoder_input, seq_length, decoder_input, decoder_target = preprocess(encoder_input_ph,
                                                                          seq_length,
                                                                          batch_size,
                                                                          num_features)

    encoder_output, encoder_state = init_simple_encoder(LSTMCell(cell_size), encoder_input, seq_length)
    decoder_logits = init_decoder_train(LSTMCell(cell_size), decoder_target, seq_length, encoder_state, num_features)
    loss, train_op = init_optimizer(decoder_logits, decoder_target)
    ########
    # Run Graph
    ########
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        df = PlouffeAnimation.make_dataset(dataset_size)
        data = df['Plouffe'].tolist()
        training_data = data[:int(dataset_size*0.8)]
        valid_data = data[int(dataset_size*0.8):int(dataset_size*0.95)]
        test_data = data[int(dataset_size*0.95):int(dataset_size)]

        train_iterator = PlouffeAnimation.Iterator(training_data,
                                                   num_nodes,
                                                   seq_length,
                                                   batch_size)

        valid_iterator = PlouffeAnimation.Iterator(valid_data,
                                                   num_nodes,
                                                   seq_length,
                                                   batch_size)

        step, mean_loss = 0,0
        train_losses, valid_losses = [], []
        current_epoch = 0
        while current_epoch < max_num_epoch:
            mean_train_duration = 0
            # tqdm iterator
            num_train_steps = int(dataset_size*0.8/batch_size)
            train_epoch = trange(num_train_steps, desc='Loss', leave=True)
            ### Training
            for _ in train_epoch:
                feed_dict = {encoder_input_ph: train_iterator.next_batch()}
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
                feed_dict = {encoder_input_ph: valid_iterator.next_batch()}
                start_time = time.time()
                valid_batch_loss = session.run([loss], feed_dict)
                duration = time.time() - start_time
                mean_valid_duration += duration
                step_desc = ('Epoch {}: loss = {} ({:.2f} sec/step)'.format(current_epoch, valid_batch_loss, duration))
                valid_epoch_mean_loss += valid_batch_loss
                valid_epoch.set_description(step_desc)
                valid_epoch.refresh()
            ### Logging
            log_dict['Epoch'] = current_epoch
            log_dict['TrainingLoss'] = train_epoch_mean_loss/num_train_steps
            log_dict['MeanTrainingDuration'] = mean_train_duration/num_train_steps
            log_dict['ValidationLoss'] = valid_epoch_mean_loss/num_valid_steps
            log_dict['MeanValidationDuration'] = mean_valid_duration/num_valid_steps

            log_df = pd.DataFrame(log_dict)
            ### Testing
            if current_epoch % sample_step == 0:
                test_and_display()


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