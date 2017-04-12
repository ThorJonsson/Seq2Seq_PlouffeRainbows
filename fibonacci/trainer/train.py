import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell

import sys
sys.path.append('../../')
from model.model import Seq2SeqModel, train_on_copy_task
import pandas as pd
import utils.seq_utils as seq_utils
from tqdm import tqdm

tf.reset_default_graph()
tf.set_random_seed(1)

def train_on_fibonacci_split():
    model = Seq2SeqModel(encoder_cell = LSTMCell(100), decoder_cell=LSTMCell(100), vocab_size=10,
                         embedding_size=10,bidirectional=False, attention=False)

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

train_on_fibonacci_split()
