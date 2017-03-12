import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from model import Seq2SeqModel, train_on_copy_task
import pandas as pd
import seq_utils
from tqdm import tqdm

tf.reset_default_graph()
tf.set_random_seed(1)


def train_original_model():
    with tf.Session() as session:
        # with bidirectional encoder, decoder state size should be
        # 2x encoder state size
        model = Seq2SeqModel(encoder_cell=LSTMCell(10),
                             decoder_cell=LSTMCell(20),
                             vocab_size=10,
                             embedding_size=10,
                             attention=True,
                             bidirectional=True,
                             debug=False)

        session.run(tf.global_variables_initializer())

        train_on_copy_task(session, model,
                           length_from=3, length_to=8,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=100,
                           max_batches=3000,
                           batches_in_epoch=1000,
                           verbose=True)


def train_on_copy_task_v2():
    model = Seq2SeqModel(encoder_cell=LSTMCell(10),
                         decoder_cell=LSTMCell(20),
                         vocab_size=10,
                         embedding_size=10,
                         bidirectional=True,
                         attention=True)

    sample_step = 100
    last_step = 1000

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        data = seq_utils.random_sequences(length_from=0, length_to=10, vocab_lower=0, vocab_upper=2, batch_size=10)
        loss_track = []

        for i in tqdm(range(last_step)):
            input_seq = next(data)
            inputs_, input_length = seq_utils.batch(input_seq)
            targets_, targets_length = seq_utils.batch(input_seq)
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
                        feed_dict[model.encoder_inputs].T,
                        session.run(model.decoder_prediction_train, feed_dict).T
                    )):
                    print('  sample {}:'.format(i + 1))
                    print('    enc input           > {}'.format(e_in))
                    print('    dec train predicted > {}'.format(dt_pred))
                    if i >= 2:
                        break
                print()


def train_on_fibonacci_split():
    model = Seq2SeqModel(encoder_cell = LSTMCell(100), decoder_cell=LSTMCell(200), vocab_size=10,
                         embedding_size=10,bidirectional=True, attention=True)

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
