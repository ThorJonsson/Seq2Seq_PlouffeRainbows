import numpy as np
import pandas as pd
import random

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
    sequence_lengths = [len(seq) for seq in inputs]

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


# Simple bucket sequence iterator
class FibonacciSequenceIterator(object):
    def __init__(self, residue=10, batch_size = 64, num_buckets = 5):
        self.batch_size = batch_size
        self.residue = residue
        self.sequences = self._make_data()
        self.size = int(len(self.sequences)/num_buckets)
        self.bucket_data = []
        # Put the shortest sequences in the first bucket etc
        # bucket_data is a list of 'buckets' where each bucket is a list of Sentences.
        self.num_buckets = num_buckets
        for bucket in range(self.num_buckets):
            self.bucket_data.append(self.sequences[bucket*self.size: (bucket+1)*self.size -1])
        self.cursor = np.array([0]*num_buckets)
        self.shuffle()
        self.epochs = 0


    def _fibonacci(self, i, max_steps=10):
        '''
        at each step we obtain the next element in the sequence as the sum of the previous two elements modulo some
        basis_len
        '''
        j = np.random.randint(self.residue)
        seq = [i,j]
        steps = 2
        while steps < max_steps:
            r = (seq[steps-2]+seq[steps-1]) % self.residue
            seq.append(r)
            steps += 1
        return seq


    def _make_fibonacci_data(self):
        x = np.random.randint(self.residue,size=5000)
        df = pd.DataFrame(x)
        df['fibonacci'] = df[0].apply(self._fibonacci)
        return df


    def _make_data(self):
        df = self._make_fibonacci_data()
        sequences = df['fibonacci'].tolist()
        return sorted(sequences, key=lambda sequence: len(sequence))


    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            random.shuffle(self.bucket_data[i])
            self.cursor[i] = 0


    def next_batch(self):
        # if any of the buckets is full go to next epoch
        if np.any(self.cursor+self.batch_size > self.size):
            self.epochs += 1
            self.shuffle() # Also resets cursor

        i = np.random.randint(0,self.num_buckets)
        all_seq = self.bucket_data[i][self.cursor[i]:self.cursor[i]+self.batch_size]

        input_seq = []
        target_seq = []
        for seq in all_seq:
            split_idx = np.random.choice(range(len(seq)))
            input_seq.append(seq[0:split_idx])
            target_seq.append(seq[split_idx:len(seq)])

        input_seq_time_major, input_seq_lengths = batch_up(input_seq, self.batch_size)

        target_seq_time_major, target_seq_lengths = batch_up(target_seq, self.batch_size)

        self.cursor[i] += self.batch_size

        return input_seq_time_major, input_seq_lengths, target_seq_time_major, target_seq_lengths

## For iterating random sequences
def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]


def make_random_sequences_data(length_from,
                               length_to,
                               alphabet_lower,
                               alphabet_upper,
                               size):
    " Returns a random sample of the random sequences data "
    X = random_sequences(length_from,
                         length_to,
                         alphabet_lower,
                         alphabet_upper,
                         size)
    return X.next()
