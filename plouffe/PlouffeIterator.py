import pandas as pd


def make_data():
    x = np.random.randint(5000, size=5000)
    df = pd.DataFrame(x)
    df['Plouffe'] = df[0].apply(

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

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        inputs_batch_major[i] = seq

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths



# Simple sequence iterator
class Iterator(object):

    def __init__(self, Plouffe_Sequences, num_nodes = 10, num_frames = 200, batch_size = 1):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.data = Plouffe_Sequences
        self.size = len(self.data)
        # cursor within an epoch
        self.cursor = 0
        self.epoch = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.data)
        self.cursor = 0

    def next_batch(self):
        # if any of the buckets is full go to next epoch
        if np.any(self.cursor+self.batch_size > self.size):
            self.epochs += 1
            self.shuffle() # Also resets cursor

        target_seq = self.data[self.cursor:self.cursor+self.batch_size]
        input_seq = [list(reversed(seq)) for seq in input_seq]

        input_seq_time_major = batch_up(input_seq, self.batch_size)

        target_seq_time_major = batch_up(target_seq, self.batch_size)

        self.cursor += self.batch_size

        return input_seq_time_major, input_seq_lengths, target_seq_time_major, target_seq_lengths
