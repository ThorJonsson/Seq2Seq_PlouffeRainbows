from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest

from tensorflow.contrib import layers
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
import pdb


def simple_decoder_fn_train(encoder_state, name=None):
    with ops.name_scope(name, "simple_decoder_fn_train", [encoder_state]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(name, "simple_decoder_fn_train",[time, cell_state, cell_input, cell_output, context_state]):
            if cell_state is None:
                return (None, encoder_state, cell_input, cell_output, context_state)
            else:
                return (None, cell_state, cell_input, cell_output, context_state)

    return decoder_fn


def regression_decoder_fn_inference(encoder_state,
                                    seq_length,
                                    batch_size,
                                    num_features,
                                    dtype=dtypes.int32,
                                    output_fn=None,
                                    name=None):
    """ Same function as simple_decoder_fn_inference but for regression on sequences with a fixed length
    """
    with ops.name_scope(name, "simple_decoder_fn_inference", [output_fn, encoder_state, seq_length, batch_size, num_features, dtype]):
        seq_length = ops.convert_to_tensor(seq_length, dtype)
        if output_fn is None:
            output_fn = lambda x: x # just take the output of the decoder

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """
        Again same as in simple_decoder_fn_inference but for regression on sequences with a fixed length
        """
        with ops.name_scope(name, "simple_decoder_fn_inference", [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" % cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input = array_ops.ones([batch_size, num_features], dtype=dtype)
                done = array_ops.zeros([batch_size], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros([num_features], dtype=dtypes.float32)
            else:
                cell_output = output_fn(cell_output)
            next_input = cell_output
            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(math_ops.greater(time, seq_length),
                                         lambda: array_ops.ones([batch_size, num_features], dtype=dtypes.bool),
                                         lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)
    return decoder_fn


def dynamic_rnn_decoder(cell, decoder_fn, inputs=None, sequence_length=None,
                        parallel_iterations=None, swap_memory=False,
                        time_major=False, scope=None, name=None):
    #pdb.set_trace()
    with ops.name_scope(name, "dynamic_rnn_decoder", [cell, decoder_fn, inputs, sequence_length,
                                              parallel_iterations, swap_memory, time_major, scope]):
        '''
        All this code is doing is making sure inputs are setup correctly
        '''
        if inputs is not None:
            inputs = ops.convert_to_tensor(inputs)
            # Test input dimensions
            if inputs.get_shape().ndims is not None and (
                inputs.get_shape().ndims < 2):
                raise ValueError("Inputs must have at least two dimensions")
            # Setup of RNN (dimensions, sizes, length, initial state, dtype)
            if not time_major:
                # [batch, seq, features] -> [seq, batch, features]
                inputs = array_ops.transpose(inputs, perm=[1, 0, 2])

            dtype = inputs.dtype
            # Get data input information
            input_depth = int(inputs.get_shape()[2])
            batch_depth = inputs.get_shape()[1].value
            max_time = inputs.get_shape()[0].value
            if max_time is None:
                max_time = array_ops.shape(inputs)[0]
            # Setup decoder inputs as TensorArray
            # TODO discuss TensorArrays
            inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unstack(inputs)
        '''
        yay everything is ok,
        The loop_fn specifies how the state of the RNN evolves as it reads the sequence
        Note that:
          inputs - (seq_length, batch_size, num_features)
          at each step we compute, using a dynamic transition function given by an rnn cell structure
          at time 0: We condition on the context vector and generate an output and evolve our state until we finish.
        '''
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_state is None:  # first call, before while loop (in raw_rnn)
                if cell_output is not None:
                    raise ValueError("Expected cell_output to be None when cell_state "
                                   "is None, but saw: %s" % cell_output)
                if loop_state is not None:
                    raise ValueError("Expected loop_state to be None when cell_state "
                                   "is None, but saw: %s" % loop_state)
                context_state = None
            else:  # subsequent calls, inside while loop, after cell excution
                if isinstance(loop_state, tuple):
                    (done, context_state) = loop_state
                else:
                    done = loop_state
                    context_state = None

            # call decoder function
            if inputs is not None:  # training
                # get next_cell_input
                if cell_state is None:
                    next_cell_input = inputs_ta.read(0)
                else:
                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = array_ops.shape(done)[0]

                    next_cell_input = control_flow_ops.cond(math_ops.equal(time, max_time),
                                                            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
                                                            lambda: inputs_ta.read(time))
                pdb.set_trace()
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = decoder_fn(time,
                                                                                                            cell_state,
                                                                                                            next_cell_input,
                                                                                                            cell_output,
                                                                                                            context_state)
            else:  # inference
                # next_cell_input is obtained through decoder_fn
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = decoder_fn(time,
                                                                                                            cell_state,
                                                                                                            None,
                                                                                                            cell_output,
                                                                                                            context_state)

            # check if we are done
            if next_done is None:  # training
                next_done = time >= sequence_length

            # build next_loop_state.
            if next_context_state is None:
                next_loop_state = next_done
            else:
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state, emit_output, next_loop_state)

        # Run raw_rnn function
        outputs_ta, final_state, final_loop_state = rnn.raw_rnn(cell, loop_fn, parallel_iterations=parallel_iterations, swap_memory=swap_memory, scope=scope)
        outputs = outputs_ta.stack()

        # Get final context_state, if generated by user
        if isinstance(final_loop_state, tuple):
            final_context_state = final_loop_state[1]
        else:
            final_context_state = None

        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
        return outputs, final_state, final_context_state


