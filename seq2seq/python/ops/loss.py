# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

__all__ = ["sequence_loss", "l2_loss"]


def _add_weighted_loss_to_collection(losses, weights):
    """Weights `losses` by weights, and adds the weighted losses, normalized by
    the number of joints present, to `tf.GraphKeys.LOSSES`.

    Specifically, the losses are summed across all dimensions (x, y,
    num_joints), producing a scalar loss per batch. That scalar loss then needs
    to be normalized by the number of joints present. This is equivalent to
    sum(weights[:, 0, 0, :]), since `weights` is a [image_dim, image_dim] map
    of eithers all 1s or all 0s, depending on whether a joints is present or
    not, respectively.

    Args:
        losses: Element-wise losses as calculated by your favourite function.
        weights: Element-wise weights.
    """
    losses = tf.transpose(a=losses, perm=[1, 2, 0, 3])
    weighted_loss = tf.multiply(losses, weights)
    per_batch_loss = tf.reduce_sum(input_tensor=weighted_loss, axis=[0, 1, 3])

    num_joints_present = tf.reduce_sum(input_tensor=weights, axis=1)

    assert_safe_div = tf.assert_greater(num_joints_present, 0.0)
    with tf.control_dependencies(control_inputs=[assert_safe_div]):
        per_batch_loss /= num_joints_present

    total_loss = tf.reduce_mean(input_tensor=per_batch_loss)
    tf.add_to_collection(name=tf.GraphKeys.LOSSES, value=total_loss)


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: A 3D Tensor of shape
      [batch_size x sequence_length x num_decoder_symbols] and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype
      int. The target represents the true class at each timestep.
    weights: A 2D Tensor of shape [batch_size x sequence_length] and dtype
      float. Weights constitutes the weighting of each prediction in the
      sequence. When using weights as masking set all valid timesteps to 1 and
      all padded timesteps to 0.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide by the cost by the total label weight across
      timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
  if len(weights.get_shape()) != 2:
    raise ValueError("Weights must be a [batch_size x sequence_length] "
                     "tensor")
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    probs_flat = array_ops.reshape(logits, [-1, num_classes])
    targets = array_ops.reshape(targets, [-1])
    if softmax_loss_function is None:
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=probs_flat)
    else:
      crossent = softmax_loss_function(probs_flat, targets)
    crossent = crossent * array_ops.reshape(weights, [-1])
    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    else:
      batch_size = array_ops.shape(logits)[0]
      sequence_length = array_ops.shape(logits)[1]
      crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
    if average_across_timesteps and not average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[1])
      total_size = math_ops.reduce_sum(weights, axis=[1])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[0])
      total_size = math_ops.reduce_sum(weights, axis=[0])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    return crossent


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
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
  with ops.name_scope(name, "sequence_loss", [logits, targets]):
    num_features = array_ops.shape(logits)[2]
    batch_size = array_ops.shape(logits)[1]
    seq_length = array_ops.shape(logits)[0]
    # Get Loss Function
    l2loss = tf.square(tf.subtract(logits, targets))

    l2loss = math_ops.reduce_sum(l2loss)
    total_size = num_features*batch_size*seq_length+1e-12 # to avoid division by 0 for all-0 weights
    l2loss /= total_size
  return l2loss
