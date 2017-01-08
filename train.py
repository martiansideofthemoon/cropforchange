# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FLAGS = None
LAYER_SIZE = 100
LEARNING_RATE = 0.05
EPOCHS = 15
DECAY = 0.97
NUM_BATCHES = 50

def load_data(data_dir, num_crops, filename, plot):
  with open(os.path.join(data_dir, filename), 'r') as f:
    data = f.readlines()
  data = [x.split(',')[:-1] for x in data]
  inputs = []
  outputs = []
  for d in data:
    inputs.append(map(int, d[:num_crops]))
    output = d[5 + plot]
    if output == "None":
      one_hot = [0.2]*num_crops
    else:
      one_hot = [0]*num_crops
      one_hot[int(output)] = 1
    outputs.append(one_hot)
  inputs = np.array(inputs, dtype=np.float32)
  sum_arr = np.sum(inputs, axis=1)
  outputs = np.array(outputs)
  inputs = inputs / sum_arr[:, None]
  return inputs, outputs

def main(_):
  # Import data
  num_crops = FLAGS.num_crops
  num_plots = FLAGS.num_plots

  # Create the model
  x = tf.placeholder(tf.float32, [None, num_crops])
  W1 = tf.Variable(tf.zeros([num_crops, LAYER_SIZE]))
  b1 = tf.Variable(tf.zeros([LAYER_SIZE]))
  y1 = tf.matmul(x, W1) + b1
  y1 = tf.sigmoid(y1)
  W3 = tf.Variable(tf.zeros([LAYER_SIZE, num_crops]))
  b3 = tf.Variable(tf.zeros([num_crops]))
  y = tf.matmul(y1, W3) + b3
  probs = tf.nn.softmax(y)

  y_ = tf.placeholder(tf.float32, [None, num_crops])
  cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  lr = tf.Variable(0.0, trainable=False)
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
  # Train

  losses = []
  probability = []
  with tf.Session() as sess:
    for plot in range(num_plots):
      sess.run(tf.initialize_all_variables())
      inputs, outputs = load_data(FLAGS.data_dir, FLAGS.num_crops, 'data_train.csv', plot)
      eval_inputs, eval_outputs = load_data(FLAGS.data_dir, FLAGS.num_crops, 'data_eval.csv', plot)
      points = []
      losses.append(0)
      probability.append([])
      print "Solving for plot " + str(plot)
      for i in range(EPOCHS):
        sess.run(tf.assign(lr, LEARNING_RATE * (DECAY ** i)))
        for j in range(NUM_BATCHES):
          batch_size = int(len(inputs)/float(NUM_BATCHES))
          batch_xs = inputs[j*batch_size:(j+1)*batch_size,:]
          batch_ys = outputs[j*batch_size:(j+1)*batch_size,:]
          _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
          #print "The loss for iteration " + str(i*NUM_BATCHES + j) + " is " + str(loss)
          points.append([i*NUM_BATCHES + j, loss])
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={x: eval_inputs, y_: eval_outputs})
        print "Accuracy for plot " + str(plot) + " and epoch " + str(i) + " is " + str(acc)
        losses[plot] = acc
        probability[plot] = sess.run(probs)
  # points = np.array(points)
  # plt.plot(points[:,0],points[:,1],linewidth=2.0)
  # plt.show()
  import pdb
  pdb.set_trace()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory for storing input data')
  parser.add_argument('--num_crops', type=int, default=5,
                   help='Number of crops')
  parser.add_argument('--num_plots', type=int, default=16,
                   help='Number of crops')
  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)