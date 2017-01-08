#!/usr/bin/env python
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
LAYER_SIZE = 500
LEARNING_RATE = 0.05
EPOCHS = 50
DECAY = 0.97
NUM_BATCHES = 50


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
  W3 = tf.Variable(tf.zeros([LAYER_SIZE, num_crops*num_plots]))
  b3 = tf.Variable(tf.zeros([num_crops*num_plots]))
  y = tf.matmul(y1, W3) + b3
  #y = tf.sigmoid(y)

  probabilities = []
  for i in range(0, num_plots):
    probabilities.append(tf.nn.softmax(y[:,i*num_crops:(i+1)*num_crops]))

  y_ = tf.placeholder(tf.float32, [None, num_crops*num_plots])
  cross_entropy_total = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_[:,0:num_crops],
                    logits=y[:,0:num_crops]))
  for i in range(1, num_plots):
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
              labels=y_[:,i*num_crops:(i+1)*num_crops],
              logits=y[:,i*num_crops:(i+1)*num_crops]))
    cross_entropy_total = tf.add(cross_entropy, cross_entropy_total)
  cross_entropy_total = tf.mul(cross_entropy_total, 1.0/num_plots)
  lr = tf.Variable(0.0, trainable=False)
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy_total)
  # Train
  probability = []

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state("save")
    saver.restore(sess, ckpt.model_checkpoint_path)
    data = [float(q) for q in FLAGS.data.split(",")]
    data = [q/16 for q in data]
    data = np.array([data], dtype=np.float32)
    print data
    probability = sess.run(probabilities, feed_dict={x: data})
  output = ""
  for i in probability:
    output += ",".join(map(str,i.tolist()[0])) + "\n"
  print output
  with open("distribution3.csv", "w") as f:
    f.write(output)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='3,3,3,3,3',
                      help='Directory for storing input data')
  parser.add_argument('--num_crops', type=int, default=5,
                   help='Number of crops')
  parser.add_argument('--num_plots', type=int, default=16,
                   help='Number of crops')
  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)