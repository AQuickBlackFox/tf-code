"""
Simple Vector Add
"""
import tensorflow as tf

with tf.Session() as sess:
  x = tf.placeholder(tf.float32, [4])
  y = tf.placeholder(tf.float32, [4])
  with tf.device("device:XLA_GPU:0"):
    z = x + y
  result = sess.run(z, {y : [0.1, -4.5, 5.43, 10.1] }, {x : [1.2, 5.1, -8.4, 0.3] })
