# Square of a vector
import tensorflow as tf
import numpy as np

_x = np.ones([128,2])
_w = np.ones([2, 4])

with tf.Session() as sess:
  x = tf.placeholder(tf.float32, [None, 2])
  w = tf.Variable(tf.zeros([2, 4]))
  with tf.device("device:XLA_GPU:0"):
    y = tf.matmul(x, w)
    print(sess.run(y, {x: _x, w: _w}))
