# Convolution Example
import tensorflow as tf

with tf.Session() as sess:
  x = tf.constant([#N \
                    [#C \
                    [#H \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] \
                    ] \
                    ] \
                    ])
  w = tf.constant([[[[1.0]],[[1.0]],[[1.0]]], [[[1.0]],[[1.0]],[[1.0]]], [[[1.0]],[[1.0]],[[1.0]]]])
  with tf.device("device:XLA_GPU:0"):
    y = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", True, 'NCHW')
  print(sess.run(y))
