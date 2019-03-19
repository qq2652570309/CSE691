import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def lrelu (tmp):
  return tf.nn.relu(tmp) - 0.01 * tf.nn.relu(-tmp)

Hin = np.array([[-0.5, 4],[-1.5, 5]])
Hout = lrelu(Hin)

sess = tf.Session()
print(sess.run(Hout))



