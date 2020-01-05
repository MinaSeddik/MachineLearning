from __future__ import absolute_import, division, print_function, unicode_literals

# https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    # Enable debug mode
    tf.debugging.set_log_device_placement(True)

    print('Tensorflow version:')
    print(tf.__version__)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)
