from __future__ import print_function

import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("Primes: ", primes)

ones = tf.ones([6], dtype=tf.int32)
print("ones: ", ones)