from __future__ import print_function
import tensorflow as tf

g = tf.Graph()

with g.as_default():

    x = tf.constant(2, name="x_const")
    y = tf.constant(2, name="y_const")
    z = tf.constant(2, name="z_const")

    sum = tf.add(x, y,name="sum")
    sum2 = tf.add(sum, z, name="sum2")
    with tf.Session() as sess:
        print(sum2.eval())