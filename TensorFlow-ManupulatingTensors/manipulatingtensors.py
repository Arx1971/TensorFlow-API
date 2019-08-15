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

sum = tf.add(primes, ones)
print("Sum of Primes and Ones: ", sum)

result = tf.multiply(primes, primes)

newResult = tf.subtract(result, 1)

print("Result: ", newResult)

# Reshaping Matrix

matrix = tf.constant(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
    dtype=tf.int32)

reshapaematrix_2_2_4 = tf.reshape(matrix, [2, 2, 4])
print(reshapaematrix_2_2_4.numpy())

a = tf.constant([5, 3, 2, 7, 1, 4], dtype=tf.int32)
b = tf.constant([4, 6, 3], dtype=tf.int32)

areshape = tf.reshape(a, [2, 3])
breshape = tf.reshape(b, [3, 1])

a_b = tf.matmul(areshape, breshape)

print("A X B :", a_b.numpy())

# Variables, Initialization and Assignment

v = tf.contrib.eager.Variable([3])

w = tf.contrib.eager.Variable(tf.random_normal([1, 4], mean=1.0, stddev=0.35))

print("V: ", v.numpy())
print("W: ", w.numpy())