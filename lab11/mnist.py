import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Read one img
img = mnist.train.images[0].reshape(28, 28)
img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=1e-2))
plt.subplot(3, 5, 1), plt.imshow(img.reshape(28, 28), cmap='gray')

# conv
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(
        3, 5, i + 1 + 5), plt.imshow(one_img.reshape(14, 14), cmap='gray')

# max pooling
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
print(pool)
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(3, 5, i + 1 + 10), plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.show()
