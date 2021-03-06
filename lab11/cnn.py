import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)
print("image.shape", image.shape)
# # Output
# plt.imshow(image.reshape(3, 3), cmap='Greys')
# plt.show()

weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],
                      [[[1., 10., -1.]], [[1., 10., -1.]]]])
print("weight.shape", weight)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# Output
# for i, one_img in enumerate(conv2d_img):
#     print(i, ": ", one_img.reshape(3, 3))
#     plt.subplot(1, 2, i+1)
#     plt.imshow(one_img.reshape(3, 3), cmap='gray')
# plt.show()

pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')
print("pool.shape", pool.shape)
print(pool.eval())
