import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# constant
learning_rate = 1e-2
training_epochs = 15
batch_size = 128

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1    ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=1e-2))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#       Conv     -> (?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
#       Pool     -> (?, 14, 14, 32)

# L2    ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=1e-2))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#       Conv     -> (?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
print(L2)
#       Pool     -> (?, 7, 7, 64)
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
#       Reshape  -> (?, 3136)

# FCL   FanIn shape=(?, 3136)
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

# cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
print("Learning started.")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch: ', '%04d' % (epoch + 1),
          'cost = ', '{:.9f}'.format(avg_cost))
print("Learning Finished!")

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy,
                            feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
