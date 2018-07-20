import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# Function
def get_layer(layer_number, fan_in, fan_out, prev_layer):
    W = tf.get_variable("W" + str(layer_number), shape=[fan_in, fan_out],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([fan_out]))
    L = tf.matmul(prev_layer, W) + b
    return W, b, L


# constant
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1, b1, _L1 = get_layer(1, 784, 512, X)
L1 = tf.nn.relu(_L1)
W2, b2, _L2 = get_layer(2, 512, 512, L1)
L2 = tf.nn.relu(_L2)
W3, b3, _L3 = get_layer(3, 512, 512, L2)
L3 = tf.nn.relu(_L3)
W4, b4, _L4 = get_layer(4, 512, 512, L3)
L4 = tf.nn.relu(_L4)
W5, b5, _L5 = get_layer(5, 512, 10, L4)
hypothesis = _L5

# cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy,
                            feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
