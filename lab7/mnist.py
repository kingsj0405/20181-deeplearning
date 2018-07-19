import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data


# Functions
def add_layer(layer_number, input_wide, output_wide, prev_layer):
    with tf.name_scope("layer" + str(layer_number)):
        W = tf.Variable(tf.random_normal([input_wide, output_wide]))
        b = tf.Variable(tf.random_normal([output_wide]))
        layer = tf.matmul(prev_layer, W) + b

        w_hist = tf.summary.histogram("weights" + str(layer_number), W)
        b_hist = tf.summary.histogram("biases" + str(layer_number), b)
        layer3_hist = tf.summary.histogram("layer" + str(layer_number), layer)

        return W, b, layer


# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Make graph
input_wide = 784
first_wide = 1000
second_wide = 500
thrid_wide = 100
output_wide = 10

X = tf.placeholder(tf.float32, [None, input_wide])
Y = tf.placeholder(tf.float32, [None, output_wide])

W1, b1, layer1 = add_layer(1, input_wide, first_wide, X)
X2 = tf.sigmoid(layer1)

W2, b2, layer2 = add_layer(2, first_wide, second_wide, X2)
X3 = tf.sigmoid(layer2)

W3, b3, layer3 = add_layer(3, second_wide, thrid_wide, X3)
X4 = tf.sigmoid(layer3)

W4, b4, layer4 = add_layer(4, thrid_wide, output_wide, X4)
hypothesis = tf.nn.softmax(layer4)

# cost/loss function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost_summ = tf.summary.scalar("cost", cost)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# correct
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Run session
training_epochs = 50
batch_size = 100

global_step = 0
with tf.Session() as sess:
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/mnist_logs')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s, c, _ = sess.run([summary, cost, optimizer],
                               feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(s, global_step=global_step)
            global_step += 1
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))
    print("Accuracy:", accuracy.eval(session=sess,
                                     feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1),
                                  feed_dict={X: mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
               cmap='Greys', interpolation='nearest')
    plt.show()
