import tensorflow as tf
import numpy as np

# filename queue
filename_queue = tf.train.string_input_producer(
    ["data-03-diabetes.csv"],
    shuffle=False
)

# file reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# file decoder
record_defaults = [[0.] for x in range(9)]
data = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')
x_data, y_data = tf.train.batch([data[0:-1], data[-1:]], batch_size=10)

# placeholder
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost function and optimizer
cost = -tf.reduce_mean(Y * tf.log(hypothesis) +
                       (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

# Accuracy computation
threshold = 0.5
predicted = tf.cast(hypothesis > threshold, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Coordinator for thread sync
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([x_data, y_data])
        cost_val, _ = sess.run([cost, train],
                               feed_dict={X: x_batch, Y: y_batch})
        if step % 200 == 0:
            print(step, cost_val)

    x_batch, y_batch = sess.run([x_data, y_data])
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    # sync on here
    coord.request_stop()
    coord.join(threads)
