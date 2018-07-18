import tensorflow as tf

# filename queue
filename_queue = tf.train.string_input_producer([
    "data-04-zoo.csv"
], shuffle=False)

# file reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# file decoder
record_defaults = [[0.] for x in range(17)]
data = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')
x_data, y_data = tf.train.batch([data[0:-1], data[-1:]], batch_size=10)

# placeholder
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])  # shape[?, 1]
Y_one_hot = tf.one_hot(Y, 7)  # shape[?, 1, 7]
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])  # shape[?, 7]

# variable
W = tf.Variable(tf.random_normal([16, 7]), name='weight')
b = tf.Variable(tf.random_normal([7]), name='bias')

# hypothesis
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross Entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# prediction and accuracy
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Make coordinator for thread sync
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # train
    for step in range(2001):
        x_batch, y_batch = sess.run([x_data, y_data])
        sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={X: x_batch, Y: y_batch})
            print("Step: {:5}\t Loss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc
            ))

    # test
    x_batch, y_batch = sess.run([x_data, y_data])
    pred = sess.run(prediction, feed_dict={X: x_batch})
    for p, y in zip(pred, y_batch.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    coord.request_stop()
    coord.join(threads)
