import tensorflow as tf
import numpy as np


# constant
input_size = 5
hidden_size = 5
sequence_length = 6
batch_size = 1
learning_rate = 1e-1

# Make data
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

# RNN model
X = tf.placeholder(tf.float32, [None, sequence_length, input_size])
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# cost/loss function
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)  # HACK: outputs not for logits
loss = tf.reduce_mean(sequence_loss)

# optimizer
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction
prediction = tf.argmax(outputs, axis=2)

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        if i % 200 == 0:
            result = sess.run(prediction, feed_dict={X: x_one_hot})
            print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction str: ", ''.join(result_str))
