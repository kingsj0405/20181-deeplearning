import tensorflow as tf
import numpy as np


# Make data
# sample = " if you want you"  # work very well
sample = " Mr. and Mrs. Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sample]

x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

# constant from sample
input_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 1e-2

# placeholder
X = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)
Y = tf.placeholder(tf.int32, [None, sequence_length])

# RNN model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

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
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        if i % 200 == 0:
            result = sess.run(prediction, feed_dict={X: x_data})
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print(i, "loss:", l, "\tPrediction str: ", ''.join(result_str))
