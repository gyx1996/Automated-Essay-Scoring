import tensorflow as tf
import numpy as np
import os
import random


def get_pure_batch(given_list, batch_size, ordering=True):
    def get_one():
        if ordering:
            while True:
                for i in range(len(given_list)):
                    yield given_list[i]
        else:
            while True:
                batch_index = random.sample(range(len(given_list)), batch_size)
                for i in range(batch_size):
                    yield given_list[batch_index[i]]

    def get_batch():
        while True:
            yield [next(one_element_generator) for _ in range(batch_size)]

    one_element_generator = get_one()
    batch_generator = get_batch()
    return batch_generator


class Model:
    def __init__(self,
                 max_length=500,
                 embedding_dim=200,
                 learning_rate=0.001,
                 epoch_num=500,
                 batch_size=32):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.name = 'essay_mse'

    def _encoder(self, embedded_batch_data):
        """

            embedded_batch_data:

        Returns:

        """
        with tf.name_scope('encoder'):
            fw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                      for size in [self.max_length]]
            initial_states_fw = [cell.zero_state(
                self.batch_size, dtype=tf.float32)
                                 for cell in fw_cells]
            bw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                for size in [self.max_length]]
            initial_states_bw = [cell.zero_state(
                self.batch_size, dtype=tf.float32)
                for cell in bw_cells]
            encoder_outputs_all, encoder_state_fw, encoder_state_bw = (
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, embedded_batch_data,
                    initial_states_fw=initial_states_fw,
                    initial_states_bw=initial_states_bw,
                    dtype=tf.float32))
            encoder_outputs = tf.concat(encoder_outputs_all, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c,
                                         encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h,
                                         encoder_state_bw[0].h), 1)
            encoder_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_state_c, h=encoder_state_h)
        return encoder_outputs, encoder_state

    def _scoring(self, input_x, output_y):
        with tf.variable_scope('scoring', reuse=tf.AUTO_REUSE):
            w = tf.get_variable(
                'w', shape=[self.batch_size, 2 * self.max_length],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', initializer=tf.zeros(
                [self.batch_size, 2 * self.max_length]))
            y_hat = tf.reduce_sum(w * input_x + b, 1)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(output_y - y_hat))
        return loss, y_hat

    def _build_graph(self, input_x, output_y):
        encoder_outputs, encoder_state = self._encoder(input_x)
        loss, y_hat = self._scoring(encoder_state.h, output_y)
        return loss, y_hat

    def train(self, x_train, y_train):
        """Train the model and save.

        Args:
            x_train, y_train: train data with the same element number
        """
        print('Building model for ' + self.name + '...')
        if not os.path.exists(self.name + '_checkpoints'):
            os.mkdir(self.name + '_checkpoints')

        with tf.name_scope('placeholder'):
            input_x = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, self.max_length, self.embedding_dim],
                name='batch_essays_embeddings')
            output_y = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, 1],
                name='score')
        loss, _ = self._build_graph(input_x, output_y)

        with tf.name_scope('optimize'):
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate).minimize(loss)

        x_train_batch_generator = get_pure_batch(
            x_train, self.batch_size, ordering=True)
        y_train_batch_generator = get_pure_batch(
            y_train, self.batch_size, ordering=True)

        print('Training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('graphs/' + self.name, sess.graph)

            for i in range(self.epoch_num):
                total_loss = 0.0
                for j in range(int(x_train.shape[0]/self.batch_size)):
                    x_train_batch = next(x_train_batch_generator)
                    y_train_batch = next(y_train_batch_generator)
                    loss_current, _ = sess.run(
                        [loss, optimizer],
                        feed_dict={
                            input_x: np.array(
                                x_train_batch).reshape(
                                (self.batch_size,
                                 self.max_length,
                                 self.embedding_dim)),
                            output_y: np.array(y_train_batch).reshape(
                                (self.batch_size, 1))})
                    total_loss += loss_current
                print('Epoch ' + str(i) + '/' + str(self.epoch_num + 1) +
                      ': loss: ' +
                      str(total_loss / int(x_train.shape[0] / self.batch_size)))
                saver.save(sess, self.name + '_checkpoints/training', i)

    def test(self, x_test, y_test):
        print('Testing...')
        x_test_batch_generator = get_pure_batch(
            x_test, self.batch_size, ordering=True)
        y_test_batch_generator = get_pure_batch(
            y_test, self.batch_size, ordering=True)
        with tf.Session() as sess:
            input_x = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, self.max_length, self.embedding_dim],
                name='batch_essays_embeddings')
            output_y = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, 1],
                name='score')

            loss, y_hat = self._build_graph(input_x, output_y)

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                self.name + '_checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0.0
            y_result = []
            for j in range(int(x_test.shape[0] / self.batch_size) + 1):
                x_test_batch = next(x_test_batch_generator)
                y_test_batch = next(y_test_batch_generator)
                loss_current, y_hat_current = sess.run(
                    [loss, y_hat],
                    feed_dict={
                        input_x: np.array(x_test_batch).reshape(
                            (self.batch_size,
                             self.max_length,
                             self.embedding_dim)),
                        output_y: np.array(y_test_batch).reshape(
                            (self.batch_size, 1))})
                total_loss += loss_current
                y_result = np.append(y_result, y_hat_current)
            y_result = y_result[:x_test.shape[0]]
        print(y_result)
        print(y_test)
        print('Test loss:' +
              str(total_loss / (int(x_test.shape[0] / self.batch_size))))
        return y_result
