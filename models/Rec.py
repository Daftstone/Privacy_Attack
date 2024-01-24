import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import math
import utils

from parse import FLAGS


class SVD:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_users = len(self.dataset.rec_data)
        self.num_items = len(self.dataset.rec_data[0])
        self.num_factors = 64
        self.reg = 1e-12
        self.build_graph()
        self.train_data = self.process_data()
        print('train data', len(self.train_data[0]))

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')

    def create_user_terms(self):
        num_users = self.num_users
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('user'):
            self.user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.p_u = tf.reduce_sum(tf.nn.embedding_lookup(
                self.user_embeddings,
                self.users_holder,
                name='p_u'), axis=1)

    def create_item_terms(self):
        num_items = self.num_items
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('item'):
            self.item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.q_i = tf.reduce_sum(tf.nn.embedding_lookup(
                self.item_embeddings,
                self.items_holder,
                name='q_i'), axis=1)

    def create_prediction(self):
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(tf.multiply(self.p_u, self.q_i), axis=1)
            self.pred = tf.expand_dims(pred, axis=-1)
            self.rate = tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings))

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.ratings_holder - self.pred))
            self.all_loss = tf.add(self.loss,
                               (tf.reduce_mean(self.p_u * self.p_u) + tf.reduce_mean(self.q_i * self.q_i)) * 0.01,
                               name='loss')
            self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.all_loss, name='optimizer')

    def build_graph(self):
        self.create_placeholders()
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, epochs, batch_size):
        for cur_epoch in range(epochs):
            loss_list = []
            batchs = self.get_batchs(batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                feed_dict = {self.users_holder: users,
                             self.items_holder: items,
                             self.ratings_holder: rates}
                _, rmse = self.sess.run([self.train_op, self.loss], feed_dict)
                loss_list.append(rmse)

            rate = self.sess.run(self.rate)
            hit, loss = self.evaluate(rate)
            print("epoch %d: train: %f test: %f %f" % (cur_epoch, np.mean(loss_list), hit, loss))

    def evaluate(self, rate):
        mask = self.train_matrix != 0
        rate -= mask * 1e10
        hit_list = []
        loss_list = []
        test_data = self.dataset.rec_data_test
        test_label = self.dataset.rec_label_test
        for i in range(len(test_data)):
            idx1 = test_data[i, 0]
            idx2 = test_data[i, 1]
            sort_rate = np.sort(rate[idx1])[::-1]
            hit_list.append(rate[idx1, idx2] >= sort_rate[FLAGS.top_k - 1])
            loss_list.append(abs(rate[idx1, idx2] - test_label[i]))
        return np.mean(hit_list), np.mean(loss_list)

    def process_data(self):
        import scipy.sparse as sp
        self.train_matrix = self.dataset.rec_data
        data_sp = sp.coo_matrix(self.dataset.rec_data)
        rows = data_sp.row
        cols = data_sp.col
        data = data_sp.data
        assert np.sum(data == 0) == 0
        return [rows[:, None], cols[:, None], data[:, None]]

    def get_batchs(self, batch_size):
        rows, cols, data = self.train_data
        idx = np.random.permutation(len(rows))
        train_list = []
        for i in range(len(idx[:-1]) // batch_size + 1):
            cur_idx = idx[batch_size * i:min(batch_size * i + batch_size, len(idx))]
            train_list.append([rows[cur_idx], cols[cur_idx], data[cur_idx]])
        return train_list
