import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class SVDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.get_svd_data()
        self.create_init()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32)
        self.label = tf.placeholder(dtype=tf.float32)

    def create_weights(self):
        self.weights = []
        self.biases = []
        w_init = tf.glorot_normal_initializer()
        b_init = tf.zeros_initializer()
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([self.dataset.label_num]):
            weight = tf.get_variable('w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('b%d' % i, [layer], initializer=b_init)
            self.weights.append(weight)
            self.biases.append(bias)
            pre_layer = layer

    def create_model(self, input):
        output = input
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return output

    def create_optimizer(self):
        self.score = self.create_model(self.input)
        self.loss = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10), axis=1))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights + self.biases])
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.loss + self.reg_loss * FLAGS.reg, name='optimizer')

    def create_init(self):
        self.create_placeholder()
        self.create_weights()
        self.create_optimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def get_svd_data(self):
        from sklearn.decomposition import TruncatedSVD
        all_data = np.concatenate(
            [self.dataset.Train_data['x'], self.dataset.Val_data['x'], self.dataset.Test_data['x']])
        svd = TruncatedSVD(n_components=FLAGS.svd_dim)
        data_trans = svd.fit_transform(all_data)
        data_trans = (data_trans - np.mean(data_trans, axis=1, keepdims=True)) / (
                np.std(data_trans, axis=1, keepdims=True) + 1e-8)
        self.dataset.Train_data['x'] = data_trans[:len(self.dataset.Train_data['x'])]
        self.dataset.Val_data['x'] = data_trans[len(self.dataset.Train_data['x']):-len(self.dataset.Test_data['x'])]
        self.dataset.Test_data['x'] = data_trans[-len(self.dataset.Test_data['x']):]
        self.dataset.feature_num = FLAGS.svd_dim

    def train(self, epochs, batch_size):
        if (FLAGS.test == False and FLAGS.defense == 'inf'):
            epochs = 3
        best_val = 0.
        best_test = 0
        best_val_loss = 100000.
        best_test_loss = 100000
        for i in range(epochs):
            batchs = self.get_batches(self.dataset.Train_data, batch_size)
            for cur_x, cur_y in batchs:
                self.sess.run(self.train_op, feed_dict={self.input: cur_x, self.label: cur_y})
            train_acc, train_loss = self.eval_data(self.dataset.Train_data)
            test_acc, test_loss = self.eval_data(self.dataset.Test_data)
            val_acc, val_loss = self.eval_data(self.dataset.Val_data)
            print("epoch %d: " % i, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
            if (best_val <= val_acc):
                best_val = val_acc
                best_test = test_acc
            if (val_loss <= best_val_loss):
                best_val_loss = val_loss
                best_test_loss = test_loss
        if (FLAGS.test == True):
            print('best', best_test)
            try:
                cur_results_acc = np.load("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                    FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt,
                    FLAGS.decrypt,
                    FLAGS.ratio, FLAGS.data_size))
                cur_results_acc = list(cur_results_acc)
                cur_results_loss = np.load("results/loss_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                    FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt,
                    FLAGS.decrypt,
                    FLAGS.ratio, FLAGS.data_size))
                cur_results_loss = list(cur_results_loss)
            except:
                cur_results_acc = []
                cur_results_loss = []
            cur_results_acc.append(best_test)
            cur_results_loss.append(best_test_loss)
            np.save("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio, FLAGS.data_size), cur_results_acc)
            np.save("results/loss_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio, FLAGS.data_size), cur_results_loss)
        # if (FLAGS.test == True and FLAGS.encrypt == 1):
        #     data = self.dataset.Test_data['x']
        #     label = self.dataset.Test_data['y']
        #     pred = self.sess.run(self.score,
        #                          feed_dict={self.input: data, self.label: label})
        #     acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))
        #     print("test acc", acc)
        #     np.save("temp/%s_test_label_%s_%d_%.2f.npy" % (FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.ratio),
        #             pred)

    def get_batches(self, train_data, batch_size):
        x = train_data['x']
        y = train_data['y']
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        data_list = []
        for i in range(len(idx[:-1]) // batch_size + 1):
            cur_idx = idx[i * batch_size:min((i + 1) * batch_size, len(idx))]
            data_list.append([x[cur_idx], y[cur_idx]])
        return data_list

    def eval_data(self, test_data):
        data = test_data['x']
        label = test_data['y']
        pred, loss = self.sess.run([self.score, self.loss],
                                   feed_dict={self.input: data, self.label: label})
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))
        # acc = sklearn.metrics.f1_score(np.argmax(pred, axis=1), np.argmax(label, axis=1))

        return acc, loss
