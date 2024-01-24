import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class BREW:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 1024
        self.create_init()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32)
        self.input_delta = tf.placeholder(dtype=tf.float32)
        self.label = tf.placeholder(dtype=tf.float32)
        self.label_delta = tf.placeholder(dtype=tf.float32)
        self.delta_assign = tf.placeholder(dtype=tf.float32)

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
        self.delta = tf.get_variable('noise', [self.batch_size, self.dataset.feature_num], initializer=w_init)
        self.assign = tf.assign(self.delta, self.delta_assign)

    def create_model(self, input):
        output = input
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return output

    def create_model_delta(self, input):
        self.noise = self.delta
        output = input + self.noise
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return output

    def create_optimizer(self):
        self.score = self.create_model(self.input)
        self.score_delta = self.create_model_delta(self.input_delta)
        self.loss = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10), axis=1))
        self.loss_delta = tf.reduce_mean(tf.reduce_sum(
            -self.label_delta * tf.log(self.score_delta + 1e-10) - (1. - self.label_delta) * tf.log(
                1. - self.score_delta + 1e-10), axis=1))
        self.gradient_delta = tf.gradients(self.loss_delta, self.weights + self.biases)
        self.gradient = tf.gradients(self.loss, self.weights + self.biases)
        gradient = tf.concat([tf.reshape(a, [-1]) for a in self.gradient], axis=0)
        gradient1 = tf.concat([tf.reshape(a, [-1]) for a in self.gradient_delta], axis=0)
        self.poison_loss = 1. - tf.reduce_sum(gradient * gradient1) / tf.sqrt(
            tf.reduce_sum(tf.square(gradient))) / tf.sqrt(tf.reduce_sum(tf.square(gradient1)))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights + self.biases])
        if (FLAGS.dataset == 'app'):
            self.optimizer = tf.train.AdamOptimizer(0.001)
        else:
            self.optimizer = tf.train.AdamOptimizer(0.001)
        self.optimizer_delta = tf.train.AdamOptimizer(0.001)
        self.train_op_delta = self.optimizer_delta.minimize(self.poison_loss, var_list=self.delta)
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

    def train(self, epochs, batch_size):
        import time
        start = time.time()
        best_val = 0.
        best_test = 0
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
        if (FLAGS.test == False):
            self.generate_noise()
        else:
            print('best', best_test)
            np.save("results/%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                    best_test)
        print("time", time.time() - start)

    def get_batches(self, train_data, batch_size):
        # x = train_data['x']
        # y = train_data['y']
        idx = np.arange(train_data['x'].shape[0])
        np.random.shuffle(idx)
        data_list = []
        for i in range(len(idx[:-1]) // batch_size + 1):
            cur_idx = idx[i * batch_size:min((i + 1) * batch_size, len(idx))]
            data_list.append([train_data['x'][cur_idx], train_data['y'][cur_idx]])
        return data_list

    def get_batches_no_shuffle(self, train_data, batch_size):
        data_list = []
        for i in range((train_data['x'].shape[0] - 1) // batch_size + 1):
            data_list.append([train_data['x'][i * batch_size:min((i + 1) * batch_size, train_data['x'].shape[0])],
                              train_data['y'][i * batch_size:min((i + 1) * batch_size, train_data['x'].shape[0])]])
        return data_list

    def eval_data(self, test_data, batch_size=10000):
        # data = test_data['x']
        # label = test_data['y']
        batches = self.get_batches(test_data, batch_size)
        pred_list = []
        loss_list = []
        label_list = []
        for cur_x, cur_y in batches:
            pred, loss = self.sess.run([self.score, self.loss],
                                       feed_dict={self.input: cur_x, self.label: cur_y})
            pred_list.append(pred)
            label_list.append(cur_y)
            loss_list.append(loss)
        pred = np.concatenate(pred_list)
        label = np.concatenate(label_list)
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))
        # acc = sklearn.metrics.f1_score(np.argmax(pred, axis=1), np.argmax(label, axis=1))

        return acc, np.mean(loss_list)

    def generate_noise(self):
        test_x = self.dataset.Test_data['x']
        test_y = self.dataset.Test_data['y']
        test_y = np.concatenate([test_y[:, 1:], test_y[:, :1]], axis=1)
        train_x = self.dataset.Exposed_data['x']
        train_y = self.dataset.Exposed_data['y']
        lens = len(train_x)
        train_x = np.concatenate([train_x, train_x[:(self.batch_size - len(train_x) % self.batch_size)]], axis=0)
        train_y = np.concatenate([train_y, train_y[:(self.batch_size - len(train_y) % self.batch_size)]], axis=0)
        batches = self.get_batches_no_shuffle({'x': train_x, 'y': train_y}, self.batch_size)

        init_delta = self.sess.run(self.delta)

        noise_list = []
        for cur_x, cur_y in batches:
            print('test')
            self.sess.run(self.assign, feed_dict={self.delta_assign: init_delta})
            for i in range(100):
                _, loss, noise = self.sess.run([self.train_op_delta, self.poison_loss, self.noise],
                                               feed_dict={self.input: test_x, self.label: test_y,
                                                          self.input_delta: cur_x,
                                                          self.label_delta: cur_y})
                print(loss, np.max(noise))
            noise = self.sess.run(self.noise)
            noise_list.append(noise)

        for num in range(5,6):
            noise_all = []
            for ii in range(len(batches)):
                noise = noise_list[ii]
                if (FLAGS.dataset == 'weibo'):
                    noise[:, :4002] = -111111
                noise_new = np.zeros_like(noise)
                for i in range(len(noise)):
                    idx = np.argsort(-noise[i])[:num]
                    noise_new[i, idx] = noise[i, idx]
                noise_all.append(noise_new)
            noises = np.concatenate(noise_all)[:lens]
            noises[noises > 0] = 1.
            if (FLAGS.dataset == 'app'):
                noises = np.round(noises * 5) // 5
            elif (FLAGS.dataset == 'weibo'):
                noises[:, 4002:] = noises[:, 4002:] > 0
            else:
                print('error')
                exit(-1)
            # sp.save_npz("temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
            #     FLAGS.dataset, FLAGS.defense, num, FLAGS.encrypt, FLAGS.decrypt),
            #             sp.csr_matrix(noises))
