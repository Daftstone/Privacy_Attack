import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class AttriGuard:
    def __init__(self, dataset):
        self.dataset = dataset
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
        if (FLAGS.dataset == 'app'):
            self.optimizer = tf.train.AdamOptimizer(0.001)
        else:
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

    def train(self, epochs, batch_size):
        if (FLAGS.test == False and FLAGS.defense == 'inf'):
            epochs = 1
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
            for i in range(5, 6):
                FLAGS.num = i
                self.generate_noise()
        else:
            print('best', best_test)
            np.save("results/%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                    best_test)
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
        from tensorflow.python.ops.parallel_for.gradients import jacobian
        gradient = jacobian(self.score, self.input)
        batches = self.get_batches_no_shuffle(self.dataset.Test_data, 1)

        prior = np.sum(self.dataset.Test_data['y'], axis=0) / np.sum(self.dataset.Test_data['y'])

        noise_list = []
        for iter, (cur_x, cur_y) in enumerate(batches):
            if (iter % 100 == 0):
                print(iter, len(batches))
            # pred = self.sess.run(self.score, feed_dict={self.input: cur_x, self.label: cur_y})
            # pred = pred - cur_y * 100
            # cur_t = np.argmax(pred, axis=1)
            cur_t = np.random.choice(np.arange(cur_y.shape[1]), FLAGS.num, p=prior)
            cur_noise = cur_x.copy()
            for i in range(FLAGS.num):
                mask = cur_noise > 0
                cur_grad = self.sess.run(gradient, feed_dict={self.input: cur_noise, self.label: cur_y})[0].transpose(
                    (1, 2, 0))
                if (FLAGS.dataset == 'weibo'):
                    cur_grad[:, :4002] = -1111111
                for ii in range(len(cur_noise)):
                    idx = np.argmax(cur_grad[ii, :, cur_t[ii]] + mask[ii, :] * -99999)
                    cur_noise[ii, idx] = 1
            noise_list.append(cur_noise - cur_x)
        noises = np.concatenate(noise_list)
        if (FLAGS.dataset == 'app'):
            noises = np.round(noises * 5) // 5
        elif (FLAGS.dataset == 'weibo'):
            noises[:, 4002:] = noises[:, 4002:] > 0
        else:
            print('error')
            exit(-1)
        sp.save_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
            FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
                    sp.csr_matrix(noises))
