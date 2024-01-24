import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class ARL:
    def __init__(self, dataset):
        self.dataset = dataset
        self.create_init()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32)
        self.label = tf.placeholder(dtype=tf.float32)
        self.noise = tf.placeholder(dtype=tf.float32)

    def create_weights(self):
        self.a_weights = []
        self.a_biases = []
        self.d_weights = []
        self.d_biases = []
        self.e_weights = []
        self.e_biases = []
        w_init = tf.glorot_normal_initializer()
        b_init = tf.zeros_initializer()
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([self.dataset.label_num]):
            weight = tf.get_variable('a_w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('a_b%d' % i, [layer], initializer=b_init)
            self.a_weights.append(weight)
            self.a_biases.append(bias)
            pre_layer = layer
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([64, 32, 64, self.dataset.feature_num]):
            weight = tf.get_variable('d_w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('d_b%d' % i, [layer], initializer=b_init)
            self.d_weights.append(weight)
            self.d_biases.append(bias)
            pre_layer = layer
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([64, 32, 64, self.dataset.feature_num]):
            weight = tf.get_variable('e_w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('e_b%d' % i, [layer], initializer=b_init)
            self.e_weights.append(weight)
            self.e_biases.append(bias)
            pre_layer = layer

    def create_a_model(self, input):
        output = input
        for weight, bias in zip(self.a_weights[:-1], self.a_biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.softmax(tf.matmul(output, self.a_weights[-1]) + self.a_biases[-1])
        return output

    def create_d_model(self, input):
        output = input
        for weight, bias in zip(self.d_weights[:2], self.d_biases[:2]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = output + self.noise
        for weight, bias in zip(self.d_weights[2:-1], self.d_biases[2:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.sigmoid(tf.matmul(output, self.d_weights[-1]) + self.d_biases[-1])
        return output

    def create_e_model(self, input):
        output = input
        for weight, bias in zip(self.e_weights[:-1], self.e_biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.nn.sigmoid(tf.matmul(output, self.e_weights[-1]) + self.e_biases[-1])
        return output

    def create_optimizer(self):
        self.adv_input = self.create_e_model(self.input)
        self.score = self.create_a_model(self.adv_input)
        self.res_input = self.create_d_model(self.adv_input)
        self.a_loss = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10), axis=1))
        self.d_loss = tf.reduce_mean(tf.square(self.res_input - self.input))
        self.e_loss = tf.reduce_mean(tf.square(self.adv_input - self.input))
        self.reg_a_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.a_weights + self.a_biases])
        self.reg_d_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.d_weights + self.d_biases])
        self.reg_e_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.e_weights + self.e_biases])
        # self.optimizer1 = tf.train.AdamOptimizer(0.0001)
        # self.optimizer2 = tf.train.AdamOptimizer(0.0002)
        self.optimizer3 = tf.train.AdamOptimizer(0.0001)
        self.step1_loss = self.d_loss
        self.step2_loss = self.a_loss
        self.step3_loss = self.d_loss - self.a_loss
        self.step1_op = self.optimizer3.minimize(self.step1_loss, name='optimizer1',
                                                 var_list=self.d_weights + self.d_biases)
        self.step2_op = self.optimizer3.minimize(self.step2_loss, name='optimizer2',
                                                 var_list=self.a_weights + self.a_biases)
        self.step3_op = self.optimizer3.minimize(self.step3_loss, name='optimizer3',
                                                 var_list=self.e_weights + self.e_biases)

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
        self.std = 0.001

        for i in range(epochs):
            batchs = self.get_batches(self.dataset.Train_data, batch_size)
            step1_list = []
            step2_list = []
            step3_list = []
            for cur_x, cur_y in batchs:
                for e in range(1):
                    noise = np.random.normal(loc=0., scale=self.std, size=(len(cur_x), 32))
                    step1, _ = self.sess.run([self.step1_loss, self.step1_op],
                                             feed_dict={self.input: cur_x, self.label: cur_y, self.noise: noise})
                for e in range(1):
                    noise = np.random.normal(loc=0., scale=self.std, size=(len(cur_x), 32))
                    step2, _ = self.sess.run([self.step2_loss, self.step2_op],
                                             feed_dict={self.input: cur_x, self.label: cur_y, self.noise: noise})

                # cur_y = np.ones_like(cur_y) * 1 / (cur_y.shape[1])
                for e in range(1):
                    noise = np.random.normal(loc=0., scale=self.std, size=(len(cur_x), 32))
                    step3, _ = self.sess.run([self.step3_loss, self.step3_op],
                                             feed_dict={self.input: cur_x, self.label: cur_y, self.noise: noise})
                step1_list.append(step1)
                step2_list.append(step2)
                step3_list.append(step3)
            print('epochs %d:' % i, np.mean(step1_list), np.mean(step2_list), np.mean(step3_list))
        for i in range(5, 6):
            FLAGS.num = i
            self.generate_noise()
        # self.generate_noise()
        print("time", time.time() - start)

    def generate_noise(self):
        batches = self.get_batches_no_shuffle(self.dataset.Test_data, 64)
        noise_list = []
        for cur_x, cur_y in batches:
            print('test')
            mnoise = np.random.normal(loc=0., scale=self.std, size=(len(cur_x), 32))
            cur_noise = self.sess.run(self.adv_input, feed_dict={self.input: cur_x, self.noise: mnoise})
            # np.save("temp.npy",cur_noise)
            # exit(0)

            noise = cur_noise - cur_x
            if (FLAGS.dataset == 'weibo'):
                noise[:, :4002] = -1111111
            noise_new = np.zeros_like(noise)
            for i in range(len(noise)):
                idx = np.argsort(-noise[i])[:FLAGS.num]
                noise_new[i, idx] = 1
            if (FLAGS.dataset == 'app'):
                noise_new = np.round(noise_new * 5) // 5
            elif (FLAGS.dataset == 'weibo'):
                noise_new[:, 4002:] = noise_new[:, 4002:] > 0
            else:
                print('error')
                exit(-1)
            noise_list.append(noise_new)
            # break
        noises = np.concatenate(noise_list)
        # sp.save_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
        #     FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
        #             sp.csr_matrix(noises))

        batches = self.get_batches_no_shuffle(self.dataset.Exposed_data, 64)
        noise_list = []
        for cur_x, cur_y in batches:
            print('test')
            mnoise = np.random.normal(loc=0., scale=self.std, size=(len(cur_x), 32))
            cur_noise = self.sess.run(self.adv_input, feed_dict={self.input: cur_x, self.noise: mnoise})
            # np.save("temp.npy",cur_noise)
            # exit(0)

            noise = cur_noise - cur_x
            noise = noise - (cur_x > 0) * 11111111
            if (FLAGS.dataset == 'weibo'):
                noise[:, :4002] = -1111111
            noise_new = np.zeros_like(noise)
            for i in range(len(noise)):
                idx = np.argsort(-noise[i])[:FLAGS.num]
                noise_new[i, idx] = 1
            if (FLAGS.dataset == 'app'):
                noise_new = np.round(noise_new * 5) // 5
            elif (FLAGS.dataset == 'weibo'):
                noise_new[:, 4002:] = noise_new[:, 4002:] > 0
            else:
                print('error')
                exit(-1)
            noise_list.append(noise_new)
            # break
        noises = np.concatenate(noise_list)
        # sp.save_npz("temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
        #     FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
        #             sp.csr_matrix(noises))

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
        for i in range(train_data['x'].shape[0] // batch_size + 1):
            data_list.append([train_data['x'][i * batch_size:min((i + 1) * batch_size, train_data['x'].shape[0])],
                              train_data['y'][i * batch_size:min((i + 1) * batch_size, train_data['x'].shape[0])]])
        return data_list
