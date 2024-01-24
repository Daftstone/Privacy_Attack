import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn

import utils

FLAGS = tf.flags.FLAGS


class Distillation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.create_init()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32)
        self.label = tf.placeholder(dtype=tf.float32)
        self.training = tf.placeholder_with_default(False, ())

    def create_weights(self):
        self.weights = []
        self.biases = []
        w_init = tf.glorot_normal_initializer()
        b_init = tf.zeros_initializer()
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([128, self.dataset.label_num]):
            weight = tf.get_variable('w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('b%d' % i, [layer], initializer=b_init)
            self.weights.append(weight)
            self.biases.append(bias)
            pre_layer = layer

    def create_model(self, input):
        output = input
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.tanh(output)
            output = tf.cond(self.training, lambda: tf.nn.dropout(output, 0.2), lambda: output)
        output = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return output

    def create_weights1(self):
        self.weights1 = []
        self.biases1 = []
        w_init = tf.glorot_normal_initializer()
        b_init = tf.zeros_initializer()
        pre_layer = self.dataset.feature_num
        for i, layer in enumerate([128, self.dataset.label_num]):
            weight = tf.get_variable('w1%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('b1%d' % i, [layer], initializer=b_init)
            self.weights1.append(weight)
            self.biases1.append(bias)
            pre_layer = layer

    def create_model1(self, input):
        output = input
        for weight, bias in zip(self.weights1[:-1], self.biases1[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.tanh(output)
            output = tf.cond(self.training, lambda: tf.nn.dropout(output, 0.2), lambda: output)
        output = tf.nn.softmax(tf.matmul(output, self.weights1[-1]) + self.biases1[-1])
        return output

    def create_optimizer(self):
        self.score = self.create_model(self.input)
        self.loss = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10), axis=1))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights + self.biases])
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.loss + self.reg_loss * 1e-4, name='optimizer')

        self.score1 = self.create_model1(self.input)
        self.loss1 = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score1 + 1e-10) - (1. - self.label) * tf.log(1. - self.score1 + 1e-10), axis=1))
        self.reg_loss1 = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights1 + self.biases1])
        self.optimizer1 = tf.train.AdamOptimizer(0.001)
        self.train_op1 = self.optimizer.minimize(self.loss1 + self.reg_loss1 * 1e-4, name='optimizer')

    def create_init(self):
        self.create_placeholder()
        self.create_weights()
        self.create_weights1()
        self.create_optimizer()
        self.build_influence()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, epochs, batch_size):
        best_val = 0
        best_test = 0
        for i in range(10):
            batchs = self.get_batches(self.dataset.Train_data, batch_size)
            for cur_x, cur_y in batchs:
                self.sess.run(self.train_op, feed_dict={self.input: cur_x, self.label: cur_y, self.training: True})
            train_acc, train_loss = self.eval_data(self.dataset.Train_data)
            test_acc, test_loss = self.eval_data(self.dataset.Test_data)
            val_acc, val_loss = self.eval_data(self.dataset.Val_data)
            print("epoch %d: " % i, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        train_data = self.dataset.Train_data['x']
        train_label = self.sess.run(self.score, feed_dict={self.input: train_data})
        for i in range(epochs):
            batchs = self.get_batches({'x': train_data, 'y': train_label}, batch_size)
            for cur_x, cur_y in batchs:
                self.sess.run(self.train_op1, feed_dict={self.input: cur_x, self.label: cur_y, self.training: True})
            train_acc, train_loss = self.eval_data1(self.dataset.Train_data)
            test_acc, test_loss = self.eval_data1(self.dataset.Test_data)
            val_acc, val_loss = self.eval_data1(self.dataset.Val_data)
            print("epoch %d: " % i, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
            if (best_val <= val_acc):
                best_val = val_acc
                best_test = test_acc
        print('best', best_test)

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
        return acc, loss

    def eval_data1(self, test_data):
        data = test_data['x']
        label = test_data['y']
        pred, loss = self.sess.run([self.score1, self.loss1],
                                   feed_dict={self.input: data, self.label: label})
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))
        return acc, loss

    def build_influence(self):
        with tf.variable_scope('influence'):
            self.params = self.weights + self.biases
            self.attack_loss = tf.reduce_sum(self.loss)

            self.scale = 100.

            dty = tf.float32
            self.v_cur_est = [tf.placeholder(dty, shape=a.get_shape(), name="v_cur_est" + str(i)) for i, a in
                              enumerate(self.params)]
            self.Test = [tf.placeholder(dty, shape=a.get_shape(), name="test" + str(i)) for i, a in
                         enumerate(self.params)]

            hessian_vector_val = utils.hessian_vector_product(self.loss, self.params, self.v_cur_est, True, 10000000.)
            self.estimation_IHVP = [g + cur_e - HV
                                    for g, HV, cur_e in zip(self.Test, hessian_vector_val, self.v_cur_est)]

            self.attack_grad = tf.gradients(self.loss, self.params)
            self.per_loss = self.loss

    def influence_vector(self):
        import time
        self.build_influence()
        i_epochs = 10000
        batch_size = 1024

        # IHVP
        start_time = time.time()
        optimized_data = self.dataset.Train_data

        feed_dict = {self.input: optimized_data['x'], self.label: optimized_data['y']}
        test_val = self.sess.run(self.attack_grad, feed_dict)

        cur_estimate = test_val.copy()
        feed1 = {place: cur for place, cur in zip(self.Test, test_val)}
        for j in range(i_epochs):
            feed2 = {place: cur for place, cur in zip(self.v_cur_est, cur_estimate)}
            idx = np.random.choice(np.arange(len(self.dataset.Train_data['x'])), batch_size)
            feed_dict = {self.input: self.dataset.Train_data['x'][idx], self.label: self.dataset.Train_data['y'][idx]}
            cur_estimate = self.sess.run(self.estimation_IHVP, feed_dict={**feed_dict, **feed1, **feed2})
            if (j % 1000 == 0):
                print(np.max(cur_estimate[0][0]))
        inverse_hvp1 = [b / self.scale for b in cur_estimate]
        print(np.max(inverse_hvp1[0]))
        duration = time.time() - start_time
        print('Inverse HVP by HVPs+Lissa: took %s minute %s sec' % (duration // 60, duration % 60))

        for i in range(25):
            val_lissa = []
            idx, mean_data = self.get_related(self.dataset.Train_data, i)
            # mean_data = self.dataset.Train_data['x'][idx:idx + 1]
            mean_label = self.dataset.Train_data['y'][idx:idx + 1]
            feed_dict = {self.input: mean_data, self.label: mean_label}
            feed2 = {place: cur for place, cur in zip(self.v_cur_est, inverse_hvp1)}
            pert_vector_val = utils.pert_vector_product(self.per_loss, self.params, self.input,
                                                        self.v_cur_est, True)

            lissa_list = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
            temp = -np.sum(np.concatenate(lissa_list, axis=0), axis=0)
            val_lissa.append(temp)
            np.save("temp/influence_%d.npy" % i, temp)
            np.save("temp/user_%d.npy" % i, mean_data[0])
        return temp, mean_data[0]

    def get_related(self, train_data, cur_label):
        x = train_data['x']
        y = train_data['y']
        idx = np.where(y[:, cur_label] > 0)[0]
        mean_data = np.mean(x[idx], axis=0, keepdims=True)
        similar = np.sum(x[idx] * mean_data, axis=1) / (
                np.sqrt(np.sum(np.square(x[idx]), axis=1)) * (np.sqrt(np.sum(np.square(mean_data), axis=1))))
        return idx[np.argmax(similar)], mean_data
