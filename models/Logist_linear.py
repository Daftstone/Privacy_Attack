import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class Logist:
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
            self.influence_vector(199)
        else:
            print('best', best_test)
            if (FLAGS.data_size == 1):
                np.save("results/%s_%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                    FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                    FLAGS.ratio), best_test)
            else:
                np.save("results/%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                    FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                    FLAGS.ratio, FLAGS.data_size), best_test)
        # if (FLAGS.test == True and FLAGS.encrypt == 1):
        #     data = self.dataset.Test_data['x']
        #     label = self.dataset.Test_data['y']
        #     pred = self.sess.run(self.score,
        #                          feed_dict={self.input: data, self.label: label})
        #     acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))
        #     print("test acc", acc)
        #     if (FLAGS.data_size == 1):
        #         np.save("temp/%s_test_label_%s_%d_%.2f.npy" % (FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.ratio),
        #                 pred)

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

            self.attack_grad = tf.gradients(self.attack_loss, self.params)
            self.per_loss = self.loss

    def influence_vector(self, flag=0):
        import time
        self.build_influence()
        i_epochs = 8000
        batch_size = 512

        # IHVP
        start_time = time.time()
        optimized_data = self.dataset.Test_data
        print(len(optimized_data['x']))
        data = optimized_data['x']
        label = optimized_data['y']
        feed_dict = {self.input: data,
                     self.label: label}
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

        feed2 = {place: cur for place, cur in zip(self.v_cur_est, inverse_hvp1)}
        pert_vector_val = utils.pert_vector_product(self.per_loss, self.params, self.input,
                                                    self.v_cur_est, True)

        def cal(data, label):
            inf_list = []
            start = time.time()
            for i in range(len(data)):
                if (i % 10000 == 0):
                    print(i, time.time() - start)
                    start = time.time()
                mean_data = data[i:i + 1]
                mean_label = label[i:i + 1]
                feed_dict = {self.input: mean_data, self.label: mean_label}
                lissa_list = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
                temp = -np.sum(np.concatenate(lissa_list, axis=0), axis=0)
                inf_list.append(temp)
            return np.array(inf_list)

        inf_vec = cal(self.dataset.Exposed_data['x'][-self.dataset.label_num:],
                      self.dataset.Exposed_data['y'][-self.dataset.label_num:])

        if (FLAGS.encrypt):
            print("encrypt")
            for num in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]:
                begin = {'app': 0, 'weibo': 4002}
                train_poison_data, train_poison_label = utils.cal_inf_encrypt_mean(
                    self.dataset.Exposed_data['x'][:-self.dataset.label_num][:, begin[FLAGS.dataset]:],
                    self.dataset.Exposed_data['y'][:-self.dataset.label_num], inf_vec[:, begin[FLAGS.dataset]:], num)
                if (begin[FLAGS.dataset] > 0):
                    train_poison_data = np.concatenate(
                        [np.zeros((len(train_poison_data), begin[FLAGS.dataset])), train_poison_data], axis=1)
                sp.save_npz("temp/%s_logist_train_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, num, FLAGS.encrypt, FLAGS.decrypt),
                            sp.csr_matrix(train_poison_data))
                sp.save_npz("temp/%s_logist_train_poison_label_%d_%d_%d.npz" % (
                    FLAGS.dataset, num, FLAGS.encrypt, FLAGS.decrypt),
                            sp.csr_matrix(train_poison_label))
            for num in [1, 2, 3, 4, 5, 6, 7, 8]:
                train_poison_data = sp.load_npz(
                    "temp/%s_logist_train_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, num * 2, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                train_poison_data = utils.step2(train_poison_data, FLAGS.dataset, num)
                sp.save_npz("temp/%s_logist3_train_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, num, FLAGS.encrypt, FLAGS.decrypt),
                            sp.csr_matrix(train_poison_data))
                sp.save_npz("temp/%s_logist3_train_poison_label_%d_%d_%d.npz" % (
                    FLAGS.dataset, num, FLAGS.encrypt, FLAGS.decrypt),
                            sp.csr_matrix(train_poison_label))
        else:
            print("decrypt")
            train_poison_data, train_poison_label = utils.cal_inf_decrypt_mean(
                self.dataset.Exposed_data['x'][:-self.dataset.label_num],
                self.dataset.Exposed_data['y'][:-self.dataset.label_num], inf_vec)
            sp.save_npz(
                "temp/%s_%s_train_poison_data_%d_%d_%d_%.2f.npz" % (
                    FLAGS.dataset, FLAGS.surrogate, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                sp.csr_matrix(train_poison_data))
            sp.save_npz(
                "temp/%s_%s_train_poison_label_%d_%d_%d_%.2f.npz" % (
                    FLAGS.dataset, FLAGS.surrogate, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                sp.csr_matrix(train_poison_label))

    def get_related(self, train_data, cur_label):
        x = train_data['x']
        y = train_data['y']
        idx = np.where(y[:, cur_label] > 0)[0]
        mean_data = np.mean(x[idx], axis=0, keepdims=True)
        similar = np.sum(x[idx] * mean_data, axis=1) / (
                np.sqrt(np.sum(np.square(x[idx]), axis=1)) * (np.sqrt(np.sum(np.square(mean_data), axis=1))))
        return idx[np.argsort(-similar)[:5]]
