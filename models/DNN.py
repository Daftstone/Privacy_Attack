import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class DNN:
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
        for i, layer in enumerate([512, 64, self.dataset.label_num]):
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
        self.embedding = output
        output = tf.matmul(output, self.weights[-1]) + self.biases[-1]
        output = tf.nn.softmax(output)
        return output

    def create_optimizer(self):
        self.score = self.create_model(self.input)
        self.loss = tf.reduce_mean(tf.reduce_sum(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10), axis=1))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights + self.biases])
        self.optimizer = tf.train.AdamOptimizer(0.0001)
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
            epochs = 2
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
        if (FLAGS.test == False):
            self.influence_vector(199)
        else:
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
        self.get_embedding()

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

    def eval_data(self, test_data, batch_size=100000):
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
        i_epochs = 2000
        batch_size = 2048
        if (FLAGS.dataset == 'ml-100k'):
            i_epochs = 2000
            batch_size = 1024

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
            for i in range(len(data)):
                # if (i % 1000 == 0):
                #     print(i)
                mean_data = data[i:i + 1]
                mean_label = label[i:i + 1]
                feed_dict = {self.input: mean_data, self.label: mean_label}
                lissa_list = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
                temp = -np.sum(np.concatenate(lissa_list, axis=0), axis=0)
                inf_list.append(temp)
            return np.array(inf_list)

        inf_vec = cal(self.dataset.Exposed_data['x'], self.dataset.Exposed_data['y'])

        if (FLAGS.encrypt):
            print("encrypt")
            train_poison_data, train_poison_label = utils.cal_inf_encrypt(self.dataset.Exposed_data['x'],
                                                                          self.dataset.Exposed_data['y'], inf_vec)
            sp.save_npz("temp/dnn_train_poison_data_%d_%d_%d.npz" % (FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
                        sp.csr_matrix(train_poison_data))
            sp.save_npz("temp/dnn_train_poison_label_%d_%d_%d.npz" % (FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
                        sp.csr_matrix(train_poison_label))
        else:
            print("decrypt")
            train_poison_data, train_poison_label = utils.cal_inf_decrypt(self.dataset.Exposed_data['x'],
                                                                          self.dataset.Exposed_data['y'], inf_vec)
            sp.save_npz(
                "temp/dnn_train_poison_data_%d_%d_%d_%.2f.npz" % (FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                sp.csr_matrix(train_poison_data))
            sp.save_npz(
                "temp/dnn_train_poison_label_%d_%d_%d_%.2f.npz" % (
                    FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                sp.csr_matrix(train_poison_label))

    def get_related(self, train_data, cur_label):
        x = train_data['x']
        y = train_data['y']
        idx = np.where(y[:, cur_label] > 0)[0]
        mean_data = np.mean(x[idx], axis=0, keepdims=True)
        similar = np.sum(x[idx] * mean_data, axis=1) / (
                np.sqrt(np.sum(np.square(x[idx]), axis=1)) * (np.sqrt(np.sum(np.square(mean_data), axis=1))))
        return idx[np.argsort(-similar)[:5]]

    def get_embedding(self):

        batches = self.get_batches(self.dataset.Test_data, 1024)
        embedding_list = []
        label_list = []
        for cur_x, cur_y in batches:
            embedding = self.sess.run(self.embedding, feed_dict={self.input: cur_x})
            embedding_list.append(embedding)
            label_list.append(np.argmax(cur_y, axis=1))
        embedding = np.concatenate(embedding_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        np.save("temp/%s_%d_test_embedding.npy" % (FLAGS.dataset, FLAGS.num), embedding)
        np.save("temp/%s_%d_test_label.npy" % (FLAGS.dataset, FLAGS.num), labels)

        batches = self.get_batches(self.dataset.Train_data, 1024)
        embedding_list = []
        label_list = []
        for cur_x, cur_y in batches:
            embedding = self.sess.run(self.embedding, feed_dict={self.input: cur_x})
            embedding_list.append(embedding)
            label_list.append(np.argmax(cur_y, axis=1))
        embedding = np.concatenate(embedding_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        np.save("temp/%s_%d_train_embedding.npy" % (FLAGS.dataset, FLAGS.num), embedding)
        np.save("temp/%s_%d_train_label.npy" % (FLAGS.dataset, FLAGS.num), labels)
