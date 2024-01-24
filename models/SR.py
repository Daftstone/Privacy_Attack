import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

import utils

from parse import FLAGS


class SR:
    def __init__(self, dataset):
        self.dataset = dataset

    def train(self, epochs, batch_size):

        train_data = self.dataset.Exposed_data['x']
        train_label = self.dataset.Exposed_data['y']

        self.sr = []
        for i in range(train_data.shape[1]):
            if (i % 1000 == 0):
                print(i)
            temp = []
            cur_a = train_data[:, i] > 0
            # threshold = np.median(train_data[:, i])
            # if (threshold == 1):
            #     cur_a = train_data[:, i] >= threshold
            # else:
            #     cur_a = train_data[:, i] > threshold
            for j in range(train_label.shape[1]):
                c00 = np.sum((cur_a == 0) * (train_label[:, j] == 0))
                c11 = np.sum((cur_a == 1) * (train_label[:, j] == 1))
                c01 = np.sum((cur_a == 0) * (train_label[:, j] == 1))
                c10 = np.sum((cur_a == 1) * (train_label[:, j] == 0))
                temp.append(1 - 4 * (c00 + c11) * (c10 + c01) / len(cur_a) / len(cur_a))
            self.sr.append(np.max(temp))
        self.sr = np.array(self.sr)[None, :]
        self.generate_noise()

    def generate_noise(self):
        test_data = self.dataset.Test_data['x']
        sr = self.sr - (test_data > 0) * 11111111
        noise = np.zeros_like(test_data)
        for i in range(len(test_data)):
            idx = np.argsort(-sr[i])[:FLAGS.num]
            noise[i, idx] = 1
        sp.save_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
            FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
                    sp.csr_matrix(noise))

        test_data = self.dataset.Exposed_data['x']
        sr = self.sr - (test_data > 0) * 11111111
        noise = np.zeros_like(test_data)
        for i in range(len(test_data)):
            idx = np.argsort(-sr[i])[:FLAGS.num]
            noise[i, idx] = 1
        sp.save_npz("temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
            FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt),
                    sp.csr_matrix(noise))
