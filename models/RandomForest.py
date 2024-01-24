import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from parse import FLAGS


class RandomForest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_features = self.dataset.feature_num
        self.clf = RandomForestClassifier(n_estimators=5)

    def train(self, epochs, batch_size):
        train_data = self.dataset.Train_data['x']
        train_label = self.dataset.Train_data['y']
        self.clf.fit(train_data, train_label)
        print("train done!")
        train_acc = self.eval_data(self.dataset.Train_data)
        test_acc = self.eval_data(self.dataset.Test_data)
        val_acc = self.eval_data(self.dataset.Val_data)
        print("acc", train_acc, val_acc, test_acc)

        try:
            cur_results_acc = np.load("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt,
                FLAGS.decrypt,
                FLAGS.ratio, FLAGS.data_size))
            cur_results_acc = list(cur_results_acc)
        except:
            cur_results_acc = []
        cur_results_acc.append(test_acc)
        np.save("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f_%.2f.npy" % (
            FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
            FLAGS.ratio, FLAGS.data_size), cur_results_acc)

    def eval_data(self, test_data):
        data = test_data['x']
        label = test_data['y']
        preds = self.clf.predict(data)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(label, axis=1))
        return acc
