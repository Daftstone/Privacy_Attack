import numpy as np
import scipy.sparse as sp
import math

import utils

from parse import FLAGS


class Dataset:
    def __init__(self):
        if (FLAGS.dataset == 'app'):
            all_data, all_label = self.read_app_data()
            print('test',len(all_data))
            self.feature_num = len(all_data[0])
            self.label_num = len(all_label[0])
            np.random.seed(100)
            idx = np.random.permutation(len(all_data))
            all_data = all_data[idx]
            all_label = all_label[idx]
            train_ratio = 0.7
            val_ratio = 0.8
        elif (FLAGS.dataset == 'weibo'):
            all_data, all_label = self.read_weibo_data()
            print('test',len(all_data))
            self.secret_label = all_label.copy()
            self.feature_num = all_data.shape[1]
            self.label_num = len(all_label[0])
            np.random.seed(100)
            idx = np.random.permutation(all_data.shape[0])
            all_data = all_data[idx]
            all_label = all_label[idx]
            train_ratio = 0.7
            val_ratio = 0.8
            self.train_idx = idx[:int(len(all_label) * train_ratio)]
            self.val_idx = idx[int(len(all_label) * train_ratio):int(len(all_label) * val_ratio)]
            self.test_idx = idx[int(len(all_label) * val_ratio):]
            self.idx = idx

        else:
            print('error')
            exit(0)

        self.Train_data = {'x': all_data[:int(len(all_label) * train_ratio)],
                           'y': all_label[:int(len(all_label) * train_ratio)]}
        self.Val_data = {'x': all_data[int(len(all_label) * train_ratio):int(len(all_label) * val_ratio)],
                         'y': all_label[int(len(all_label) * train_ratio):int(len(all_label) * val_ratio)]}
        self.Test_data = {'x': all_data[int(len(all_label) * val_ratio):],
                          'y': all_label[int(len(all_label) * val_ratio):]}
        self.Exposed_data = {'x': all_data[:int(len(all_label) * val_ratio)],
                             'y': all_label[:int(len(all_label) * val_ratio)]}

        random_idx = np.random.permutation(len(self.Exposed_data['y']))
        select_idx = random_idx[:int(len(random_idx) * FLAGS.ratio)]
        non_select_idx = random_idx[int(len(random_idx) * FLAGS.ratio):]
        non_select_idx = non_select_idx[:int(len(non_select_idx) * FLAGS.data_size)]
        print(len(random_idx))

        if (FLAGS.test):
            print('test %s' % FLAGS.defense)
            if (FLAGS.num == 0):
                pass
            elif (FLAGS.defense == 'mask'):
                poison_data = np.load("temp/test_poison_data_%s_%s_%d.npy" % (FLAGS.dataset, FLAGS.defense, FLAGS.num))
                self.Test_data['x'] = np.clip(self.Test_data['x'] + poison_data, 0, 1)
            elif (FLAGS.defense == 'guard' or FLAGS.defense == "pga"):
                print(FLAGS.defense)
                poison_data = sp.load_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.sum(poison_data) / (FLAGS.num * len(poison_data)))
                self.Test_data['x'] = self.Test_data['x'] + poison_data
                assert np.max(self.Test_data['x']) <= 1
            elif (FLAGS.defense == 'maxent' or FLAGS.defense == 'ppco' or FLAGS.defense == 'arl'):
                print(FLAGS.defense)
                poison_data = sp.load_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.sum(poison_data) / (FLAGS.num * len(poison_data)))
                self.Test_data['x'] = self.Test_data['x'] + poison_data
                assert np.max(self.Test_data['x']) <= 1

                train_poison_data_full = sp.load_npz("temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.mean(np.sum(train_poison_data_full > 0, axis=1)))
                train_poison_data = np.zeros_like(train_poison_data_full)
                train_poison_data[select_idx] = train_poison_data_full[select_idx]

                if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                    self.modify_graph_data_all(train_poison_data, poison_data)
                self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                     'y': self.Exposed_data['y']}

            elif (('pga' in FLAGS.defense or 'guard' in FLAGS.defense) and (
                    'poigen' in FLAGS.defense or 'brew' in FLAGS.defense or 'inf' in FLAGS.defense)):
                if ('poigen' in FLAGS.defense):
                    defense = 'poigen'
                elif ('brew' in FLAGS.defense):
                    defense = 'brew'
                elif ('inf' in FLAGS.defense):
                    defense = 'inf'
                else:
                    print('error')
                    exit(-1)
                if (defense == 'inf'):
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, 5, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                else:
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                            FLAGS.dataset, defense, 5, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.mean(np.sum(train_poison_data_full > 0, axis=1)))
                train_poison_data = np.zeros_like(train_poison_data_full)
                train_poison_data[select_idx] = train_poison_data_full[select_idx]

                if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                    self.modify_graph_data(train_poison_data)
                self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                     'y': self.Exposed_data['y']}

                poison_data = sp.load_npz("temp/%s_%s_test_poison_data_%d_%d_%d.npz" % (
                    FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.sum(poison_data) / (FLAGS.num * len(poison_data)))
                self.Test_data['x'] = self.Test_data['x'] + poison_data
                assert np.max(self.Test_data['x']) <= 1

            elif (FLAGS.defense == 'poigen' or FLAGS.defense == 'brew'):
                train_poison_data_full = sp.load_npz(
                    "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                        FLAGS.dataset, FLAGS.defense, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.mean(np.sum(train_poison_data_full > 0, axis=1)))
                train_poison_data = np.zeros_like(train_poison_data_full)
                train_poison_data[select_idx] = train_poison_data_full[select_idx]

                if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                    self.modify_graph_data(train_poison_data)
                self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                     'y': self.Exposed_data['y']}
            elif (FLAGS.defense == 'inf'):
                if (FLAGS.encrypt):
                    print("test encrypt!!!!")
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                    print(np.mean(np.sum(train_poison_data_full > 0, axis=1)))
                    train_poison_data = np.zeros_like(train_poison_data_full)
                    train_poison_data[select_idx] = train_poison_data_full[select_idx]

                    if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                        self.modify_graph_data(train_poison_data)
                    self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                         'y': self.Exposed_data['y']}
                elif (FLAGS.decrypt != 0):
                    print("test decrypt!!!!")
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_1_0.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, FLAGS.num)).toarray()
                    train_poison_data_full1 = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d_%.2f.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                            FLAGS.ratio)).toarray()
                    train_poison_data_full = train_poison_data_full + train_poison_data_full1
                    train_poison_data = np.zeros_like(train_poison_data_full)
                    train_poison_data[select_idx] = train_poison_data_full[select_idx]

                    if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                        self.modify_graph_data(train_poison_data)
                    self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                         'y': self.Exposed_data['y']}
                else:
                    print('error')
                    exit(0)
                if (FLAGS.data_size < 1.):
                    print('data size')
                    print(len(self.Exposed_data['x']))
                    idxs = np.concatenate([select_idx, non_select_idx])
                    self.Exposed_data = {'x': self.Exposed_data['x'][idxs], 'y': self.Exposed_data['y'][idxs]}
                    print(len(self.Exposed_data['x']))
            elif (FLAGS.defense == 'none'):
                pass
            else:
                print('error')
                exit(0)
            self.Val_data = {'x': self.Exposed_data['x'][-len(self.Val_data['y']):],
                             'y': self.Exposed_data['y'][-len(self.Val_data['y']):]}
            self.Train_data = {'x': self.Exposed_data['x'][:-len(self.Val_data['y'])],
                               'y': self.Exposed_data['y'][:-len(self.Val_data['y'])]}
            print(len(self.Val_data['y']), len(self.Train_data['y']))

        else:
            if (FLAGS.num == 0):
                exit(0)
            elif (FLAGS.defense == 'inf'):
                if (FLAGS.encrypt):
                    print("encrypt!!!!")
                    if ('logist' in FLAGS.surrogate):
                        mean_x, mean_y = utils.cal_mean_data(self.Exposed_data)
                        self.Exposed_data['x'] = np.concatenate([self.Exposed_data['x'], mean_x])
                        self.Exposed_data['y'] = np.concatenate([self.Exposed_data['y'], mean_y])
                    self.Train_data = self.Exposed_data
                    self.Val_data = self.Exposed_data

                elif (FLAGS.decrypt == 1):
                    print("decrypt!!!!")
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_1_0.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, FLAGS.num)).toarray()
                    train_poison_data = np.zeros_like(train_poison_data_full)
                    train_poison_data[select_idx] = train_poison_data_full[select_idx]
                    self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                         'y': self.Exposed_data['y']}
                    if ('logist' in FLAGS.surrogate):
                        mean_x, mean_y = utils.cal_mean_data(self.Exposed_data)
                        self.Exposed_data['x'] = np.concatenate([self.Exposed_data['x'], mean_x])
                        self.Exposed_data['y'] = np.concatenate([self.Exposed_data['y'], mean_y])
                    self.Train_data = self.Exposed_data
                    self.Val_data = self.Exposed_data
                elif (FLAGS.decrypt == 2):
                    print("decrypt!!!!")
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_1_0.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, FLAGS.num)).toarray()
                    train_poison_data = np.zeros_like(train_poison_data_full)
                    train_poison_data[select_idx] = train_poison_data_full[select_idx]
                    self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                         'y': self.Exposed_data['y']}
                    labels = np.load(
                        "temp/%s_test_label_%s_%d_%.2f.npy" % (
                            FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.ratio))
                    self.Test_data['y'] = labels
                    if ('logist' in FLAGS.surrogate):
                        mean_x, mean_y = utils.cal_mean_data(self.Exposed_data)
                        self.Exposed_data['x'] = np.concatenate([self.Exposed_data['x'], mean_x])
                        self.Exposed_data['y'] = np.concatenate([self.Exposed_data['y'], mean_y])
                    self.Train_data = self.Exposed_data
                    self.Val_data = self.Exposed_data
                elif (FLAGS.decrypt == 0 and FLAGS.encrypt == False):
                    print("error")
                    exit(0)
            elif (
                    FLAGS.defense == 'pga' or FLAGS.defense == 'guard' or FLAGS.defense == 'maxent' or FLAGS.defense == 'poigen' or FLAGS.defense == 'brew' or FLAGS.defense == 'ppco' or FLAGS.defense == 'arl'):
                pass
            elif (('pga' in FLAGS.defense or 'guard' in FLAGS.defense) and (
                    'poigen' in FLAGS.defense or 'brew' in FLAGS.defense or 'inf' in FLAGS.defense)):
                if ('poigen' in FLAGS.defense):
                    defense = 'poigen'
                elif ('brew' in FLAGS.defense):
                    defense = 'brew'
                elif ('inf' in FLAGS.defense):
                    defense = 'inf'
                else:
                    print('error')
                    exit(-1)
                if (defense == 'inf'):
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                            FLAGS.dataset, FLAGS.surrogate, 5, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                else:
                    train_poison_data_full = sp.load_npz(
                        "temp/%s_%s_train_poison_data_%d_%d_%d.npz" % (
                            FLAGS.dataset, defense, 5, FLAGS.encrypt, FLAGS.decrypt)).toarray()
                print(np.mean(np.sum(train_poison_data_full > 0, axis=1)))
                train_poison_data = np.zeros_like(train_poison_data_full)
                train_poison_data[select_idx] = train_poison_data_full[select_idx]

                if ('gcn' in FLAGS.model or 'gat' in FLAGS.model):
                    self.modify_graph_data(train_poison_data)
                self.Exposed_data = {'x': np.clip(self.Exposed_data['x'] + train_poison_data, 0, 1),
                                     'y': self.Exposed_data['y']}
                self.Val_data = {'x': self.Exposed_data['x'][-len(self.Val_data['y']):],
                                 'y': self.Exposed_data['y'][-len(self.Val_data['y']):]}
                self.Train_data = {'x': self.Exposed_data['x'][:-len(self.Val_data['y'])],
                                   'y': self.Exposed_data['y'][:-len(self.Val_data['y'])]}

            else:
                print('error')
                exit(0)

        if (FLAGS.dataset == 'app'):
            self.get_rec_data()

    def read_app_data(self):
        with open("data/%s/test_user.txt" % (FLAGS.dataset), 'r') as f:
            with open("data/%s/train_user.txt" % (FLAGS.dataset), 'r') as f1:
                lines = f.readlines()
                lines1 = f1.readlines()
                lines = lines + lines1
                users = np.zeros((len(lines), 10000))
                for i, line in enumerate(lines):
                    data = line.split(" ")
                    for cur_data in data:
                        item, value = cur_data.split(":")
                        users[i, int(item) - 1] = float(value)
        with open("data/%s/test_label.txt" % (FLAGS.dataset), 'r') as f:
            with open("data/%s/train_label.txt" % (FLAGS.dataset), 'r') as f1:
                lines = f.readlines()
                lines1 = f1.readlines()
                lines = lines + lines1
                labels = np.zeros((len(lines), 25))
                for i, line in enumerate(lines):
                    labels[i, int(line)] = 1.
        assert len(users) == len(labels)
        return users, labels

    def read_weibo_data(self):
        import scipy.sparse as sp
        users = sp.load_npz("data/weibo/filter_feature.npz").toarray()
        self.feature = users
        self.adj = sp.load_npz("data/weibo/filter_adj.npz")
        features = self.adj.toarray()
        users = np.concatenate([users, features], axis=1)
        labels = np.load('data/weibo/filter_secret_label.npy')

        all_data = np.load("data/weibo/filter_all_data.npy")
        all_label = np.load("data/weibo/filter_all_label.npy")

        np.random.seed(100)
        idx0 = np.where(all_label == 0)[0]
        idx1 = np.where(all_label == 1)[0]
        idx0 = np.random.choice(idx0, len(idx1), False)
        all_data = np.concatenate([all_data[idx0], all_data[idx1]])
        all_label = np.concatenate([all_label[idx0], all_label[idx1]])

        idx = np.random.permutation(len(all_data))
        all_data = all_data[idx]
        all_label = all_label[idx]

        self.all_train_data = all_data[:int(len(all_data) * 0.7)]
        self.all_train_label = all_label[:int(len(all_data) * 0.7)]
        self.all_val_data = all_data[int(len(all_data) * 0.7):int(len(all_data) * 0.8)]
        self.all_val_label = all_label[int(len(all_data) * 0.7):int(len(all_data) * 0.8)]
        self.all_test_data = all_data[int(len(all_data) * 0.8):]
        self.all_test_label = all_label[int(len(all_data) * 0.8):]

        return users, labels

    def get_rec_data(self):
        self.rec_data = np.concatenate([self.Exposed_data['x'], self.Test_data['x']])
        test_data_list = []
        test_label_list = []
        for i in range(len(self.rec_data)):
            idx = np.where(self.rec_data[i] > 0)[0]
            if (len(idx) > 0):
                cur_idx = idx[np.argsort(-self.rec_data[i, idx])[0]]
            else:
                cur_idx = 0
            test_data_list.append([i, cur_idx])
            test_label_list.append(self.rec_data[i, cur_idx])
            self.rec_data[i, cur_idx] = 0
        self.rec_data_test = np.array(test_data_list).astype(np.int)
        self.rec_label_test = np.array(test_label_list)

        print(np.min(self.rec_label_test), np.max(self.rec_label_test))
        print(np.min(self.rec_data), np.max(self.rec_data))

    def modify_graph_data(self, poison_data):
        print('modify')
        poison_data = np.concatenate(
            [poison_data, np.zeros((self.feature.shape[0] - poison_data.shape[0], poison_data.shape[1]))])
        feature_num = self.feature.shape[1]
        self.feature[self.idx] = self.feature[self.idx] + poison_data[:, :feature_num]
        self.adj = self.adj.tocsr()
        idx = np.argsort(self.idx)
        self.adj = self.adj + sp.csr_matrix(poison_data[idx][:, feature_num:])

    def modify_graph_data_all(self, poison_train_data, poison_test_data):
        poison_data = np.concatenate([poison_train_data, poison_test_data], axis=0)
        print('modify')
        feature_num = self.feature.shape[1]
        self.feature[self.idx] = self.feature[self.idx] + poison_data[:, :feature_num]
        self.adj = self.adj.tocsr()
        idx = np.argsort(self.idx)
        self.adj = self.adj + sp.csr_matrix(poison_data[idx][:, feature_num:])
