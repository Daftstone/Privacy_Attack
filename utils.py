from time import time
from time import strftime
from time import localtime
import os
import tensorflow as tf
import numpy as np
import math
import copy
import scipy.sparse as sp

from parse import FLAGS


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def pert_vector_product(ys, xs1, xs2, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs1)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs2)[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs2)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(xs2) \
                    for grad_elem in grads_with_none]
    return return_grads


def hessian_vector_product(ys, xs, v, do_not_sum_up=True, scales=1.):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i] / scales, xs[i])[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products / scales, xs)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads


def cal_inf_encrypt(x, y, inf_vec):
    count = np.sum(x != 0, axis=1)
    poison_data = np.zeros_like(x)
    temp_data = np.zeros_like(poison_data)
    # for i in range(10000):
    #     temp_data[:, i] = np.clip(np.random.normal(distribution[0][i], distribution[1][i], len(temp_data)), 0, 1)
    temp_data = np.round(temp_data * 5) / 5
    inf_list = 0
    for i in range(199, 200):
        temp = inf_vec
        temp = temp / np.sum(np.abs(temp))
        if (i == 0):
            inf_list = temp
        else:
            inf_list = inf_list * 0.9 + temp * 0.1
    for i in range(len(x)):
        inf = inf_list[i]

        temp_data[i] = np.ones_like(inf)
        value = (temp_data[i] - x[i]) * inf
        idx = np.where((x[i] == 0) * (value > 0))[0]
        max_idx = idx[np.argsort(-value[idx])[:FLAGS.num]]
        poison_data[i, max_idx] = temp_data[i, max_idx] - x[i, max_idx]
    return poison_data, y


def cal_inf_decrypt(x, y, inf_vec):
    count = np.sum(x != 0, axis=1)
    poison_data = np.zeros_like(x)
    temp_data = np.zeros_like(poison_data)
    # for i in range(10000):
    #     temp_data[:, i] = np.clip(np.random.normal(distribution[0][i], distribution[1][i], len(temp_data)), 0, 1)
    temp_data = np.round(temp_data * 5) / 5
    inf_list = 0
    for i in range(2, 3):
        temp = inf_vec
        temp = temp / np.sum(np.abs(temp))
        if (i == 0):
            inf_list = temp
        else:
            inf_list = inf_list * 0.9 + temp * 0.1

    for i in range(len(x)):
        inf = inf_list[i]

        temp_data[i] = np.zeros_like(inf)
        value = (temp_data[i] - x[i]) * inf
        idx = np.where((x[i] > 0) * (value < 0))[0]
        max_idx = idx[np.argsort(value[idx])[:FLAGS.num]]
        poison_data[i, max_idx] = temp_data[i, max_idx] - x[i, max_idx]
    return poison_data, y


def cal_mean_data(data):
    all_x = data['x']
    all_y = data['y']
    label_num = all_y.shape[1]
    mean_x = all_x[:label_num].copy()
    for i in range(label_num):
        idx = np.where(all_y[:, i] == 1)[0]
        mean_x[i] = np.mean(all_x[idx], axis=0)
    mean_y = np.eye(label_num)[np.arange(label_num)]
    return mean_x, mean_y


def cal_inf_encrypt_mean1(x, y, inf_vec, num=FLAGS.num):
    poison_data = np.zeros_like(x)
    temp_data = np.zeros_like(poison_data)
    temp_data = np.round(temp_data * 5) / 5
    inf_list = 0
    temp = inf_vec / np.sum(np.abs(inf_vec))
    # inf_list = inf_list * 0.9 + temp * 0.1
    labels = np.argmax(y, axis=1)
    for i in range(len(x)):
        label = labels[i]
        inf = temp[label]

        temp_data[i] = np.ones_like(inf)
        value = (temp_data[i] - x[i]) * inf
        idx = np.where((x[i] == 0) * (value > 0))[0]
        max_idx = idx[np.argsort(-value[idx])[:num]]
        poison_data[i, max_idx] = temp_data[i, max_idx] - x[i, max_idx]
    return poison_data, y


def cal_inf_encrypt_mean(x, y, inf_vec, num=FLAGS.num):
    poison_data = np.zeros_like(x)
    temp_data = np.zeros_like(poison_data)
    temp_data = np.round(temp_data * 5) / 5
    inf_list = 0
    temp = inf_vec / np.sum(np.abs(inf_vec))
    # inf_list = inf_list * 0.9 + temp * 0.1
    labels = np.argmax(y, axis=1)
    inf = temp[labels]
    temp_data = np.ones_like(inf[0])
    values = (1. - x) * inf
    cond = (x == 0) * (values > 0)
    for i in range(len(x)):
        value = values[i]
        idx = np.where(cond[i])[0]
        max_idx = idx[np.argsort(-value[idx])[:num]]
        poison_data[i, max_idx] = 1. - x[i, max_idx]
    return poison_data, y


def cal_inf_decrypt_mean(x, y, inf_vec, num=FLAGS.num):
    poison_data = np.zeros_like(x)
    temp_data = np.zeros_like(poison_data)
    temp_data = np.round(temp_data * 5) / 5
    inf_list = 0
    temp = inf_vec
    temp = temp / np.sum(np.abs(temp))
    inf_list = inf_list * 0.9 + temp * 0.1
    for i in range(len(x)):
        label = np.argmax(y[i])
        inf = inf_list[label]

        temp_data[i] = np.zeros_like(inf)
        value = (temp_data[i] - x[i]) * inf
        idx = np.where((x[i] > 0) * (value < 0))[0]
        max_idx = idx[np.argsort(value[idx])[:num]]
        poison_data[i, max_idx] = temp_data[i, max_idx] - x[i, max_idx]
    return poison_data, y


def step2(train_poison_data, dataset, num):
    import cvxpy as cp
    Exposed_data = sp.load_npz("temp/%s_logist_exposed_data_%d_%d.npz" % (dataset, 1, 0)).toarray()
    assert len(train_poison_data) == len(Exposed_data)
    print(len(train_poison_data))

    idx = np.where(np.sum(train_poison_data, axis=0) > 0)[0]

    print(len(idx))

    poison_data = train_poison_data[:, idx]
    exposed_data = Exposed_data[:, idx]
    prior = np.sum(exposed_data > 0, axis=0) / np.sum(exposed_data > 0)
    print(np.sort(prior)[::-1])
    mask = poison_data > 0

    Q = cp.Variable((poison_data.shape[0], poison_data.shape[1]))
    poster = cp.sum(cp.multiply(mask, Q), axis=0) / (len(poison_data) * num)
    objective = cp.sum_squares(prior - poster)
    # objective = -cp.sum(prior * poster)
    # objective = cp.sum(cp.kl_div(prior, cp.sum(temp, axis=0) / cp.sum(temp)))
    obj = cp.Minimize(objective)

    constraints = [cp.sum(cp.multiply(Q, mask), axis=1) == num, Q >= 0, Q <= 1]
    prob = cp.Problem(obj, constraints)
    # prob.solve(verbose=True, max_iters=1000, feastol=1e-3)
    prob.solve(verbose=True)
    print(np.sort((np.array(Q.value) * (poison_data > 0))[0])[::-1])
    Q_value = np.array(Q.value) * (poison_data > 0)
    for i in range(len(poison_data)):
        cur_idx = np.argsort(-Q_value[i])[num:]
        train_poison_data[i, idx[cur_idx]] = 0
    return train_poison_data
