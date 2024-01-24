import cvxpy
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

for dataset in ['weibo']:
    for num in [1, 2, 3, 4, 5, 6, 7, 8]:
        train_poison_data = sp.load_npz(
            "temp/%s_logist_train_poison_data_%d_%d_%d.npz" % (dataset, num * 2, 1, 0)).toarray()
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
        sp.save_npz("temp/%s_logist3_train_poison_data_%d_%d_%d.npz" % (
            dataset, num, 1, 0), sp.csr_matrix(train_poison_data))
