import tensorflow as tf
import numpy as np
from surprise import SVDpp
from surprise import Dataset, accuracy, Reader
from surprise.model_selection import cross_validate
import pandas as pd
from surprise.model_selection import train_test_split

import utils

from parse import FLAGS


class MF_plus:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self.prepare_data()
        self.algo = SVDpp()

    def train(self, epochs, batch_size):
        trainset, testset = train_test_split(self.data, test_size=0.2)
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)

        # Then compute RMSE
        results = accuracy.rmse(predictions)
        np.save("results/%s_%s_%s_%d_%d_%d_%.2f.npy" % (
            FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                results)

    def prepare_data(self):
        rec_data = np.concatenate([self.dataset.Exposed_data['x'], self.dataset.Test_data['x']])
        idx = np.where(rec_data > 0)
        rating_list = []
        for i in range(len(idx[0])):
            rating_list.append(rec_data[idx[0][i], idx[1][i]])
        ratings_dict = {
            "itemID": list(idx[0]),
            "userID": list(idx[1]),
            "rating": rating_list,
        }
        df = pd.DataFrame(ratings_dict)

        # A reader is still needed but only the rating_scale param is requiered.
        reader = Reader(rating_scale=(1, 5))

        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
        return data
