import warnings

warnings.filterwarnings("ignore")
import numpy as np
import os

import argparse

from Dataset import Dataset
from models.Logist_linear1 import Logist as Logist1
from models.DNN import DNN
# from models.RandomForest import RandomForest
# from models.Adv_train import Adv_train
# from models.Distillation import Distillation
# from models.AttriGuard import AttriGuard
# from models.Rec import SVD
# from models.MF import MF
# from models.MF_plus import MF_plus
# from models.SVD import SVDA
from models.GCN import GCN
from models.GAT import GCN as GAT
from models.GATA import GCN as GATA

from parse import FLAGS

models = {'gcn': GCN, 'gat': GAT, 'gata': GATA}

dataset = Dataset()

model = models[FLAGS.model](dataset)
if (FLAGS.dataset == 'app'):
    if (FLAGS.model == 'dnn'):
        model.my_train(50, 256)
    else:
        model.my_train(50, 256)
elif (FLAGS.dataset == 'ml-100k'):
    if (FLAGS.model == 'dnn'):
        model.my_train(20, 32)
    else:
        model.my_train(200, 32)
elif (FLAGS.dataset == 'weibo'):
    if (FLAGS.model == 'dnn'):
        model.my_train(30, 256)
    else:
        model.my_train(30, 256)
