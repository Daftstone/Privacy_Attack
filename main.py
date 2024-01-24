import warnings

warnings.filterwarnings("ignore")
import numpy as np
import os

from Dataset import Dataset
from models.Logist_linear import Logist
from models.Logist_linear1 import Logist as Logist1
from models.DNN import DNN
from models.RandomForest import RandomForest
from models.GradientBoosting import GradientBoost
from models.Adv_train import Adv_train
from models.Distillation import Distillation
from models.AttriGuard import AttriGuard
from models.PGA import PGA
from models.MaxEnt import MaxEnt
from models.PPCO import PPCO
from models.ARL import ARL
from models.Rec import SVD
from models.MF import MF
from models.MF_plus import MF_plus
from models.Slopeone import Slopeone
from models.SVD import SVDA
from models.PoiGen import PoiGen
from models.BREW import BREW

from parse import FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

models = {'logist': Logist, 'dnn': DNN, 'rf': RandomForest, 'adv': Adv_train, 'dis': Distillation, 'guard': AttriGuard,
          'gradientboost': GradientBoost, 'pga': PGA, 'maxent': MaxEnt, 'poigen': PoiGen, 'brew': BREW, 'ppco': PPCO,
          'arl': ARL,
          'rec': SVD, 'svd': SVDA, 'logist1': Logist1, 'mf': MF, 'mf_plus': MF_plus, 'slopeone': Slopeone}

dataset = Dataset()

model = models[FLAGS.model](dataset)
if (FLAGS.dataset == 'app'):
    if (FLAGS.model == 'dnn'):
        model.train(50, 256)
    else:
        model.train(50, 256)
elif (FLAGS.dataset == 'weibo'):
    if (FLAGS.model == 'dnn'):
        model.train(20, 1024)
    else:
        model.train(20, 1024)
elif (FLAGS.dataset == 'ml-100k'):
    if (FLAGS.model == 'dnn'):
        model.train(20, 32)
    else:
        model.train(200, 32)