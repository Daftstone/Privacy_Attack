import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="app", help="Choose a dataset.")  # city
parser.add_argument('--model', type=str, default="logist1", help="Choose a model.")  # city
parser.add_argument('--surrogate', type=str, default="logist1", help="Choose a model.")  # city
parser.add_argument('--defense', type=str, default="inf", help="Choose a defense model.")  # city
parser.add_argument('--gpu', type=str, default='0', help="gpu device")
parser.add_argument('--reg', type=float, default=1e-3, help="regularization")
parser.add_argument('--num', type=int, default=0, help="poison number")
parser.add_argument('--ratio', type=float, default=1., help="poison ratio")
parser.add_argument('--svd_dim', type=int, default=100, help="svd dimension")
parser.add_argument('--encrypt', type=bool, default=False, help="encrypt")
parser.add_argument('--test', type=bool, default=False, help="test")
parser.add_argument('--pred', type=bool, default=False, help="test")
parser.add_argument('--decrypt', type=int, default=0, help="encrypt")
parser.add_argument('--top_k', type=int, default=500, help="hit ratio")
parser.add_argument('--data_size', type=float, default=1., help="hit ratio")


FLAGS = parser.parse_args()
