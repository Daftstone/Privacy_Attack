import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--defense', type=str, help='the attack model', default='inf')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

for model in ['gat']:
    for ee in range(10):
        for i in range(8, -1, -1):
            print(i)
            # os.system(
            #     "python main_gcn.py --defense inf --gpu %d --num %d --encrypt True --model %s --surrogate logist --dataset weibo --test True" % (
            #         args.gpu, i, args.model))
            os.system(
                "python main_gcn.py --defense %s --gpu %d --num %d --encrypt True --model %s --surrogate logist1 --dataset weibo --test True" % (
                    args.defense, args.gpu, i, model))
            # os.system(
            #     "python main_gcn.py --defense inf --gpu %d --num %d --encrypt True --model %s --surrogate logist2 --dataset weibo --test True" % (
            #         args.gpu, i, args.model))
