import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

for epoch in range(2):
    for defense in ['maxent', 'arl']:
        for ratio in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            os.system(
                "python main.py --defense %s --gpu %d --num 5 --encrypt True --test True --ratio %f --model dnn --surrogate logist1 --dataset %s" % (
                    defense, args.gpu, ratio, args.dataset))
