import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

# for epoch in range(1):
#     for defense in ['inf']:
#         for ratio in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
#             os.system(
#                 "python main.py --defense %s --gpu %d --num 8 --encrypt True --test True --ratio %f --model logist1 --surrogate logist1 --dataset %s" % (
#                     defense, args.gpu, ratio, args.dataset))

for epoch in range(1):
    for i in range(11):
        for ratio in [1.]:
            os.system(
                "python main.py --defense inf --gpu %d --num 8 --encrypt True --test True --ratio 0.2 --model logist1 --surrogate logist1 --dataset %s --data_size %f" % (
                    args.gpu, args.dataset, i / 10))
