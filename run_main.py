import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--defense', type=str, help='the defense model', default='inf')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

# for model in ['dnn', 'logist1', 'svd', 'rf']:
#     for defense in ['inf', 'poigen', 'brew']:
#         for epoch in range(1):
#             for i in range(0, 1):
#                 # os.system("python main.py --defense inf --gpu %d --num %d --encrypt True --model %s --surrogate %s --dataset %s" % (
#                 #     args.gpu, i, args.surrogate, args.surrogate, args.dataset))
#                 for ratio in [1.]:
#                     os.system(
#                         "python main.py --defense %s --gpu %d --num %d --encrypt True --test True --ratio %f --model %s --surrogate %s --dataset %s" % (
#                             defense, args.gpu, i, ratio, model, args.surrogate, args.dataset))

for model in ['dnn', 'logist1', 'svd', 'rf']:
    for defense in ['arl']:
        for epoch in range(10):
            for i in range(5, 6):
                # os.system("python main.py --defense inf --gpu %d --num %d --encrypt True --model %s --surrogate %s --dataset %s" % (
                #     args.gpu, i, args.surrogate, args.surrogate, args.dataset))
                for ratio in [1.]:
                    os.system(
                        "python main.py --defense %s --gpu %d --num %d --encrypt True --test True --ratio %f --model %s --surrogate %s --dataset %s" % (
                            defense, args.gpu, i, ratio, model, args.surrogate, args.dataset))
