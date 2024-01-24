import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

for model in ['logist1']:
    args.model = model
    for i in range(11):
        # os.system("python main.py --defense inf --gpu %d --num %d --encrypt True --model %s --surrogate %s --dataset %s" % (
        #     args.gpu, i, args.surrogate, args.surrogate, args.dataset))
        for ratio in [1.]:
            os.system(
                "python main.py --defense inf --gpu %d --num %d --encrypt True --test True --ratio %f --model %s --surrogate %s --dataset %s --data_size %f" % (
                    args.gpu, 5, 0.5, args.model, args.surrogate, args.dataset, i/10))
