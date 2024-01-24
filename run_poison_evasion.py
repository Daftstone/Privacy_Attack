import os
import argparse

parser = argparse.ArgumentParser(description='run all cases')

parser.add_argument('--model', type=str, help='the attack model', default='logist')
parser.add_argument('--defense', type=str, help='the defense model', default='inf')
parser.add_argument('--surrogate', type=str, help='the surrogate model', default='logist')
parser.add_argument('--dataset', type=str, help='dataset', default='app')
parser.add_argument('--gpu', type=int, help='gpu device', default=0)

args = parser.parse_args()

for i in range(10):
    for defense in ['inf', 'poigen', 'brew']:
        for attack in ['pga', 'guard']:
            os.system(
                "python main.py --defense %s_%s --gpu 8 --num 5 --encrypt True --model dnn --surrogate logist1 --dataset weibo --test True" % (
                    defense, attack))
            # os.system(
            #     "python main.py --defense %s_%s --gpu 7 --num 5 --encrypt True --model %s --surrogate logist1 --dataset app" % (
            #         defense, attack, attack))

for i in range(10):
    for attack in ['pga', 'guard']:
        os.system(
            "python main.py --defense %s --gpu 8 --num 5 --encrypt True --model dnn --surrogate logist1 --dataset weibo --test True" % (
                attack))
        # os.system(
        #     "python main.py --defense %s_%s --gpu 7 --num 5 --encrypt True --model %s --surrogate logist1 --dataset app" % (
        #         defense, attack, attack))