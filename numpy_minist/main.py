import argparse
import matplotlib.pyplot as plt
import numpy as np 
import time
import os

from data_utils import *
from data_utils.data_preprocess import *
from data_utils.normalize import *
from model.CNN_model import CNN_Net
from model.Linear_model import Linear_Net
from sampler import *

from loss_function.CrossEntropyLoss import CrossEntropyLoss
from evaluate import evaluate


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--image_path', default="dataset/train-images-idx3-ubyte", type=str)
parser.add_argument('--label_path', default="dataset/train-labels-idx1-ubyte", type=str)
parser.add_argument('--epoch_num', default=3, type=int)
parser.add_argument('--normalize_x', default=True, type=str2bool)
parser.add_argument('--model', default="cnn", type=str)
args = parser.parse_args()

Proportion = [0.98, 0.01, 0.01] # proportion of train, valid and test data

if __name__ == '__main__':
	train_data, valid_data, test_data = split_data(Proportion, args)
	sampler = Sampler(train_data[0], train_data[1], args)
	f = open(str(args.model) + '_record', 'w')
	
	criterion = CrossEntropyLoss()

	if args.model == 'cnn':
		net = CNN_Net(args)
	else:
		net = Linear_Net(args)

	print((train_data[1].shape[0] // args.batch_size))
	
	for e in range(1, args.epoch_num + 1):
		for step in range(train_data[1].shape[0] // args.batch_size):
			t1 = time.time()
			X, y = sampler.next_batch()
			if args.normalize_x:
				X = normalize(X)
			
			if args.model == "cnn":
				X = X.reshape(-1, 1, 28, 28)
				
			out = net.forward(X)
			loss = criterion.cal_loss(out, y)
			gardient = criterion.gradient()
			net.backpropagation(gardient)

			if (step % 20 == 0):
				t2 = time.time()
				f.write(str(e) + ',' + str(step) + ":" + str(loss) +'\n'+ str(evaluate(net, valid_data, args)) + '\n')
				f.flush()
