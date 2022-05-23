import argparse
import matplotlib.pyplot as plt
import numpy as np 
import time

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
parser.add_argument('--image_path', default="dataset/train-images-idx3-ubyte", type=str)
parser.add_argument('--label_path', default="dataset/train-labels-idx1-ubyte", type=str)
parser.add_argument('--epoch_num', default=10, type=int)
parser.add_argument('--normalize_x', default=True, type=str2bool)
parser.add_argument('--model', default="cnn", type=str)
args = parser.parse_args()

Proportion = [0.8, 0.1, 0.1] # proportion of train, valid and test data

if __name__ == '__main__':
	train_data, valid_data, test_data = split_data(Proportion, args)
	sampler = Sampler(train_data[0], train_data[1], args)
	

	criterion = CrossEntropyLoss()

	if args.model == 'cnn':
		net = CNN_Net()
	else:
		net = Linear_Net()

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
			t2 = time.time()
			print(loss, t2 - t1)

		print(e, loss,  evaluate(net, valid_data, args))
