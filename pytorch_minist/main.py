from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.optim as optim

from Utils.normalize import *
from Utils.data_preprocess import *
from sampler import *
from model import *
from evaluate import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--image_path', default="dataset/train-images-idx3-ubyte", type=str)
parser.add_argument('--label_path', default="dataset/train-labels-idx1-ubyte", type=str)
parser.add_argument('--epoch_num', default=1, type=int)
parser.add_argument('--normalize_input', default=True, type=str2bool)
parser.add_argument('--normalize_w', default=True, type=str2bool)
parser.add_argument('--model', default='cnn', type=str)
args = parser.parse_args()

Proportion = [0.98, 0.01, 0.01] # proportion of train, valid and test data

if __name__ == '__main__':
	train_data, valid_data, test_data = split_data(Proportion, args)
	sampler = Sampler(train_data[0], train_data[1], args)
	
	if args.model == 'cnn':
		net = CNN_Net()	
	else:
		net = Linear_Net()

	if args.normalize_w:
		for name, param in net.named_parameters():
			try:
				torch.nn.init.xavier_normal_(param.data)
			except:
				pass # just ignore those failed init layers

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	for e in range(1, args.epoch_num + 1):
		for step in range(train_data[1].shape[0] // args.batch_size):
			X, y = sampler.next_batch()
			if args.normalize_input:
				X = normalize(X)
			
			X, y = torch.tensor(X), torch.tensor(y)
			if args.model == 'cnn': 
				X = X.reshape(-1, 1, 28, 28).float()
			else:
				X = X.float()
			
			out = net(X)
			
			loss = criterion(out, y.reshape(-1).long())
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			if (step % 20 == 0):
				print(evaluate(net, valid_data, args))
	


		