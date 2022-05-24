from importlib.metadata import requires
from tkinter import W
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
parser.add_argument('--epoch_num', default=10, type=int)
parser.add_argument('--normalize_input', default=True, type=str2bool)
parser.add_argument('--normalize_w', default=True, type=str2bool)
parser.add_argument('--model', default='cnn', type=str)
args = parser.parse_args()

Proportion = [0.8, 0.1, 0.1] # proportion of train, valid and test data

if __name__ == '__main__':
    # net = Linear_Net()

    # if args.normalize_w:
    #     for name, param in net.named_parameters():
    #         try:
    #             torch.nn.init.xavier_normal_(param.data)
    #         except:
    #             pass # just ignore those failed init layers

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001)

    
    # # print(net.fc1.weight.shape)
    # np.random.seed(10)    
    # X = np.random.randint(0,255,size = [128, 28 * 28])
    # y = np.random.randint(1,9,size = 128)
    # X = normalize(X)  
    # w1 = np.random.uniform(-0.1,0.1,(28 * 28, 300)).T
    # # print(w1[:3,:3])
    # b1 = np.random.uniform(-0.1,0.1,(300))
    # w2 = np.random.uniform(-0.1,0.1,(300, 10)).T
    # # print(w2[:3,:3])
    # b2 = np.random.uniform(-0.1,0.1,(10))
    # net.fc1.weight.data = torch.tensor(w1).float()
    # net.fc2.weight.data = torch.tensor(w2).float()
    # net.fc1.bias.data = torch.tensor(b1).float()
    # net.fc2.bias.data = torch.tensor(b2).float()
    # X = torch.tensor(X)
    # y = torch.tensor(y)
            
    # X = X.float()

    # out = net(X)

    # loss = criterion(out, y.reshape(-1).long())
    # print(loss)
    # loss.backward()
    # print(net.fc2.weight[:3,:3])

    # optimizer.step()
    # print(net.fc2.weight[:3,:3])
    # optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
   
    net = CNN_Net()

    optimizer = optim.SGD(net.parameters(), lr=0.001)
    np.random.seed(11) 

    w1 = np.random.uniform(-0.1,0.1,net.conv1.weight.shape).astype('float32')
    w2 = np.random.uniform(-0.1,0.1,net.conv2.weight.shape).astype('float32')
    w3 = np.random.uniform(-0.1,0.1,(256,120)).astype('float32')
    w4 = np.random.uniform(-0.1,0.1,(120,10)).astype('float32')

    b1 = np.random.uniform(-0.1,0.1,net.conv1.bias.shape).astype('float32')
    b2 = np.random.uniform(-0.1,0.1,net.conv2.bias.shape).astype('float32')
    b3 = np.random.uniform(-0.1,0.1,net.linear1.bias.shape).astype('float32')
    b4 = np.random.uniform(-0.1,0.1,net.linear2.bias.shape).astype('float32')

    net.conv1.weight.data = torch.tensor(w1)
    net.conv1.bias.data = torch.tensor(b1)
    net.conv2.weight.data = torch.tensor(w2)
    net.conv2.bias.data = torch.tensor(b2)
    net.linear1.weight.data = torch.tensor(w3.T)
    net.linear1.bias.data = torch.tensor(b3)
    net.linear2.weight.data = torch.tensor(w4.T)
    net.linear2.bias.data = torch.tensor(b4)

    X = np.random.randint(0,255,size = [128, 1, 28, 28])
    X = normalize(X).astype('float32')
    y = np.random.randint(0,9,size = 128)
    X = torch.tensor(X)
    y = torch.tensor(y)

    out = net(X)
    loss = criterion(out, y.reshape(-1).long())
    loss.backward()
    print(loss.item())
    # print(net.conv1.weight.grad[:2,:2,:3,:3])