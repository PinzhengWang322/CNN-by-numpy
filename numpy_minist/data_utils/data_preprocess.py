from random import random
from mlxtend.data import loadlocal_mnist
import numpy as np
import gzip
import matplotlib.pyplot as plt
import random
import argparse

def extract_data(image_path, label_path):
    X, y = loadlocal_mnist(
            images_path=image_path, 
            labels_path=label_path)
    return X,y

def split_data(Proportion, args):
    X, y = loadlocal_mnist(
            images_path = args.image_path, 
            labels_path = args.label_path)
    y = y.reshape(-1, 1)
    A = np.concatenate([X, y], axis = 1)
    A = A[np.random.choice(A.shape[0], A.shape[0], replace=False), :]
    for i in range(2): Proportion[i + 1] += Proportion[i]
    Proportion = np.array([int(i * X.shape[0]) for i in Proportion])
    return [np.hsplit(a, np.array([a.shape[1] - 1])) for a in np.vsplit(A, Proportion[:-1])]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_path', default="../dataset/train-images-idx3-ubyte", type=str)
    parser.add_argument('--label_path', default="../dataset/train-labels-idx1-ubyte", type=str)
    parser.add_argument('--epoch_num', default=10, type=int)
    args = parser.parse_args()


    train, valid, test = split_data([0.8, 0.1, 0.1], args)
    print(train[0].shape, train[1].shape, test[0].shape, test[1].shape, valid[0].shape, valid[1].shape)

    print(train[0][120])

    X, y = valid
    idx = random.randint(0, X.shape[0])
    plt.imshow(X[idx].reshape(28, 28))
    plt.title(y[idx].item())
    plt.show()
    