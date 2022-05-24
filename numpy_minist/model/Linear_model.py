import sys 
sys.path.append("..") 
from layers.linear import *
from activation import *

class Linear_Net():
    def __init__(self, args):
        super().__init__()
        self.linear1 = Linear(784, 300)
        self.linear2 = Linear(300, 10)
        self.relu1 = Relu()
        self.lr = args.lr
        self.momentum = args.momentum

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear2.forward(x)
        
        return x

    def backpropagation(self, d_out):
        d_out = self.linear2.gradient(d_out)
        d_out = self.relu1.gradient(d_out)
        d_out = self.linear1.gradient(d_out)

        
        self.linear2.backward(self.lr, self.momentum)
        self.linear1.backward(self.lr, self.momentum)
        return 

if __name__ == '__main__':
    a = np.ones([128,784])
    linear_net = Linear_Net()
    b = linear_net.forward(a)
    linear_net.backpropagation(b - 1)
    print(b.shape)