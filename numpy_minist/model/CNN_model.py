import sys
from tkinter import S 
sys.path.append("..") 
from layers.linear import *
from layers.conv2d import *
from layers.pool import *
from activation import *
import time

class CNN_Net():
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1,6,5)
        self.conv2 = Conv2d(6,16,5)
        self.pool1 = MaxPool2d(2, 2)
        self.pool2 = MaxPool2d(2, 2)
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()
        self.linear1 = Linear(16 * 4 * 4, 120)
        self.linear2 = Linear(120, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        # print(x[0,0,:3,:3])
        
        x = x.reshape(x.shape[0], -1)

        x = self.linear1.forward(x)
        x = self.relu3.forward(x)
        x = self.linear2.forward(x)
        t2 = time.time()
        
        return x

    def backpropagation(self, d_out):
        
        d_out = self.linear2.gradient(d_out)
        d_out = self.relu3.gradient(d_out)
        d_out = self.linear1.gradient(d_out)
        
        

        d_out = d_out.reshape((128, 16, 4, 4))
        
        d_out = self.pool2.gradient(d_out)
        d_out = self.relu2.gradient(d_out)
        d_out = self.conv2.gradient(d_out)

        d_out = self.pool1.gradient(d_out)
        d_out = self.relu1.gradient(d_out)
        d_out = self.conv1.gradient(d_out)

        print(self.conv1.w_gradient[:2,:2,:3,:3])

        self.linear2.backward()
        self.linear1.backward()

        self.conv2.backward()
        self.conv1.backward()
        return 

if __name__ == '__main__':
   pass