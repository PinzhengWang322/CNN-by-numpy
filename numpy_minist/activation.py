import numpy as np

class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, d_out):
        self.d_out = d_out
        self.d_out[self.x<0]=0
        return self.d_out

if __name__ == '__main__':
    X = (np.arange(8) - 3).reshape(2,4)
    relu = Relu()
    Y = relu.forward(X)
    d_out = np.ones([2,4])
    g = relu.gradient(d_out)
    print("X:",X)
    print("Y:",Y)
    print("g:",g)