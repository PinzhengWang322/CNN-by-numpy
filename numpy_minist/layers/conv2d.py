
import numpy as np

def zero_pad(X, pad):
    return np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)))

def conv_single_step(input_slice, W, b):
    s = np.multiply(input_slice, W)
    out = np.sum(s)
    out = out + b
    return out


class Conv2d():
    def __init__(self, in_C, out_C, K_size, stride = 1, pad = 0):
        a = np.sqrt(2) * np.sqrt(6/(in_C + out_C)) / K_size
        self.W = np.random.uniform(-a, a, size = (out_C, in_C, K_size, K_size))
        self.b = np.random.uniform(-a, a, size = (out_C, 1, 1, 1))
        self.stride = stride
        self.pad = pad
        self.w_gradient = np.zeros(self.W.shape)
        self.b_gradient = np.zeros(self.b.shape)
        self.K_size = K_size
        self.out_C = out_C
        self.in_C = in_C

    def conv_single_step(self, input_slice, W, b):
        s = input_slice * W
        out = np.sum(s)
        out = out + b
        return out

    def forward(self, x):
        self.x = x
        
        batch_size, in_C, in_H, in_W,  = x.shape # batch_size, in_channels, in_height, in_weight 
        out_C, in_C, K_size, K_size = self.W.shape # (in_C, K_size, K_size) is one convolution kernel

        self.batch_size = batch_size

        self.out_H = int((in_H + 2 * self.pad - K_size)/self.stride) + 1
        self.out_W = int((in_W + 2 * self.pad - K_size)/self.stride) + 1
        
        x = zero_pad(x, self.pad)
        out_tensor = np.zeros([batch_size, out_C, self.out_H, self.out_W])

        for i in range(batch_size):
            one_input = x[i]
            for c in range(out_C):
                for h in range(self.out_H):
                    for w in range(self.out_W):
                        vert_start = h * self.stride
                        vert_end = vert_start + K_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + K_size
                        input_slice = one_input[:,vert_start:vert_end,horiz_start:horiz_end]
                        weights = self.W[c, :, :, :]
                        biases = self.b[c, :, :, :]
                        out_tensor[i, c, h, w] = conv_single_step(input_slice, weights, biases)

        return out_tensor

    def gradient(self, d_out):
        x = zero_pad(self.x, self.pad)
        self.d_in = np.zeros(x.shape)
        for i in range(self.batch_size):
            one_input = x[i]
            for c in range(self.out_C):
                for h in range(self.out_H):
                    for w in range(self.out_W):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.K_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.K_size
                        input_slice = one_input[:,vert_start:vert_end,horiz_start:horiz_end]
                        self.w_gradient[c] += input_slice * d_out[i, c, h, w]
                        self.b_gradient[c] += d_out[i, c, h, w]
                        self.d_in[i, :, vert_start:vert_end, horiz_start:horiz_end] += d_out[i, c, h, w] * self.W[c, :, :, :]
        
        if self.pad != 0: self.d_in = self.d_in[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return self.d_in

    def backward(self, alpha=0.001, momentum=0.9):
        self.W -= alpha * self.w_gradient
        self.b -= alpha * self.b_gradient
        # zero gradient
        self.w_gradient *= momentum
        self.b_gradient *= momentum
        return

if __name__ == '__main__':
    conv2d = Conv2d(2,3,2,pad=1)
    np.random.seed(11)
    X = np.random.randint(18,size = (1,2,3,3))
    w = np.random.uniform(-0.1,0.1,(3,2,2,2))
    b = np.random.uniform(-0.1,0.1,(3))
    b = b.reshape(3,1,1,1)
    conv2d.W = w
    conv2d.b = b
    # print(conv2d.W)
    # print(conv2d.b)
    Y = conv2d.forward(X)
    # print(Y)
    g = conv2d.gradient(np.ones(Y.shape))
    # print(conv2d.w_gradient)
    # print(conv2d.b_gradient)
    print(g)
    conv2d.backward()