# CNN-by-numpy
这是一个主要使用 numpy 实现 CNN的项目，并完成了minist分类任务。

分为pytorch版和numpy版两种，每版里面都有全连接模型和CNN模型俩种，以供对比验证。

下面的介绍主要针对numpy版的CNN，pytorch主要作用是验证numpy版CNN的精度。

### 运行方法：

```bash
python main.py
```



### 参数设置：

```
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--momentum					 FLOAT         Momentum of SGD.							  Default is 0.9.
--image_path				 STR    			 The path of minist image path. Default is "dataset/train-images-idx3-ubyte".
--label_path  			 STR    			 The path of minist label path. Default is "dataset/train-labels-idx1-ubyte".
--epoch_num          INT           The number of epochs.          Default is 3.
--normalize_x        BOOL          Whether to normalize the input Default is True.
--model              STR           Use cnn model or linear model  Default is "cnn".
```



###模型结构：

```python
class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 4 * 4, 120)
        self.linear2 = nn.Linear(120,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

主要结构如上图pytorch所示，两层卷积层，两层池化层，两层线性层，激活函数选用Relu函数。numpy版模型结构与pytorch版的完全一样。



### 效果演示：





![Figure_1](/Users/wangpinzheng/Desktop/gitcode/CNN-5.24/pics/Figure_1.png)

能在3个回合内到达95%以上的准确度





#CNN-by-numpy(English)

This is a project that mainly uses numpy to implement a CNN and completes the minist classification task.

There are two versions of CNN: pytorch version and numpy version. Each version has two types of fully connected model and CNN model for comparison and verification.

The following introduction is mainly for the numpy version of CNN, as the main function of pytorch is to verify the accuracy of the numpy version of CNN.



### Commands：

To train our model on the default data with default parameters:

```
python main.py
```



### Options:
```
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--momentum					 FLOAT         Momentum of SGD.							  Default is 0.9.
--image_path				 STR    			 The path of minist image path. Default is "dataset/train-images-idx3-ubyte".
--label_path  			 STR    			 The path of minist label path. Default is "dataset/train-labels-idx1-ubyte".
--epoch_num          INT           The number of epochs.          Default is 3.
--normalize_x        BOOL          Whether to normalize the input Default is True.
--model              STR           Use cnn model or linear model  Default is "cnn".
```

###Model structure:

```python
class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 4 * 4, 120)
        self.linear2 = nn.Linear(120,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```
The main structure is shown in pytorch above, with two layers of convolution layers, two layers of pooling layers, two layers of linear layers, and the activation function uses the Relu function. The model structure of the numpy version is exactly the same as that of the pytorch version.

### Effect demonstration

![Figure_1](/Users/wangpinzheng/Desktop/gitcode/CNN-5.24/pics/Figure_1.png)


Able to achieve more than 95% accuracy in 3 epochs
