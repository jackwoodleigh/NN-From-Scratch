import numpy as np
import tensorflow as tf
from NN_test import Network


'''def transform_data(x):
    return np.array([x[i].reshape(-1, 1) for i in range(len(x))])


def one_hot_encode(x):
    new_arr = []
    for i in range(len(x)):
        temp = np.zeros((10, 1), dtype=int)
        temp[x[i, 0]] = 1
        new_arr.append(temp)
    return np.array(new_arr)'''

def transform_data(x):
    return np.array([x[i].reshape(-1, 1) for i in range(len(x))])


def one_hot_encode(x):
    new_arr = []
    for i in range(len(x)):
        temp = np.zeros((10, 1), dtype=int)
        temp[x[i, 0]] = 1
        new_arr.append(temp)
    return np.array(new_arr)





np.random.seed(81)

# Load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

nn = Network([784, 256, 128, 64, 10])
x_train = transform_data(x_train)
y_train = one_hot_encode(transform_data(y_train))
x_test = transform_data(x_test)
y_test = one_hot_encode(transform_data(y_test))



epochs = 60000
batch_size = 30
learning_rate = 0.05
lr1 = 0.0005
lr2 = 0.001
lambda_reg = 0.00008
b1 = 0.85
b2 = 0.98
e = 1e-8


nn.train(x_train, y_train, epochs, batch_size, learning_rate, lr1, lr2, lambda_reg, b1, b2, e)

nn.test(x_test, y_test)