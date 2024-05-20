import numpy as np
import tensorflow as tf
from NN import Network


def transform_data(x):
    return np.array([x[i].reshape(-1, 1) for i in range(len(x))])


def one_hot_encode(x):
    new_arr = []
    for i in range(len(x)):
        temp = np.zeros((10, 1), dtype=int)
        temp[x[i, 0]] = 1
        new_arr.append(temp)
    return np.array(new_arr)


np.random.seed(8)

# Load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

nn = Network([784, 15, 15, 10])
x_train = transform_data(x_train/255.0)
y_train = one_hot_encode(transform_data(y_train))


epochs = 20
step_size = 10
min_lr = 0.01
max_lr = 0.003
lambda_reg = 0.0001
step_size = 100
gamma = 0.9999



nn.train(x_train, y_train, epochs, lambda_reg, min_lr, max_lr, step_size, gamma)


x_test = transform_data(x_test/255.0)
y_test = one_hot_encode(transform_data(y_test))

nn.test(x_test, y_test)