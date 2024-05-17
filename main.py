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

np.random.seed(78)

# Load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

nn = Network([784, 15, 15, 10])

x_train = transform_data(x_train/255.0)
y_train = one_hot_encode(transform_data(y_train))

nn.train(x_train, y_train, 20, 0.0003)


x_test = transform_data(x_test/255.0)
y_test = one_hot_encode(transform_data(y_test))

nn.test(x_test, y_test)