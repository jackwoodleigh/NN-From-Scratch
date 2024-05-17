import numpy as np


def MSE(predicted, actual):
    return np.mean((predicted - actual) ** 2)


def MSE_grad(predicted, actual):
    return 2 * (predicted - actual)#(predicted - actual)


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad


def cross_entropy(predicted, actual):
    epsilon = 1e-10
    return -np.sum(actual * np.log(predicted + epsilon))


def cross_entropy_grad(predicted, actual):
    return predicted - actual

def c_sigmoid(x):
    return 1 / (1 + 2*np.exp(-1 * 5*x))

class Layer:

    def __init__(self, input_count, node_count):
        self.x = np.zeros((input_count, 1))
        self.out = np.zeros((node_count, 1))

        # create links from all nodes to input nodes
        self.w = np.random.rand(node_count, input_count) * np.sqrt(2. / input_count)

        # create bias for each node
        self.b = np.random.rand(node_count, 1)

        # create base gradient for w
        self.dw = np.zeros((node_count, input_count))

        # create base gradient for b
        self.db = self.db = np.zeros((node_count, 1))

    def output(self, x):
        self.x = x
        self.out = np.dot(self.w, x) + self.b
        return relu(self.out)
################################################
    def apply_gradient(self, learning_rate):
        # put iterations in to enable batches  / iterations_preformed
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def calculate_grad(self, dj_dy):
        dj_da = dj_dy * relu_grad(self.out)
        self.dw += np.dot(dj_da, self.x.T)
        self.db += dj_da
        return np.dot(self.w.T, dj_da)


class Network:

    def __init__(self, network_shape):
        self.shape = network_shape

        # making layer classes for every layer except input
        self.layers = [Layer(network_shape[i - 1], network_shape[i]) for i in range(1, len(network_shape))]

        self.learning_rate = 0.000013

        self.alpha = 0.5


    def forward_pass(self, input):
        inp = input
        for i in range(len(self.layers)):
            inp = self.layers[i].output(inp)

        return inp



    def backprop(self, pred, actual, learning_rate):
        # print(MSE(pred, actual))
        
        dj_dy = MSE_grad(pred, actual)
        for layer in reversed(self.layers):
            dj_dy = layer.calculate_grad(dj_dy)
            layer.apply_gradient(learning_rate)


    def train(self, x_train, y_train, epoch, learning_rate):
        #print(c_sigmoid(1 - ((epoch - 6)/epoch)))
        for e in range(epoch):
            all_pred = []
            total_loss = 0
                # 2 * c_sigmoid(1-(epoch - e)/epoch) *

            l = (e+1) * learning_rate


            for i in range(len(x_train)):
                pred = self.forward_pass(x_train[i])
                all_pred.append(pred)
                self.backprop(pred, y_train[i], l)
                total_loss += MSE(pred, y_train[i])
            print(f"{e}: {total_loss / len(x_train)}")
        print()



    def test(self, x_test, y_test):
        true = 0
        false = 0
        for i in range(len(x_test)):
            pred = self.forward_pass(x_test[i])
            num = np.argmax(pred)
            #print(f"{np.argmax(y_test[i])} {num}")
            if y_test[i][num][0] == 1:
                true += 1
            else:
                false += 1
        print(true)
        print(false)
        print(true/(true+false))

'''
nn = Network([1,3, 3, 1])
x = np.random.rand(1000, 1, 1) * 5

y = np.array(x)
nn.train(x, y**2, 200, 0.000013)
print(nn.forward_pass([[3]]))'''