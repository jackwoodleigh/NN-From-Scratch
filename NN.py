import numpy as np


def MSE(predicted, actual):
    return np.mean((predicted - actual) ** 2)


def MSE_grad(predicted, actual):
    return 2 * (predicted - actual)


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


# https://github.com/brianmanderson/Cyclical_Learning_Rate
def clr(iterations, step_size, base_lr, max_lr, gamma):
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** (iterations)
    return lr



class Layer:

    def __init__(self, input_count, node_count, name):
        self.x = np.zeros((input_count, 1))
        self.out = []
        self.name = name
        # create links from all nodes to input nodes
        self.w = np.random.rand(node_count, input_count) / input_count

        # create bias for each node
        self.b = np.random.rand(node_count, 1)

        # create base gradient for w
        self.dw = np.zeros((node_count, input_count))

        # create base gradient for b
        self.db = self.db = np.zeros((node_count, 1))

    def output(self, x):
        self.x = x
        out = np.dot(self.w, x) + self.b
        self.out.append(out)
        return self.activation(out)

    def activation(self, x):
        if self.name != 0:
            return relu(x)
        else:
            return x

    def activation_grad(self, x):
        if self.name != 0:
            return relu_grad(x)
        else:
            return 1



    def apply_gradient(self, learning_rate, lambda_reg, batch_size):
        # l2

        self.dw += lambda_reg * self.w

        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db

        # reset
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.out = []

    def calculate_grad(self, dj_dy, epoch):
        dj_da = dj_dy * self.activation_grad(self.out[epoch])
        self.dw += np.dot(dj_da, self.x.T)
        self.db += dj_da
        return np.dot(self.w.T, dj_da)


class Network:

    def __init__(self, network_shape):
        self.shape = network_shape

        # making layer classes for every layer except input
        self.layers = [Layer(network_shape[i - 1], network_shape[i], len(network_shape) - i + 1) for i in range(1, len(network_shape))]




    def forward_pass(self, input):
        inp = input
        for i in range(len(self.layers)):
            inp = self.layers[i].output(inp)

        return inp



    def backprop(self, pred, actual, lambda_reg, epoch):
        dj_dy = MSE_grad(pred, actual)
        for layer in reversed(self.layers):
            dj_dy = layer.calculate_grad(dj_dy, epoch)
            #layer.apply_gradient(learning_rate, lambda_reg)


    def train(self, x_train, y_train, epoch, lambda_reg, min_lr, max_lr, step_size, gamma):

        for e in range(epoch):
            total_loss = 0

            for i in range(len(x_train)):
                lr = clr(i, step_size, min_lr, max_lr, gamma)
                pred = self.forward_pass(x_train[i])
                self.backprop(pred, y_train[i], lambda_reg, 0)

                for layer in reversed(self.layers):
                    layer.apply_gradient(lr, lambda_reg, 1)

                total_loss += MSE(pred, y_train[i])
            print(f"{e}: {total_loss / len(x_train)}")
        print()





    def test(self, x_test, y_test):
        true = 0
        false = 0
        for i in range(len(x_test)):
            pred = self.forward_pass(x_test[i])
            num = np.argmax(pred)
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