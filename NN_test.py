import numpy as np


def MSE(predicted, actual):
    return np.mean((actual - predicted) ** 2)


def MSE_grad(predicted, actual):
    return (2/len(predicted)) * (predicted - actual)


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

    def __init__(self, input_count, node_count, id):

        self.id = id

        # outputs
        self.x = np.zeros((input_count, 1))
        self.z = None
        self.a = None
        self.a_normalized_tr = None
        self.a_normalized_inf = None
        self.a_hat_normalized = None



        # weights for each
        #self.w = np.random.rand(node_count, input_count) / (input_count) # * np.sqrt(2. / (node_count + input_count)) # He initialization / (input_count + node_count)
        self.w = np.random.rand(node_count, input_count) * np.sqrt(1. / (input_count+node_count))
        # bias for each node
        self.b = np.random.rand(node_count, 1)

        # gradients
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

        # adam parameters
        self.m_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)

        self.v_w = np.zeros_like(self.w)
        self.v_b = np.zeros_like(self.b)

        # Batch normalization parameters
        self.gamma = np.ones((node_count, 1))   # variance scaler
        self.beta = np.zeros((node_count, 1))   # mean scaler

        self.mean = None
        self.var = None
        self.running_mean = np.zeros((node_count, 1))
        self.running_var = np.ones((node_count, 1))
        self.N = 0
        self.is_training = 1

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)




    # used to customize layer activation
    def activation(self, x):
        if self.id != 0:
            return relu(x)

        else:
            return x

    # used to customize layer activation
    def activation_grad(self, x):
        if self.id != 0:
            return relu_grad(x)
        else:
            return 1

    def batch_norm(self, e):
        if self.is_training:
            self.mean = np.mean(self.a, axis=0)
            self.var = np.var(self.a, axis=0)
            batch_size = len(self.a)


            self.running_mean = (self.N * self.running_mean + batch_size * self.mean) / (self.N + batch_size)
            self.running_var = (self.N * self.running_var + batch_size * self.var) / (self.N + batch_size)
            self.N += batch_size


            self.a_normalized_tr = (self.a - self.mean) / np.sqrt(self.var + e)
            self.a_hat_normalized = self.gamma * self.a_normalized_tr + self.beta

        else:
            self.a_normalized_inf = self.gamma / (np.sqrt(self.running_var + e)) * self.a + (self.beta - (self.gamma*self.running_mean) / (np.sqrt(self.running_var + e)))
            self.a_hat_normalized = self.gamma * self.a_normalized_inf + self.beta

        return self.a_hat_normalized





    def batch_norm_grad(self, dj_dy, e):


        self.dbeta = np.sum(dj_dy, axis=0)
        self.dgamma = np.sum(dj_dy * self.a_hat_normalized, axis=0)


        m = len(self.a)
        norm = self.a - self.mean
        sqrt_var_eps = -1/np.sqrt(self.var + e)

        da_hat_normalized = dj_dy * self.gamma
        dsigma = np.sum(da_hat_normalized * norm, axis=0) * -0.5 * (self.var + e)**(-3/2)
        dmu = np.sum(da_hat_normalized * sqrt_var_eps, axis=0) + dsigma * (np.sum(-2*norm, axis=0)/m)
        da = da_hat_normalized * sqrt_var_eps + dsigma * (2*norm/m) + dmu/m

        dz = da*relu_grad(self.a)


        return dz

    def output(self, x):

        self.x = x
        self.z = np.matmul(self.w, x) + self.b
        self.a = self.activation(self.z)
        # self.a_hat_normalized = self.batch_norm(0.000001)
        # return self.a_hat_normalized
        return self.a

    def calculate_grad(self, dj_dy):
        #dj_dz = self.batch_norm_grad(dj_dy, 1e-8)
        dj_dz = dj_dy * self.activation_grad(self.a)

        self.dw = np.matmul(dj_dz, self.x.transpose(0, 2, 1))
        self.db = dj_dz
        return np.matmul(self.w.T, dj_dz)


    def apply_gradient(self, learning_rate, lambda_reg, b1, b2, e, t):
        #self.dw = np.mean(self.dw, axis=0)
        #self.db = np.mean(self.db, axis=0)

        # adam parameters
        self.m_w = b1 * self.m_w + (1 - b1) * self.dw
        self.m_b = b1 * self.m_b + (1 - b1) * self.db
        self.v_w = b2 * self.v_w + (1 - b2) * (self.dw ** 2)
        self.v_b = b2 * self.v_b + (1 - b2) * (self.db ** 2)

        # Compute bias-corrected moments



        t_w = ((self.m_w / (1 - (b1 ** t))) / (np.sqrt((self.v_w / (1 - (b2 ** t))) + e)))
        t_b = ((self.m_b / (1 - (b1 ** t))) / (np.sqrt((self.v_b / (1 - (b2 ** t))) + e)))
        update_w = np.mean(learning_rate * t_w, axis=0)
        update_b = np.mean(learning_rate * t_b, axis=0)

        self.w -= update_w
        self.b -= update_b

        '''w = learning_rate * np.mean(self.dw, axis=0)
        b = learning_rate * np.mean(self.db, axis=0)
        self.w -= w
        self.b -= b'''
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta






class Network:

    def __init__(self, network_shape):
        self.shape = network_shape

        # making layer classes for every layer except input
        self.layers = [Layer(network_shape[i - 1], network_shape[i], len(network_shape) - i - 1) for i in range(1, len(network_shape))]


    def forward_pass(self, input):
        inp = input
        for i in range(len(self.layers)):
            inp = self.layers[i].output(inp)
        return inp



    def backprop(self, pred, actual):
        dj_dy = MSE_grad(pred, actual)
        for layer in reversed(self.layers):
            dj_dy = layer.calculate_grad(dj_dy)


    def train(self, x_train, y_train, epoch, batch_size, learning_rate, lr1, lr2, lambda_reg, b1, b2, e):
        batch_loss = 0

        for t in range(epoch):
            t1 = t//10
            indices = np.random.choice(len(x_train), batch_size, replace=False)
            x_batch = x_train[indices]/255.0
            y_batch = y_train[indices]
            lr = clr(t, 1000, lr1, lr2, 0.9999)
            if t%100 == 0:
                print(f"{t}: {batch_loss}")
                batch_loss = 0

            pred = self.forward_pass(x_batch)
            self.backprop(pred, y_batch)
            for layer in reversed(self.layers):
                layer.apply_gradient(lr, lambda_reg, b1, b2, e, t+1)



            batch_loss += MSE(pred, y_batch)


        print()




    def test(self, x_test, y_test):
        true = 0
        false = 0
        x_test = x_test/255.0
        for layer in reversed(self.layers):
            layer.is_training = 0
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