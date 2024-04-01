import numpy as np

WEIGHTS_PATH = "model/weights.npy"
BIASES_PATH = "model/biases.npy"

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def save_parameters(self):
        np.save(WEIGHTS_PATH, self.weights)
        np.save(BIASES_PATH, self.biases)

    def load_parameters(self):
        self.weights = np.load(WEIGHTS_PATH)
        self.biases = np.load(BIASES_PATH)

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
  
    def backprop(self, grad_L_out, lr):
        for i, gradient in enumerate(grad_L_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)

            grad_out_t = -t_exp[i] * t_exp / (S ** 2)
            grad_out_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            grad_t_w = self.last_input
            grad_t_b = 1
            grad_t_inputs = self.weights

            grad_L_t = gradient * grad_out_t

            grad_L_w = grad_t_w[np.newaxis].T @ grad_L_t[np.newaxis]
            grad_L_b = grad_L_t * grad_t_b
            grad_L_inputs = grad_t_inputs @ grad_L_t

            self.weights -= lr * grad_L_w
            self.biases -= lr * grad_L_b

            return grad_L_inputs.reshape(self.last_input_shape)