import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
  
    def backprop(self, grad_L_out, lr):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - grad_L_out is the loss gradient for this layer's outputs.
        - lr is a float.
        '''
        # We know only 1 element of grad_L_out will be nonzero
        for i, gradient in enumerate(grad_L_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            grad_out_t = -t_exp[i] * t_exp / (S ** 2)
            grad_out_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            grad_t_w = self.last_input
            grad_t_b = 1
            grad_t_inputs = self.weights

            # Gradients of loss against totals
            grad_L_t = gradient * grad_out_t

            # Gradients of loss against weights/biases/input
            grad_L_w = grad_t_w[np.newaxis].T @ grad_L_t[np.newaxis]
            grad_L_b = grad_L_t * grad_t_b
            grad_L_inputs = grad_t_inputs @ grad_L_t

            # Update weights / biases
            self.weights -= lr * grad_L_w
            self.biases -= lr * grad_L_b

            return grad_L_inputs.reshape(self.last_input_shape)