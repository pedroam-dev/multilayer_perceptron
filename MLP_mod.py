import numpy as np
from activation_functions_mod import sigmoid, relu, leaky_relu, sigmoid_derivative, relu_derivative, leaky_relu_derivative
from utils_mod import int_to_onehot

class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, activation='sigmoid', random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        self.activation = activation
        
        # hidden
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
    
    def _get_activation_function(self):
        if self.activation == 'sigmoid':
            return sigmoid
        elif self.activation == 'relu':
            return relu
        elif self.activation == 'leaky_relu':
            return leaky_relu
        else:
            raise ValueError(f"Activation {self.activation} not supported")
    
    def _get_activation_derivative(self):
        if self.activation == 'sigmoid':
            return sigmoid_derivative
        elif self.activation == 'relu':
            return relu_derivative
        elif self.activation == 'leaky_relu':
            return leaky_relu_derivative
        else:
            raise ValueError(f"Activation {self.activation} not supported")

    def forward(self, x):
        activation_func = self._get_activation_function()
        
        # Hidden layer
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = activation_func(z_h)

        # Output layer (always sigmoid for classification)
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out, z_h

    def backward(self, x, a_h, a_out, y, z_h):  
        activation_derivative = self._get_activation_derivative()
        
        # onehot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Output layer gradients
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # gradient for output weights
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        # Hidden layer gradients
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # Use the stored z_h for derivative calculation
        d_a_h__d_z_h = activation_derivative(z_h)
        
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)