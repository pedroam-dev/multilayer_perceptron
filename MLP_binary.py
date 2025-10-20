import numpy as np
from activation_functions_mod import sigmoid, relu, leaky_relu, sigmoid_derivative, relu_derivative, leaky_relu_derivative

class BinaryMLP:
    def __init__(self, num_features, num_hidden, activation='relu', random_seed=123):
        super().__init__()
        
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.activation = activation
        
        # Inicialización de pesos mejorada
        rng = np.random.RandomState(random_seed)
        
        # Inicialización Xavier para capa oculta
        xavier_std_h = np.sqrt(2.0 / (num_features + num_hidden))
        self.weight_h = rng.normal(0.0, xavier_std_h, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # Inicialización Xavier para capa de salida
        xavier_std_out = np.sqrt(2.0 / (num_hidden + 1))
        self.weight_out = rng.normal(0.0, xavier_std_out, size=num_hidden)
        self.bias_out = 0.0
    
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
        """Forward pass para clasificación binaria"""
        activation_func = self._get_activation_function()
        
        # Hidden layer
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = activation_func(z_h)

        # Output layer - una sola neurona con sigmoid para clasificación binaria
        z_out = np.dot(a_h, self.weight_out) + self.bias_out
        a_out = sigmoid(z_out)
        
        return a_h, a_out, z_h

    def backward(self, x, a_h, a_out, y, z_h):
        """Backpropagation para clasificación binaria"""
        activation_derivative = self._get_activation_derivative()
        
        # Output layer gradients - Binary Cross Entropy
        # Para evitar log(0), añadimos pequeño epsilon
        epsilon = 1e-15
        a_out = np.clip(a_out, epsilon, 1 - epsilon)
        
        # Gradiente de Binary Cross Entropy Loss
        d_loss__d_a_out = -(y / a_out - (1 - y) / (1 - a_out)) / len(y)
        
        # Gradiente de sigmoid
        d_a_out__d_z_out = a_out * (1 - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # Gradientes para pesos de salida
        d_loss__dw_out = np.dot(delta_out, a_h)
        d_loss__db_out = np.sum(delta_out)
        
        # Hidden layer gradients
        d_loss__a_h = delta_out[:, np.newaxis] * self.weight_out
        d_a_h__d_z_h = activation_derivative(z_h)
        
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, x)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h

    def predict_proba(self, X):
        """Predice probabilidades"""
        _, proba, _ = self.forward(X)
        return proba
    
    def predict(self, X):
        """Predice clases binarias"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)