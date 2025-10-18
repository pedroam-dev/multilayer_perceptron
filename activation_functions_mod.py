import numpy as np

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))  # Clip para evitar overflow

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)