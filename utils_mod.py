import numpy as np

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

def standardize_data(X_train, X_test=None, X_valid=None):
    """
    Estandariza los datos para tener media 0 y desviación estándar 1
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Evitar división por cero
    
    X_train_std = (X_train - mean) / std
    
    results = [X_train_std]
    if X_test is not None:
        X_test_std = (X_test - mean) / std
        results.append(X_test_std)
    if X_valid is not None:
        X_valid_std = (X_valid - mean) / std
        results.append(X_valid_std)
    
    if len(results) == 1:
        return results[0]
    return tuple(results)