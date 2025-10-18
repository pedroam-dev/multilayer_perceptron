import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time

from MLP_mod import NeuralNetMLP
from batch_generator_mod import minibatch_generator
from metrics_mod import compute_mse_and_acc
from utils_mod import standardize_data

def load_and_split_data():
    """Carga y divide los datos MNIST"""
    print("Descargando datos MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.values.astype(np.float32)
    y = y.astype(int).values
    
    # División 80%-20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=123, stratify=y_temp)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model(model, X_train, y_train, X_valid, y_valid, num_epochs=150,
                learning_rate=0.1, minibatch_size=100):
    """Entrena el modelo y devuelve las métricas por época"""
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            # Compute outputs
            a_h, a_out, z_h = model.forward(X_train_mini)

            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini, z_h)

            # Update weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        # Epoch Logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        
        if (e + 1) % 25 == 0:
            print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                  f'| Train MSE: {train_mse:.4f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc

def run_experiment(experiment_config, X_train, X_valid, X_test, y_train, y_valid, y_test):
    """Ejecuta un experimento específico"""
    
    print(f"\n=== Experimento: {experiment_config['name']} ===")
    print(f"Estandarización: {experiment_config['standardize']}")
    print(f"Learning rate: {experiment_config['learning_rate']}")
    print(f"Activación: {experiment_config['activation']}")
    
    # Preparar datos
    if experiment_config['standardize']:
        X_train_exp, X_test_exp, X_valid_exp = standardize_data(X_train, X_test, X_valid)
        print("Datos estandarizados aplicados")
    else:
        X_train_exp, X_valid_exp, X_test_exp = X_train/255.0, X_valid/255.0, X_test/255.0
        print("Normalización [0,1] aplicada")
    
    # Crear modelo
    model = NeuralNetMLP(
        num_features=28*28,
        num_hidden=50,
        num_classes=10,
        activation=experiment_config['activation']
    )
    
    # Entrenar
    start_time = time.time()
    epoch_loss, epoch_train_acc, epoch_valid_acc = train_model(
        model, X_train_exp, y_train, X_valid_exp, y_valid,
        num_epochs=150, learning_rate=experiment_config['learning_rate']
    )
    training_time = time.time() - start_time
    
    # Evaluar en test
    test_mse, test_acc = compute_mse_and_acc(model, X_test_exp, y_test)
    
    print(f"Tiempo de entrenamiento: {training_time:.1f}s")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    results = {
        'epoch_loss': epoch_loss,
        'epoch_train_acc': epoch_train_acc,
        'epoch_valid_acc': epoch_valid_acc,
        'final_train_acc': epoch_train_acc[-1],
        'final_valid_acc': epoch_valid_acc[-1],
        'test_acc': test_acc * 100,
        'training_time': training_time
    }
    
    return results

def plot_results(experiments_results, save_plots=True):
    """Genera gráficas de los resultados"""
    
    # Plot accuracy
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (config, results) in enumerate(experiments_results.items()):
        if i < len(axes):
            ax = axes[i]
            epochs = range(len(results['epoch_train_acc']))
            ax.plot(epochs, results['epoch_train_acc'], label='Train', linewidth=2)
            ax.plot(epochs, results['epoch_valid_acc'], label='Validation', linewidth=2)
            ax.set_title(f"{config}\nTest Acc: {results['test_acc']:.2f}%")
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Ocultar axes no usados
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot loss
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (config, results) in enumerate(experiments_results.items()):
        if i < len(axes):
            ax = axes[i]
            epochs = range(len(results['epoch_loss']))
            ax.plot(epochs, results['epoch_loss'], linewidth=2, color='red')
            ax.set_title(f"{config}")
            ax.set_xlabel('Epochs')
            ax.set_ylabel('MSE Loss')
            ax.grid(True, alpha=0.3)
    
    # Ocultar axes no usados
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('figures/loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Cargar datos
    print("Cargando datos MNIST...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # Configuración de experimentos
    experiments = {
        'No_std_0.1_ReLU': {
            'name': 'Sin estandarización, LR=0.1, ReLU',
            'standardize': False,
            'learning_rate': 0.1,
            'activation': 'relu'
        },
        'Std_0.1_ReLU': {
            'name': 'Con estandarización, LR=0.1, ReLU', 
            'standardize': True,
            'learning_rate': 0.1,
            'activation': 'relu'
        },
        'Std_0.5_ReLU': {
            'name': 'Con estandarización, LR=0.5, ReLU',
            'standardize': True,
            'learning_rate': 0.5,
            'activation': 'relu'
        },
        'Std_0.7_ReLU': {
            'name': 'Con estandarización, LR=0.7, ReLU',
            'standardize': True,
            'learning_rate': 0.7,
            'activation': 'relu'
        },
        'Std_0.1_LeakyReLU': {
            'name': 'Con estandarización, LR=0.1, LeakyReLU',
            'standardize': True,
            'learning_rate': 0.1,
            'activation': 'leaky_relu'
        },
        'Std_0.5_LeakyReLU': {
            'name': 'Con estandarización, LR=0.5, LeakyReLU',
            'standardize': True,
            'learning_rate': 0.5,
            'activation': 'leaky_relu'
        },
        'Std_0.7_LeakyReLU': {
            'name': 'Con estandarización, LR=0.7, LeakyReLU',
            'standardize': True,
            'learning_rate': 0.7,
            'activation': 'leaky_relu'
        }
    }
    
    # Ejecutar experimentos
    results = {}
    for exp_name, config in experiments.items():
        try:
            results[exp_name] = run_experiment(config, X_train, X_valid, X_test, 
                                             y_train, y_valid, y_test)
        except Exception as e:
            print(f"Error en experimento {exp_name}: {e}")
            continue
    
    # Generar tabla de resultados
    print("\n" + "="*100)
    print("RESULTADOS FINALES")
    print("="*100)
    print(f"{'Experimento':<25} {'Test Acc(%)':<12} {'Tiempo(s)':<12} {'Std':<8} {'LR':<6} {'Activación':<12}")
    print("-"*100)
    
    for exp_name, result in results.items():
        config = experiments[exp_name]
        print(f"{exp_name:<25} {result['test_acc']:<12.2f} {result['training_time']:<12.1f} "
              f"{'Sí' if config['standardize'] else 'No':<8} "
              f"{config['learning_rate']:<6} {config['activation']:<12}")
    
    # Generar gráficas
    if results:
        plot_results(results)
    
    return results

if __name__ == "__main__":
    results = main()