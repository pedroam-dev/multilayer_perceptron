import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
import os

from MLP_mod import NeuralNetMLP
from batch_generator_mod import minibatch_generator
from metrics_mod import compute_mse_and_acc
from utils_mod import standardize_data, adaptive_learning_rate

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

def train_model_static_lr(model, X_train, y_train, X_valid, y_valid, num_epochs=150,
                         learning_rate=0.1, minibatch_size=100):
    """Entrena el modelo con learning rate estático"""
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    lr_history = []
    
    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            # Compute outputs
            a_h, a_out, z_h = model.forward(X_train_mini)

            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini, z_h)

            # Update weights with static learning rate
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        # Track learning rate
        lr_history.append(learning_rate)
        
        # Epoch Logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        
        if (e + 1) % 25 == 0:
            print(f'Época: {e+1:03d}/{num_epochs:03d} '
                  f'| LR: {learning_rate:.6f} '
                  f'| Train MSE: {train_mse:.4f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc, lr_history

def train_model_adaptive_lr(model, X_train, y_train, X_valid, y_valid, num_epochs=150,
                           c1=0.1, c2=10, minibatch_size=100):
    """Entrena el modelo con learning rate adaptativo"""
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    lr_history = []
    
    for e in range(num_epochs):
        # Calculate adaptive learning rate
        current_lr = adaptive_learning_rate(e, c1, c2)
        lr_history.append(current_lr)
        
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            # Compute outputs
            a_h, a_out, z_h = model.forward(X_train_mini)

            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini, z_h)

            # Update weights with adaptive learning rate
            model.weight_h -= current_lr * d_loss__d_w_h
            model.bias_h -= current_lr * d_loss__d_b_h
            model.weight_out -= current_lr * d_loss__d_w_out
            model.bias_out -= current_lr * d_loss__d_b_out
        
        # Epoch Logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        
        if (e + 1) % 25 == 0:
            print(f'Época: {e+1:03d}/{num_epochs:03d} '
                  f'| LR: {current_lr:.6f} '
                  f'| Train MSE: {train_mse:.4f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc, lr_history

def run_lr_comparison_experiments(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """Ejecuta experimentos comparando LR estático vs adaptativo"""
    
    # Mejor configuración de experimentos anteriores
    best_config = {
        'activation': 'relu',  # o 'leaky_relu' dependiendo de los resultados
        'standardize': True,
        'static_lr': 0.1
    }
    
    print("=== EXPERIMENTOS DE LEARNING RATE ===")
    print(f"Configuración base: {best_config}")
    
    # Preparar datos (estandarizados)
    X_train_std, X_test_std, X_valid_std = standardize_data(X_train, X_test, X_valid)
    
    results = {}
    
    # Experimento 1: Learning Rate Estático
    print(f"\n=== Experimento 1: LR Estático (η = {best_config['static_lr']}) ===")
    model_static = NeuralNetMLP(
        num_features=28*28,
        num_hidden=50,
        num_classes=10,
        activation=best_config['activation']
    )
    
    start_time = time.time()
    epoch_loss_static, epoch_train_acc_static, epoch_valid_acc_static, lr_history_static = \
        train_model_static_lr(model_static, X_train_std, y_train, X_valid_std, y_valid,
                             num_epochs=150, learning_rate=best_config['static_lr'])
    training_time_static = time.time() - start_time
    
    # Evaluar en test
    test_mse_static, test_acc_static = compute_mse_and_acc(model_static, X_test_std, y_test)
    
    results['static'] = {
        'epoch_loss': epoch_loss_static,
        'epoch_train_acc': epoch_train_acc_static,
        'epoch_valid_acc': epoch_valid_acc_static,
        'lr_history': lr_history_static,
        'test_acc': test_acc_static * 100,
        'training_time': training_time_static,
        'final_train_acc': epoch_train_acc_static[-1],
        'final_valid_acc': epoch_valid_acc_static[-1]
    }
    
    print(f"Tiempo de entrenamiento: {training_time_static:.1f}s")
    print(f"Test accuracy: {test_acc_static*100:.2f}%")
    
    # Experimentos con diferentes valores de c2
    c2_values = [5, 10, 20, 50]
    c1 = 0.1
    
    for c2 in c2_values:
        print(f"\n=== Experimento: LR Adaptativo (c1={c1}, c2={c2}) ===")
        
        model_adaptive = NeuralNetMLP(
            num_features=28*28,
            num_hidden=50,
            num_classes=10,
            activation=best_config['activation']
        )
        
        start_time = time.time()
        epoch_loss_adaptive, epoch_train_acc_adaptive, epoch_valid_acc_adaptive, lr_history_adaptive = \
            train_model_adaptive_lr(model_adaptive, X_train_std, y_train, X_valid_std, y_valid,
                                   num_epochs=150, c1=c1, c2=c2)
        training_time_adaptive = time.time() - start_time
        
        # Evaluar en test
        test_mse_adaptive, test_acc_adaptive = compute_mse_and_acc(model_adaptive, X_test_std, y_test)
        
        results[f'adaptive_c2_{c2}'] = {
            'epoch_loss': epoch_loss_adaptive,
            'epoch_train_acc': epoch_train_acc_adaptive,
            'epoch_valid_acc': epoch_valid_acc_adaptive,
            'lr_history': lr_history_adaptive,
            'test_acc': test_acc_adaptive * 100,
            'training_time': training_time_adaptive,
            'final_train_acc': epoch_train_acc_adaptive[-1],
            'final_valid_acc': epoch_valid_acc_adaptive[-1],
            'c1': c1,
            'c2': c2
        }
        
        print(f"Tiempo de entrenamiento: {training_time_adaptive:.1f}s")
        print(f"Test accuracy: {test_acc_adaptive*100:.2f}%")
    
    return results

def plot_lr_comparison_results(results, save_plots=True):
    """Genera gráficas comparativas de los resultados"""
    
    # Crear directorio para figuras si no existe
    os.makedirs('figures', exist_ok=True)
    
    # 1. Gráfica de Accuracy vs Epochs
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Train Accuracy
    plt.subplot(2, 2, 1)
    for exp_name, result in results.items():
        epochs = range(len(result['epoch_train_acc']))
        if exp_name == 'static':
            plt.plot(epochs, result['epoch_train_acc'], 
                    label=f'LR Estático (η=0.1)', linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['epoch_train_acc'], 
                    label=f'LR Adaptativo (c2={c2})', linewidth=2)
    
    plt.title('Train Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Validation Accuracy
    plt.subplot(2, 2, 2)
    for exp_name, result in results.items():
        epochs = range(len(result['epoch_valid_acc']))
        if exp_name == 'static':
            plt.plot(epochs, result['epoch_valid_acc'], 
                    label=f'LR Estático (η=0.1)', linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['epoch_valid_acc'], 
                    label=f'LR Adaptativo (c2={c2})', linewidth=2)
    
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Loss
    plt.subplot(2, 2, 3)
    for exp_name, result in results.items():
        epochs = range(len(result['epoch_loss']))
        if exp_name == 'static':
            plt.plot(epochs, result['epoch_loss'], 
                    label=f'LR Estático (η=0.1)', linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['epoch_loss'], 
                    label=f'LR Adaptativo (c2={c2})', linewidth=2)
    
    plt.title('MSE Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Learning Rate Evolution
    plt.subplot(2, 2, 4)
    for exp_name, result in results.items():
        epochs = range(len(result['lr_history']))
        if exp_name == 'static':
            plt.plot(epochs, result['lr_history'], 
                    label=f'LR Estático (η=0.1)', linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['lr_history'], 
                    label=f'LR Adaptativo (c2={c2})', linewidth=2)
    
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('figures/lr_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Gráfica separada para mejor visualización de accuracy
    plt.figure(figsize=(12, 5))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    for exp_name, result in results.items():
        epochs = range(len(result['epoch_valid_acc']))
        if exp_name == 'static':
            plt.plot(epochs, result['epoch_valid_acc'], 
                    label=f'LR Estático (Test: {result["test_acc"]:.2f}%)', 
                    linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['epoch_valid_acc'], 
                    label=f'LR Adaptativo c2={c2} (Test: {result["test_acc"]:.2f}%)', 
                    linewidth=2)
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss comparison
    plt.subplot(1, 2, 2)
    for exp_name, result in results.items():
        epochs = range(len(result['epoch_loss']))
        if exp_name == 'static':
            plt.plot(epochs, result['epoch_loss'], 
                    label=f'LR Estático', linewidth=3, linestyle='--')
        else:
            c2 = result['c2']
            plt.plot(epochs, result['epoch_loss'], 
                    label=f'LR Adaptativo c2={c2}', linewidth=2)
    
    plt.title('MSE Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('figures/lr_comparison_focus.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results_table(results):
    """Imprime tabla de resultados comparativos"""
    
    print("\n" + "="*100)
    print("COMPARACIÓN DE LEARNING RATE: ESTÁTICO vs ADAPTATIVO")
    print("="*100)
    print(f"{'Configuración':<25} {'Test Acc(%)':<12} {'Final Valid(%)':<15} {'Final Train(%)':<15} {'Tiempo(s)':<12}")
    print("-"*100)
    
    for exp_name, result in results.items():
        if exp_name == 'static':
            config_name = "LR Estático (η=0.1)"
        else:
            c2 = result['c2']
            config_name = f"LR Adaptativo (c2={c2})"
        
        print(f"{config_name:<25} {result['test_acc']:<12.2f} {result['final_valid_acc']:<15.2f} "
              f"{result['final_train_acc']:<15.2f} {result['training_time']:<12.1f}")

def main():
    # Cargar datos
    print("Cargando datos MNIST...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # Ejecutar experimentos de comparación
    results = run_lr_comparison_experiments(X_train, X_valid, X_test, y_train, y_valid, y_test)
    
    # Mostrar tabla de resultados
    print_results_table(results)
    
    # Generar gráficas
    plot_lr_comparison_results(results)
    
    # Análisis de resultados
    print("\n" + "="*100)
    print("ANÁLISIS DE RESULTADOS")
    print("="*100)
    
    static_result = results['static']
    print(f"\n1. LEARNING RATE ESTÁTICO:")
    print(f"   - Test Accuracy: {static_result['test_acc']:.2f}%")
    print(f"   - Convergencia: Learning rate constante de 0.1")
    print(f"   - Tiempo de entrenamiento: {static_result['training_time']:.1f}s")
    
    print(f"\n2. LEARNING RATE ADAPTATIVO:")
    best_adaptive = max([r for k, r in results.items() if k != 'static'], 
                       key=lambda x: x['test_acc'])
    best_c2 = best_adaptive['c2']
    
    print(f"   - Mejor configuración: c2 = {best_c2}")
    print(f"   - Test Accuracy: {best_adaptive['test_acc']:.2f}%")
    print(f"   - Mejora vs estático: {best_adaptive['test_acc'] - static_result['test_acc']:.2f}%")
    print(f"   - LR inicial: {best_adaptive['lr_history'][0]:.6f}")
    print(f"   - LR final: {best_adaptive['lr_history'][-1]:.6f}")
    
    print(f"\n3. OBSERVACIONES:")
    print(f"   - El LR adaptativo {'mejora' if best_adaptive['test_acc'] > static_result['test_acc'] else 'empeora'} el rendimiento")
    print(f"   - La convergencia {'es más suave' if best_adaptive['epoch_loss'][-1] < static_result['epoch_loss'][-1] else 'es similar'}")
    print(f"   - El decaimiento ayuda a {'evitar oscilaciones' if best_adaptive['test_acc'] > static_result['test_acc'] else 'pero puede ser muy agresivo'}")
    
    return results

if __name__ == "__main__":
    results = main()