import numpy as np
import matplotlib.pyplot as plt
import time
import os
from itertools import product

from text_processor import prepare_movie_data
from MLP_binary import BinaryMLP
from batch_generator_mod import minibatch_generator
from metrics_binary import compute_binary_metrics, plot_confusion_matrix, plot_training_curves, print_classification_report
from utils_mod import standardize_data

def train_binary_model(model, X_train, y_train, X_valid, y_valid, 
                      num_epochs=100, learning_rate=0.01, minibatch_size=32):
    """Entrena modelo para clasificación binaria"""
    
    train_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'loss': []
    }
    valid_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'loss': []
    }
    
    print(f"Entrenando por {num_epochs} épocas...")
    
    for epoch in range(num_epochs):
        # Entrenamiento
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        
        for X_batch, y_batch in minibatch_gen:
            # Forward pass
            a_h, a_out, z_h = model.forward(X_batch)
            
            # Backward pass
            dw_out, db_out, dw_h, db_h = model.backward(X_batch, a_h, a_out, y_batch, z_h)
            
            # Update weights
            model.weight_out -= learning_rate * dw_out
            model.bias_out -= learning_rate * db_out
            model.weight_h -= learning_rate * dw_h
            model.bias_h -= learning_rate * db_h
        
        # Calcular métricas
        train_results = compute_binary_metrics(model, X_train, y_train)
        valid_results = compute_binary_metrics(model, X_valid, y_valid)
        
        # Guardar métricas
        for metric in train_metrics.keys():
            train_metrics[metric].append(train_results[metric])
            valid_metrics[metric].append(valid_results[metric])
        
        # Log progreso
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1:3d}/{num_epochs} | "
                  f"Train Acc: {train_results['accuracy']:.4f} | "
                  f"Valid Acc: {valid_results['accuracy']:.4f} | "
                  f"Valid Loss: {valid_results['loss']:.4f}")
    
    return train_metrics, valid_metrics

def run_hyperparameter_search(data):
    """Búsqueda de hiperparámetros"""
    
    X_train, X_valid, X_test = data['X_train'], data['X_valid'], data['X_test']
    y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']
    
    # Definir hiperparámetros a probar
    hyperparams = {
        'num_hidden': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
        'activation': ['relu', 'leaky_relu'],
        'minibatch_size': [32, 64, 128],
        'num_epochs': [50, 100]
    }
    
    print("=== BÚSQUEDA DE HIPERPARÁMETROS ===")
    print(f"Probando {np.prod([len(v) for v in hyperparams.values()])} combinaciones...")
    
    best_score = 0
    best_params = {}
    best_model = None
    results = []
    
    # Estandarizar datos
    X_train_std, X_valid_std, X_test_std = standardize_data(X_train, X_valid, X_test)
    
    counter = 0
    total_combinations = np.prod([len(v) for v in hyperparams.values()])
    
    for params in product(*hyperparams.values()):
        counter += 1
        param_dict = dict(zip(hyperparams.keys(), params))
        
        print(f"\n[{counter}/{total_combinations}] Probando: {param_dict}")
        
        try:
            # Crear modelo
            model = BinaryMLP(
                num_features=X_train_std.shape[1],
                num_hidden=param_dict['num_hidden'],
                activation=param_dict['activation']
            )
            
            # Entrenar
            start_time = time.time()
            train_metrics, valid_metrics = train_binary_model(
                model, X_train_std, y_train, X_valid_std, y_valid,
                num_epochs=param_dict['num_epochs'],
                learning_rate=param_dict['learning_rate'],
                minibatch_size=param_dict['minibatch_size']
            )
            training_time = time.time() - start_time
            
            # Evaluar
            final_score = valid_metrics['f1'][-1]
            
            result = {
                'params': param_dict.copy(),
                'score': final_score,
                'training_time': training_time,
                'final_accuracy': valid_metrics['accuracy'][-1],
                'final_loss': valid_metrics['loss'][-1]
            }
            results.append(result)
            
            print(f"F1-Score: {final_score:.4f}, Tiempo: {training_time:.1f}s")
            
            # Actualizar mejor modelo
            if final_score > best_score:
                best_score = final_score
                best_params = param_dict.copy()
                best_model = model
                print(f"*** NUEVO MEJOR MODELO: F1={final_score:.4f} ***")
        
        except Exception as e:
            print(f"Error en experimento: {e}")
            continue
    
    return best_model, best_params, best_score, results

def run_final_experiment(data, best_params):
    """Ejecuta experimento final con mejores parámetros"""
    
    print("\n=== EXPERIMENTO FINAL CON MEJORES PARÁMETROS ===")
    print(f"Parámetros: {best_params}")
    
    X_train, X_valid, X_test = data['X_train'], data['X_valid'], data['X_test']
    y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']
    
    # Estandarizar datos
    X_train_std, X_valid_std, X_test_std = standardize_data(X_train, X_valid, X_test)
    
    # Crear modelo final
    model = BinaryMLP(
        num_features=X_train_std.shape[1],
        num_hidden=best_params['num_hidden'],
        activation=best_params['activation']
    )
    
    # Entrenar modelo final con más épocas
    final_epochs = max(100, best_params['num_epochs'])
    print(f"Entrenando modelo final por {final_epochs} épocas...")
    
    train_metrics, valid_metrics = train_binary_model(
        model, X_train_std, y_train, X_valid_std, y_valid,
        num_epochs=final_epochs,
        learning_rate=best_params['learning_rate'],
        minibatch_size=best_params['minibatch_size']
    )
    
    # Evaluar en conjunto de test
    print("\nEvaluando en conjunto de test...")
    test_results = compute_binary_metrics(model, X_test_std, y_test)
    
    return model, train_metrics, valid_metrics, test_results

def analyze_results(data, model, train_metrics, valid_metrics, test_results, best_params):
    """Analiza y visualiza resultados"""
    
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Crear directorio para figuras
    os.makedirs('figures', exist_ok=True)
    
    # 1. Gráficas de entrenamiento
    plot_training_curves(
        train_metrics, valid_metrics,
        title=f"Entrenamiento - {best_params['activation']} - {best_params['num_hidden']} neuronas",
        save_path='figures/review_training_curves.png'
    )
    
    # 2. Matriz de confusión en test
    plot_confusion_matrix(
        test_results['targets'], test_results['predictions'],
        title="Matriz de Confusión - Conjunto de Test",
        save_path='figures/review_confusion_matrix.png'
    )
    
    # 3. Reporte de clasificación
    print_classification_report(test_results['targets'], test_results['predictions'])
    
    # 4. Análisis de predicciones
    print("\n=== ANÁLISIS DE PREDICCIONES ===")
    
    # Ejemplos bien clasificados
    correct_predictions = test_results['targets'] == test_results['predictions']
    correct_indices = np.where(correct_predictions)[0]
    
    # Ejemplos mal clasificados
    incorrect_indices = np.where(~correct_predictions)[0]
    
    print(f"Predicciones correctas: {np.sum(correct_predictions)}/{len(test_results['targets'])}")
    print(f"Predicciones incorrectas: {len(incorrect_indices)}")
    
    # Mostrar algunos ejemplos del dataset original
    if len(incorrect_indices) > 0:
        print(f"\nEjemplos mal clasificados (primeros 3):")
        original_data = data['original_data']
        test_indices = np.arange(len(data['y_test']))
        
        for i, idx in enumerate(incorrect_indices[:3]):
            actual_idx = test_indices[idx]
            if actual_idx < len(original_data):
                print(f"\nEjemplo {i+1}:")
                print(f"Texto: {original_data.iloc[actual_idx]['text'][:200]}...")
                print(f"Verdadero: {'Positivo' if test_results['targets'][idx] == 1 else 'Negativo'}")
                print(f"Predicho: {'Positivo' if test_results['predictions'][idx] == 1 else 'Negativo'}")
                print(f"Probabilidad: {test_results['probabilities'][idx]:.4f}")
    
    return {
        'final_accuracy': test_results['accuracy'],
        'final_f1': test_results['f1'],
        'final_precision': test_results['precision'],
        'final_recall': test_results['recall'],
        'confusion_matrix': (test_results['targets'], test_results['predictions'])
    }

def print_hyperparameter_results(results):
    """Imprime resultados de búsqueda de hiperparámetros"""
    
    print("\n=== RESULTADOS DE BÚSQUEDA DE HIPERPARÁMETROS ===")
    
    # Ordenar por F1-score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print(f"{'Rank':<4} {'F1':<6} {'Acc':<6} {'Loss':<6} {'Hidden':<7} {'LR':<6} {'Act':<10} {'Batch':<6} {'Epochs':<7}")
    print("-" * 80)
    
    for i, result in enumerate(results_sorted[:10]):  # Top 10
        params = result['params']
        print(f"{i+1:<4} {result['score']:<6.4f} {result['final_accuracy']:<6.4f} "
              f"{result['final_loss']:<6.4f} {params['num_hidden']:<7} {params['learning_rate']:<6.3f} "
              f"{params['activation']:<10} {params['minibatch_size']:<6} {params['num_epochs']:<7}")

def main():
    """Función principal"""
    
    print("=== CLASIFICACIÓN DE SENTIMIENTOS EN RESEÑAS DE PELÍCULAS ===")
    
    # 1. Cargar y preparar datos
    print("\n1. Cargando y preparando datos...")
    data = prepare_movie_data(vocab_size=3000, max_length=400)
    
    if data is None:
        print("Error cargando datos. Terminando ejecución.")
        return
    
    print(f"Datos preparados:")
    print(f"- Características: {data['X_train'].shape[1]}")
    print(f"- Train: {data['X_train'].shape[0]} muestras")
    print(f"- Valid: {data['X_valid'].shape[0]} muestras")  
    print(f"- Test: {data['X_test'].shape[0]} muestras")
    
    # 2. Búsqueda de hiperparámetros
    print("\n2. Búsqueda de hiperparámetros...")
    best_model, best_params, best_score, all_results = run_hyperparameter_search(data)
    
    print(f"\n*** MEJORES PARÁMETROS ENCONTRADOS ***")
    print(f"F1-Score: {best_score:.4f}")
    print(f"Parámetros: {best_params}")
    
    # 3. Imprimir resultados de búsqueda
    print_hyperparameter_results(all_results)
    
    # 4. Experimento final
    print("\n3. Experimento final...")
    final_model, train_metrics, valid_metrics, test_results = run_final_experiment(data, best_params)
    
    # 5. Análisis y visualización
    print("\n4. Análisis de resultados...")
    final_results = analyze_results(
        data, final_model, train_metrics, valid_metrics, test_results, best_params
    )
    
    # 6. Resumen final
    print("\n=== RESUMEN FINAL ===")
    print(f"Mejores hiperparámetros: {best_params}")
    print(f"Accuracy en test: {final_results['final_accuracy']:.4f}")
    print(f"F1-Score en test: {final_results['final_f1']:.4f}")
    print(f"Precision en test: {final_results['final_precision']:.4f}")
    print(f"Recall en test: {final_results['final_recall']:.4f}")
    
    return {
        'model': final_model,
        'best_params': best_params,
        'results': final_results,
        'data': data
    }

if __name__ == "__main__":
    results = main()