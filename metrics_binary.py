import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from batch_generator_mod import minibatch_generator

def binary_cross_entropy_loss(y_true, y_pred):
    """Calcula Binary Cross Entropy Loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_binary_metrics(model, X, y, minibatch_size=100):
    """Calcula métricas para clasificación binaria"""
    all_predictions = []
    all_probabilities = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    # Procesar en minibatches para datasets grandes
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for features, targets in minibatch_gen:
        # Forward pass
        _, probas, _ = model.forward(features)
        predictions = (probas >= 0.5).astype(int)
        
        # Calcular loss
        loss = binary_cross_entropy_loss(targets, probas)
        total_loss += loss
        num_batches += 1
        
        all_predictions.extend(predictions)
        all_probabilities.extend(probas)
        all_targets.extend(targets)
    
    # Convertir a arrays numpy
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    avg_loss = total_loss / num_batches
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión", save_path=None):
    """Plotea matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_training_curves(train_metrics, valid_metrics, title="Curvas de Entrenamiento", save_path=None):
    """Plotea curvas de entrenamiento"""
    epochs = range(len(train_metrics['accuracy']))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(epochs, train_metrics['accuracy'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, valid_metrics['accuracy'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, train_metrics['loss'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, valid_metrics['loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Binary Cross Entropy Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score
    ax3.plot(epochs, train_metrics['f1'], 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, valid_metrics['f1'], 'r-', label='Validation', linewidth=2)
    ax3.set_title('F1 Score')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision & Recall
    ax4.plot(epochs, train_metrics['precision'], 'b-', label='Train Precision', linewidth=2)
    ax4.plot(epochs, valid_metrics['precision'], 'r-', label='Valid Precision', linewidth=2)
    ax4.plot(epochs, train_metrics['recall'], 'b--', label='Train Recall', linewidth=2)
    ax4.plot(epochs, valid_metrics['recall'], 'r--', label='Valid Recall', linewidth=2)
    ax4.set_title('Precision & Recall')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_report(y_true, y_pred, target_names=['Negativo', 'Positivo']):
    """Imprime reporte de clasificación"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print("REPORTE DE CLASIFICACIÓN")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*50}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de Confusión:")
    print(f"{'':>12} {'Pred Neg':>8} {'Pred Pos':>8}")
    print(f"{'True Neg':<12} {cm[0,0]:>8} {cm[0,1]:>8}")
    print(f"{'True Pos':<12} {cm[1,0]:>8} {cm[1,1]:>8}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }