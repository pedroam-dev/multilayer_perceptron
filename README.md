# Multilayer Perceptron (MLP) - Clasificación MNIST y Análisis de Sentimientos

Este proyecto implementa un perceptrón multicapa desde cero para dos tareas principales:
1. **Clasificación de dígitos manuscritos** del dataset MNIST (clasificación multiclase)
2. **Análisis de sentimientos** en reseñas de películas (clasificación binaria)

El proyecto incluye experimentos exhaustivos con diferentes funciones de activación, técnicas de regularización, y optimización del learning rate.

## Descripción

El proyecto incluye dos implementaciones completas de redes neuronales multicapa:

### Clasificación MNIST (Multiclase)
- **50 neuronas en la capa oculta**
- **10 neuronas de salida** (para los 10 dígitos: 0-9)
- **Funciones de activación**: ReLU, LeakyReLU, Sigmoid
- **Experimentos con estandarización** de datos
- **Learning rate adaptativo** con decaimiento lineal inverso
- **Análisis comparativo** de diferentes configuraciones

### Análisis de reseñas (Binario)
- **Arquitectura configurable** (32, 64, 128 neuronas ocultas)
- **1 neurona de salida** (positivo/negativo)
- **Procesamiento de texto** con Bag-of-Words
- **Búsqueda automática de hiperparámetros**
- **Métricas especializadas** para clasificación binaria
- **Matriz de confusión** y análisis de errores

## Estructura del proyecto

```
multilayer_perceptron/
├── 📁 Implementación MNIST (Multiclase)
│   ├── MLP.py                          # Implementación original del MLP
│   ├── MLP_mod.py                      # Versión modificada con múltiples activaciones
│   ├── activation_functions.py         # Funciones de activación originales
│   ├── activation_functions_mod.py     # Funciones de activación extendidas
│   ├── batch_generator.py             # Generador de mini-batches original
│   ├── batch_generator_mod.py         # Versión modificada
│   ├── load_dataset.py                # Carga del dataset MNIST
│   ├── metrics.py                     # Métricas originales
│   ├── metrics_mod.py                 # Métricas modificadas
│   ├── train.py                       # Script de entrenamiento original
│   ├── train_mod.py                   # Script de entrenamiento modificado
│   ├── utils.py                       # Utilidades originales
│   ├── utils_mod.py                   # Utilidades con estandarización
│   ├── run_model.py                   # Script principal de experimentos MNIST
│   └── adaptive_lr_experiments.py     # Experimentos con LR adaptativo
│
├── 📁 Análisis de Sentimientos (Binario)
│   ├── MLP_binary.py                  # MLP para clasificación binaria
│   ├── text_processor.py              # Procesamiento de texto y BOW
│   ├── metrics_binary.py              # Métricas especializadas binarias
│   ├── movie_sentiment.py             # Script principal de sentimientos
│   └── quick_sentiment_test.py        # Test rápido simplificado
│
├── 📁 Datos
│   └── dataset/
│       └── test.csv                   # Dataset de reseñas de películas
│
├── 📁 Resultados
│   └── figures/                       # Gráficas generadas
│       ├── accuracy_comparison.png    # Comparación de accuracy MNIST
│       ├── loss_comparison.png        # Comparación de loss MNIST
│       ├── lr_comparison_*.png        # Comparación de learning rates
│       ├── sentiment_training_curves.png  # Curvas de entrenamiento sentimientos
│       ├── sentiment_confusion_matrix.png # Matriz de confusión
│       └── sentiment_results.png      # Resultados completos sentimientos
│
└── __pycache__/                       # Archivos compilados de Python
```

## Experimentos realizados

### Experimento 1: Comparación de Configuraciones MNIST

El proyecto evalúa el impacto de diferentes configuraciones en el rendimiento del modelo:

| Experimento | Estandarización | Learning Rate | Función de Activación |
|------------|----------------|---------------|----------------------|
| No_std_0.1_ReLU | No | 0.1 | ReLU |
| Std_0.1_ReLU | Sí | 0.1 | ReLU |
| Std_0.5_ReLU | Sí | 0.5 | ReLU |
| Std_0.7_ReLU | Sí | 0.7 | ReLU |
| Std_0.1_LeakyReLU | Sí | 0.1 | LeakyReLU |
| Std_0.5_LeakyReLU | Sí | 0.5 | LeakyReLU |
| Std_0.7_LeakyReLU | Sí | 0.7 | LeakyReLU |

### Experimento 2: Learning Rate Adaptativo

Implementación de decaimiento lineal inverso:
```
η_t = c1 / (iteración + c2)
```

**Parámetros evaluados:**
- c1 = 0.1 (learning rate inicial)
- c2 ∈ {5, 10, 20, 50} (velocidad de decaimiento)

**Comparación:** LR estático vs LR adaptativo

### 🎬 Experimento 3: Análisis de reseñas

**Búsqueda automática de hiperparámetros:**
- **Neuronas ocultas**: [32, 64, 128]
- **Learning rate**: [0.001, 0.01, 0.1]
- **Funciones de activación**: [ReLU, LeakyReLU]
- **Tamaño de batch**: [32, 64, 128]
- **Épocas**: [50, 100]

### Parámetros generales MNIST
- **Épocas de entrenamiento**: 150
- **División de datos**: 80% entrenamiento, 20% prueba
- **Validación**: 20% del conjunto de entrenamiento
- **Tamaño de mini-batch**: 100
- **Inicialización de pesos**: Distribución normal (μ=0, σ=0.1)

## Instalación y uso

### Requisitos
```bash
pip install numpy matplotlib scikit-learn pandas seaborn
```

### Ejecución MNIST
```bash
# Ejecutar todos los experimentos MNIST
python run_model.py

# Experimentos de learning rate adaptativo
python adaptive_lr_experiments.py

# Entrenamiento básico
python train.py
```

### Ejecución análisis de sentimientos
```bash
# Test rápido (recomendado para empezar)
python quick_sentiment_test.py

# Búsqueda completa de hiperparámetros (más lento)
python movie_sentiment.py
```

## Funciones de activación implementadas

### 1. ReLU (Rectified Linear Unit)
```python
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

### 2. LeakyReLU
```python
f(x) = x if x > 0, else α*x  (α = 0.01)
f'(x) = 1 if x > 0, else α
```

### 3. Sigmoid
```python
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) * (1 - f(x))
```

## Métricas evaluadas

### MNIST (Multiclase)
- **Accuracy**: Porcentaje de predicciones correctas
- **MSE Loss**: Error cuadrático medio
- **Tiempo de entrenamiento**: Duración del proceso de entrenamiento
- **Curvas de aprendizaje**: Evolución de accuracy y loss por época
- **Learning rate evolution**: Seguimiento del LR adaptativo

### Sentimientos (Binario)
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Verdaderos positivos / (Verdaderos + Falsos positivos)
- **Recall**: Verdaderos positivos / (Verdaderos positivos + Falsos negativos)
- **F1-Score**: Media armónica de precision y recall
- **Binary Cross Entropy Loss**: Función de pérdida especializada
- **Matriz de Confusión**: Visualización de errores de clasificación

## Características técnicas

### Estandarización de satos
```python
X_standardized = (X - μ) / σ
```
Donde:
- μ = media del conjunto de entrenamiento
- σ = desviación estándar del conjunto de entrenamiento

### Learning Rate adaptativo
```python
η_t = c1 / (iteration + c2)
```
Donde:
- η_t = learning rate en la iteración t
- c1 = learning rate inicial
- c2 = controla la velocidad de decaimiento

### Arquitectura MNIST
- **Capa de entrada**: 784 neuronas (28×28 píxeles)
- **Capa oculta**: 50 neuronas con función de activación configurable
- **Capa de salida**: 10 neuronas con activación sigmoid
- **Función de pérdida**: Error cuadrático medio (MSE)

### Arquitectura reseña
- **Capa de entrada**: Variable (tamaño del vocabulario)
- **Capa oculta**: Configurable (32, 64, 128 neuronas)
- **Capa de salida**: 1 neurona con activación sigmoid
- **Función de pérdida**: Binary Cross Entropy
- **Representación**: Bag-of-Words (BOW)

### Procesamiento de texto
- **Limpieza**: Conversión a minúsculas, eliminación de caracteres especiales
- **Vocabulario**: Top 1000-5000 palabras más frecuentes
- **Tokens especiales**: `<PAD>`, `<UNK>` para padding y palabras desconocidas
- **Características**: Matriz de frecuencias de palabras

### Algoritmo de entrenamiento
- **Optimizador**: Descenso de gradiente por mini-batches
- **Backpropagation**: Cálculo de gradientes capa por capa
- **Actualización de pesos**: W = W - η∇W
- **Inicialización**: Xavier/He para mejor convergencia

## Resultados esperados

### Experimentos MNIST

Los experimentos permiten analizar:

1. **Impacto de la estandarización**: 
   - Los datos estandarizados facilitan la convergencia
   - Mejora la estabilidad del entrenamiento
   - Reducción del tiempo de convergencia

2. **Efecto del learning rate**:
   - Valores muy altos pueden causar inestabilidad
   - Valores muy bajos ralentizan la convergencia
   - LR adaptativo mejora la convergencia final

3. **Comparación de funciones de activación**:
   - ReLU: Rápida convergencia pero riesgo de "neuronas muertas"
   - LeakyReLU: Más robusta que ReLU, evita neuronas muertas
   - Sigmoid: Convergencia más lenta, riesgo de gradiente desvaneciente

4. **Learning rate adaptativo vs estático**:
   - Convergencia más suave y estable
   - Mejor accuracy final
   - Reducción de oscilaciones en el entrenamiento

### 🎬 Experimentos Sentimientos

1. **Procesamiento de texto**:
   - Importancia del tamaño del vocabulario
   - Efectividad de BOW para sentimientos
   - Limpieza de texto mejora resultados

2. **Arquitectura óptima**:
   - Número de neuronas ocultas vs overfitting
   - Balance entre complejidad y generalización
   - Funciones de activación para texto

3. **Hiperparámetros**:
   - Learning rate óptimo para clasificación binaria
   - Tamaño de batch vs estabilidad
   - Número de épocas vs convergencia

### Métricas típicas esperadas
- **MNIST**: 85-95% accuracy (dependiendo de configuración)
- **Sentimientos**: 70-85% accuracy (dependiendo de dataset y preprocesamiento)

## Archivos principales

### Implementación MNIST
- `MLP_mod.py`: Clase principal del perceptrón multicapa
- `activation_functions_mod.py`: Funciones de activación y derivadas
- `utils_mod.py`: Utilidades incluyendo estandarización y LR adaptativo
- `metrics_mod.py`: Cálculo de métricas de evaluación
- `run_model.py`: Script principal para experimentos básicos
- `adaptive_lr_experiments.py`: Experimentos con learning rate adaptativo

### Implementación sentimientos
- `MLP_binary.py`: Red neuronal para clasificación binaria
- `text_processor.py`: Procesamiento de texto y creación de BOW
- `metrics_binary.py`: Métricas especializadas para clasificación binaria
- `movie_sentiment.py`: Script principal con búsqueda de hiperparámetros
- `quick_sentiment_test.py`: Test rápido simplificado

### Scripts de soporte
- `batch_generator_mod.py`: Generación de mini-batches optimizada
- `train_mod.py`: Función de entrenamiento

## Objetivos del proyecto

1. **Implementar** un MLP desde cero usando solo NumPy
2. **Comparar** diferentes funciones de activación (ReLU, LeakyReLU, Sigmoid)
3. **Evaluar** el impacto de la estandarización de datos
4. **Analizar** el efecto de diferentes learning rates (estático vs adaptativo)
5. **Desarrollar** sistema de clasificación binaria para análisis de sentimientos
6. **Optimizar** hiperparámetros automáticamente
7. **Visualizar** curvas de aprendizaje y métricas
8. **Generar** análisis comparativo de resultados
9. **Implementar** procesamiento de texto con BOW
10. **Crear** matrices de confusión y análisis de errores

## Conceptos implementados

### Redes Neuronales
- **Forward Propagation**: Cálculo de salidas capa por capa
- **Backward Propagation**: Cálculo y propagación de gradientes
- **Mini-batch Gradient Descent**: Optimización por lotes pequeños
- **Adaptive Learning Rate**: Decaimiento lineal inverso del LR
- **Weight Initialization**: Xavier/He para mejor convergencia

### Procesamiento de datos
- **One-hot Encoding**: Codificación de etiquetas categóricas
- **Data Standardization**: Normalización estadística de features
- **Text Preprocessing**: Limpieza y tokenización de texto
- **Bag-of-Words**: Representación vectorial de documentos
- **Vocabulary Building**: Construcción de diccionarios de palabras

### Evaluación y validación
- **Cross-validation**: División de datos para validación
- **Binary Classification Metrics**: Precision, Recall, F1-Score
- **Confusion Matrix**: Análisis detallado de errores
- **Hyperparameter Search**: Búsqueda automática de configuraciones
- **Learning Curves**: Visualización del proceso de aprendizaje

### Optimización
- **Gradient Descent**: Optimización de parámetros
- **Learning Rate Scheduling**: Ajuste dinámico del LR
- **Regularization**: Técnicas para evitar overfitting
- **Early Stopping**: Prevención de sobreentrenamiento

## Contribuciones

Este proyecto fue desarrollado como parte de un estudio comparativo de técnicas de deep learning, implementando algoritmos fundamentales desde cero para comprender mejor su funcionamiento interno.

### Aspectos educativos
- **Implementación desde cero**: Sin frameworks de alto nivel para comprender los fundamentos
- **Comparación exhaustiva**: Múltiples configuraciones y enfoques
- **Visualización completa**: Gráficas y métricas detalladas
- **Análisis profundo**: Discusión de resultados y observaciones
- **Aplicación práctica**: Dos dominios diferentes (visión y texto)

### Características técnicas
- **Código modular**: Separación clara de responsabilidades
- **Experimentación sistemática**: Búsqueda estructurada de hiperparámetros
- **Reproducibilidad**: Seeds fijos para resultados consistentes
- **Documentación**: Comentarios y explicaciones detalladas

**¡Feliz experimentación con redes neuronales!**
