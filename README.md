# Multilayer Perceptron (MLP) para Clasificación MNIST

Este proyecto implementa un perceptrón multicapa desde cero para la clasificación de dígitos manuscritos del dataset MNIST, con soporte para diferentes funciones de activación y experimentos de estandarización de datos.

## Descripción

El proyecto incluye una implementación completa de una red neuronal multicapa con:
- **50 neuronas en la capa oculta**
- **10 neuronas de salida** (para los 10 dígitos: 0-9)
- **Funciones de activación**: ReLU, LeakyReLU, Sigmoid
- **Experimentos con estandarización** de datos
- **Análisis comparativo** de diferentes configuraciones

## Estructura del proyecto

```
multilayer_perceptron/
├── MLP.py                          # Implementación original del MLP
├── MLP_mod.py                      # Versión modificada con múltiples activaciones
├── activation_functions.py         # Funciones de activación originales
├── activation_functions_mod.py     # Funciones de activación extendidas
├── batch_generator.py             # Generador de mini-batches original
├── batch_generator_mod.py         # Versión modificada
├── load_dataset.py                # Carga del dataset MNIST
├── metrics.py                     # Métricas originales
├── metrics_mod.py                 # Métricas modificadas
├── train.py                       # Script de entrenamiento original
├── train_mod.py                   # Script de entrenamiento modificado
├── utils.py                       # Utilidades originales
├── utils_mod.py                   # Utilidades con estandarización
├── run_model.py                   # Script principal de experimentos
├── dataset/                       # Datos del dataset
│   └── test.csv
├── figures/                       # Gráficas generadas
└── __pycache__/                   # Archivos compilados de Python
```

## Experimentos realizados

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

### Parámetros de configuración
- **Épocas de entrenamiento**: 150
- **División de datos**: 80% entrenamiento, 20% prueba
- **Validación**: 20% del conjunto de entrenamiento
- **Tamaño de mini-batch**: 100
- **Inicialización de pesos**: Distribución normal (μ=0, σ=0.1)

## Instalación y uso

### Requisitos
```bash
pip install numpy matplotlib scikit-learn
```

### Ejecución
```bash
# Ejecutar todos los experimentos
python run_model.py

# Ejecutar entrenamiento básico
python train.py
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

- **Accuracy**: Porcentaje de predicciones correctas
- **MSE Loss**: Error cuadrático medio
- **Tiempo de entrenamiento**: Duración del proceso de entrenamiento
- **Curvas de aprendizaje**: Evolución de accuracy y loss por época

## Características técnicas

### Estandarización de datos
```python
X_standardized = (X - μ) / σ
```
Donde:
- μ = media del conjunto de entrenamiento
- σ = desviación estándar del conjunto de entrenamiento

### Arquitectura de la red
- **Capa de entrada**: 784 neuronas (28×28 píxeles)
- **Capa oculta**: 50 neuronas con función de activación configurable
- **Capa de salida**: 10 neuronas con activación sigmoid
- **Función de pérdida**: Error cuadrático medio (MSE)

### Algoritmo de entrenamiento
- **Optimizador**: Descenso de gradiente por mini-batches
- **Backpropagation**: Cálculo de gradientes capa por capa
- **Actualización de pesos**: W = W - η∇W

## Resultados esperados

Los experimentos permiten analizar:

1. **Impacto de la estandarización**: 
   - Los datos estandarizados facilitan la convergencia
   - Mejora la estabilidad del entrenamiento

2. **Efecto del learning rate**:
   - Valores muy altos pueden causar inestabilidad
   - Valores muy bajos ralentizan la convergencia

3. **Comparación de funciones de activación**:
   - ReLU: Rápida convergencia pero riesgo de "neuronas muertas"
   - LeakyReLU: Más robusta que ReLU
   - Sigmoid: Convergencia más lenta, riesgo de gradiente desvaneciente

## Archivos principales

### Implementación core
- `MLP_mod.py`: Clase principal del perceptrón multicapa
- `activation_functions_mod.py`: Funciones de activación y derivadas
- `utils_mod.py`: Utilidades incluyendo estandarización
- `metrics_mod.py`: Cálculo de métricas de evaluación

### Scripts de ejecución
- `run_model.py`: Script principal para todos los experimentos
- `train_mod.py`: Función de entrenamiento
- `batch_generator_mod.py`: Generación de mini-batches

## Objetivos del proyecto

1. **Implementar** un MLP desde cero usando solo NumPy
2. **Comparar** diferentes funciones de activación
3. **Evaluar** el impacto de la estandarización de datos
4. **Analizar** el efecto de diferentes learning rates
5. **Visualizar** curvas de aprendizaje y métricas
6. **Generar** análisis comparativo de resultados

## Conceptos implementados

- **Forward Propagation**: Cálculo de salidas capa por capa
- **Backward Propagation**: Cálculo y propagación de gradientes
- **Mini-batch Gradient Descent**: Optimización por lotes pequeños
- **One-hot Encoding**: Codificación de etiquetas categóricas
- **Data Standardization**: Normalización estadística de features
- **Cross-validation**: División de datos para validación

## Análisis y discusión

Los resultados obtenidos permiten entender:
- La importancia del preprocesamiento de datos
- El rol de las funciones de activación en el aprendizaje
- La influencia del learning rate en la convergencia
- Las diferencias de rendimiento entre arquitecturas

## Contribuciones

Este proyecto fue desarrollado como parte de un estudio comparativo de técnicas de deep learning, implementando algoritmos fundamentales desde cero para comprender mejor su funcionamiento interno.
