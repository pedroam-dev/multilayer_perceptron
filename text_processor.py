import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split

class TextProcessor:
    def __init__(self, vocab_size=5000, max_length=500):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = None
        
    def clean_text(self, text):
        """Limpia y preprocesa el texto"""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales y mantener solo letras y espacios
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        # Eliminar espacios al inicio y final
        text = text.strip()
        return text
    
    def build_vocabulary(self, texts):
        """Construye el vocabulario a partir de los textos"""
        all_words = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            all_words.extend(words)
        
        # Contar palabras y tomar las más frecuentes
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 para <UNK> y <PAD>
        
        # Crear vocabulario
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, count) in enumerate(most_common, 2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        self.vocab = list(self.word_to_idx.keys())
        print(f"Vocabulario construido con {len(self.vocab)} palabras")
    
    def text_to_sequence(self, text):
        """Convierte texto a secuencia de índices"""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        sequence = []
        for word in words[:self.max_length]:  # Truncar si es muy largo
            idx = self.word_to_idx.get(word, 1)  # 1 es <UNK>
            sequence.append(idx)
        
        # Padding si es necesario
        while len(sequence) < self.max_length:
            sequence.append(0)  # 0 es <PAD>
            
        return sequence[:self.max_length]
    
    def texts_to_sequences(self, texts):
        """Convierte múltiples textos a secuencias"""
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return np.array(sequences)
    
    def create_bow_features(self, texts):
        """Crea características de bag-of-words"""
        features = np.zeros((len(texts), len(self.vocab)))
        
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            
            for word in words:
                if word in self.word_to_idx:
                    idx = self.word_to_idx[word]
                    features[i, idx] += 1
                    
        return features

def load_movie_reviews(file_path='dataset/test.csv'):
    """Carga el dataset de reseñas de películas"""
    print(f"Cargando dataset desde: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset cargado: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        
        # Mostrar distribución de clases
        print(f"Distribución de reseñas:")
        print(df['review'].value_counts())
        
        # Convertir labels a números
        label_map = {'pos': 1, 'neg': 0}
        df['review_num'] = df['review'].map(label_map)
        
        # Verificar que no hay valores nulos
        print(f"Valores nulos en text: {df['text'].isnull().sum()}")
        print(f"Valores nulos en review: {df['review'].isnull().sum()}")
        
        return df['text'].values, df['review_num'].values, df
        
    except Exception as e:
        print(f"Error cargando el dataset: {e}")
        return None, None, None

def prepare_movie_data(vocab_size=5000, max_length=500, test_size=0.2, random_state=42):
    """Prepara los datos para entrenamiento"""
    
    # Cargar datos
    texts, labels, df = load_movie_reviews()
    
    if texts is None:
        return None
    
    print(f"\nPreprocesando {len(texts)} reseñas...")
    
    # Dividir datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, 
        stratify=labels
    )
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, 
        stratify=y_temp
    )
    
    print(f"División de datos:")
    print(f"Train: {len(X_train)} muestras")
    print(f"Valid: {len(X_valid)} muestras") 
    print(f"Test: {len(X_test)} muestras")
    
    # Crear procesador de texto
    processor = TextProcessor(vocab_size=vocab_size, max_length=max_length)
    
    # Construir vocabulario solo con datos de entrenamiento
    processor.build_vocabulary(X_train)
    
    # Crear características BOW
    print("Creando características Bag-of-Words...")
    X_train_bow = processor.create_bow_features(X_train)
    X_valid_bow = processor.create_bow_features(X_valid)
    X_test_bow = processor.create_bow_features(X_test)
    
    print(f"Características creadas:")
    print(f"Train BOW: {X_train_bow.shape}")
    print(f"Valid BOW: {X_valid_bow.shape}")
    print(f"Test BOW: {X_test_bow.shape}")
    
    return {
        'X_train': X_train_bow,
        'X_valid': X_valid_bow, 
        'X_test': X_test_bow,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'processor': processor,
        'original_data': df
    }

if __name__ == "__main__":
    # Probar el procesador
    data = prepare_movie_data()
    if data:
        print("\n=== RESUMEN ===")
        print(f"Vocabulario: {len(data['processor'].vocab)} palabras")
        print(f"Características: {data['X_train'].shape[1]}")
        print(f"Ejemplos de entrenamiento: {data['X_train'].shape[0]}")