import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display
from tqdm import tqdm

# Configurações de estilo
plt.style.use('ggplot')
sns.set_palette("viridis")
%matplotlib inline

class WeedDetector:
    def __init__(self):
        self.paths = {
            'image': '../datasets/Pinas/AOI_img.tif',
            'soil': '../datasets/Pinas/Solo.shp',
            'vegetation': '../datasets/Pinas/Veg.shp',
            'weeds': '../datasets/Pinas/Invasoras.shp'
        }
        self.class_names = {0: 'Solo', 1: 'Vegetação', 2: 'Ervas Daninhas'}
        self.class_colors = {0: 'red', 1: 'green', 2: 'yellow'}
        
    def load_and_prepare_data(self):
        """Carrega e prepara os dados iniciais"""
        print("⏳ Carregando dados...")
        
        # Carrega shapefiles
        self.gdfs = {
            'soil': gpd.read_file(self.paths['soil']),
            'vegetation': gpd.read_file(self.paths['vegetation']),
            'weeds': gpd.read_file(self.paths['weeds'])
        }
        
        # Visualização inicial
        self.plot_original_data()
        
        # Prepara dados para modelo
        self.prepare_training_data()
        
    def plot_original_data(self):
        """Plota a imagem original com as classes sobrepostas"""
        print("\n📊 Visualizando dados originais...")
        
        fig, ax = plt.subplots(figsize=(15, 15))
        with rasterio.open(self.paths['image']) as src:
            # Ajusta CRS dos shapefiles
            for name, gdf in self.gdfs.items():
                self.gdfs[name] = gdf.to_crs(src.crs.to_dict())
            
            # Mostra imagem de fundo
            show(src, ax=ax, title='Imagem de Satélite com Classes Sobrepostas')
            
            # Plota cada classe
            self.gdfs['soil'].plot(ax=ax, color='red', label='Solo')
            self.gdfs['vegetation'].plot(ax=ax, color='green', label='Vegetação')
            self.gdfs['weeds'].plot(ax=ax, color='yellow', label='Ervas Daninhas')
            
            ax.legend(prop={'size': 12})
            plt.tight_layout()
            plt.show()
    
    def prepare_training_data(self):
        """Prepara os dados para treinamento"""
        print("\n🔧 Preparando dados para treinamento...")
        
        # Adiciona IDs de classe
        self.gdfs['soil']['class_id'] = 0
        self.gdfs['vegetation']['class_id'] = 1
        self.gdfs['weeds']['class_id'] = 2
        
        # Combina todos os dados
        combined_gdf = pd.concat([self.gdfs['soil'], 
                                self.gdfs['vegetation'], 
                                self.gdfs['weeds']], axis=0)
        
        # Extrai coordenadas
        coords = [(x, y) for x, y in zip(combined_gdf.geometry.x, combined_gdf.geometry.y)]
        
        # Extrai valores dos pixels
        with rasterio.open(self.paths['image']) as src:
            pixel_values = np.array([x for x in src.sample(coords)])
        
        # Separa features (X) e labels (Y)
        self.X = pixel_values[:, 0:3]  # Usa apenas RGB
        self.Y = combined_gdf['class_id'].values.reshape(-1, 1)
        
        # Visualização da distribuição das classes
        self.plot_class_distribution(combined_gdf['class_id'])
        
        # One-hot encoding
        encoder = OneHotEncoder()
        self.Y_encoded = encoder.fit_transform(self.Y).toarray()
        
        # Split treino/teste
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y_encoded, test_size=0.3, random_state=42)
        
    def plot_class_distribution(self, class_ids):
        """Plota a distribuição das classes"""
        print("\n📈 Distribuição das classes:")
        
        counts = pd.Series(class_ids).value_counts().sort_index()
        counts.index = counts.index.map(self.class_names.get)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind='bar', ax=ax, color=[self.class_colors[i] for i in sorted(self.class_names)])
        ax.set_title('Distribuição das Classes no Dataset')
        ax.set_ylabel('Número de Amostras')
        ax.set_xlabel('Classe')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        display(counts)
    
    def build_model(self):
        """Constrói e compila o modelo"""
        print("\n🧠 Construindo modelo...")
        
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(8, activation='relu'),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model.summary()
    
    def train_and_evaluate(self):
        """Treina e avalia o modelo"""
        print("\n🏋️ Treinando modelo...")
        
        history = self.model.fit(
            self.X_train, self.Y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Plota histórico de treinamento
        self.plot_training_history(history)
        
        # Avaliação no conjunto de teste
        print("\n🧪 Avaliando modelo no conjunto de teste...")
        test_loss, test_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Matriz de confusão e relatório
        self.evaluate_model()
    
    def plot_training_history(self, history):
        """Plota o histórico de treinamento"""
        print("\n📉 Histórico de Treinamento:")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Gera matriz de confusão e relatório de classificação"""
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.Y_test, axis=1)
        
        # Relatório de classificação
        print("\n📝 Relatório de Classificação:")
        print(classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.class_names.values()))
        
        # Matriz de confusão
        print("\n🔢 Matriz de Confusão:")
        self.plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        # Kappa score
        kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
        print(f"\n📊 Cohen's Kappa Score: {kappa:.4f}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plota a matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=self.class_names.values(), 
                            columns=self.class_names.values())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                   annot_kws={"size": 16}, cbar=False)
        plt.title('Matriz de Confusão', fontsize=14)
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict_full_image(self):
        """Classifica a imagem completa"""
        print("\n🌍 Classificando imagem completa...")
        
        with rasterio.open(self.paths['image']) as src:
            # Lê a imagem
            img = src.read()
            meta = src.meta.copy()
            
            # Redimensiona para (height × width × bands)
            img = img.transpose([1, 2, 0])
            original_shape = img.shape
            
            # Filtra apenas pixels válidos (mask == 255)
            valid_mask = img[:, :, 3] == 255
            valid_pixels = img[valid_mask][:, 0:3]  # Apenas RGB
            
            # Faz as predições em lotes para economizar memória
            print(f"🔍 Predizendo {len(valid_pixels)} pixels...")
            batch_size = 10000
            predictions = []
            
            for i in tqdm(range(0, len(valid_pixels), batch_size)):
                batch = valid_pixels[i:i + batch_size]
                pred = self.model.predict(batch, verbose=0)
                predictions.append(np.argmax(pred, axis=1))
            
            predictions = np.concatenate(predictions)
            
            # Cria imagem classificada
            classified = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
            classified[valid_mask] = predictions
            
            # Visualização do resultado
            self.plot_classification_result(classified)
            
            # Salva o resultado
            self.save_classified_image(classified, meta)
    
    def plot_classification_result(self, classified):
        """Plota o resultado da classificação"""
        print("\n🎨 Visualizando resultado da classificação...")
        
        plt.figure(figsize=(15, 15))
        
        # Cria imagem colorida baseada nas classes
        colored = np.zeros((*classified.shape, 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            mask = classified == class_id
            if color == 'red':
                colored[mask] = [255, 0, 0]
            elif color == 'green':
                colored[mask] = [0, 255, 0]
            elif color == 'yellow':
                colored[mask] = [255, 255, 0]
        
        plt.imshow(colored)
        plt.title('Mapa de Classificação', fontsize=16)
        
        # Cria legenda
        patches = [plt.Rectangle((0,0),1,1, color=np.array([c])/255) 
                  for c in [[255,0,0], [0,255,0], [255,255,0]]]
        plt.legend(patches, self.class_names.values(), loc='upper right', fontsize=12)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_classified_image(self, classified, meta):
        """Salva a imagem classificada"""
        print("\n💾 Salvando imagem classificada...")
        
        # Atualiza metadados
        meta.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": 'uint8',
            "nodata": None,
            "compress": 'lzw'
        })
        
        # Salva como GeoTIFF
        with rasterio.open('/content/mapa_classificado.tif', 'w', **meta) as dst:
            dst.write(classified[np.newaxis, :, :])
        
        print("✅ Imagem classificada salva como 'mapa_classificado.tif'")

# Execução principal
if __name__ == "__main__":
    detector = WeedDetector()
    
    # Fluxo completo
    detector.load_and_prepare_data()
    detector.build_model()
    detector.train_and_evaluate()
    detector.predict_full_image()