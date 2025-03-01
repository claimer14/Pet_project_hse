import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import pickle
from imblearn.over_sampling import SMOTE

class BaselineModels:
    def __init__(self):
        # Создаем директорию для результатов
        self.results_dir = '../results/baseline'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Инициализация TF-IDF векторизатора
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7
        )
        
        # Инициализация классификаторов с весами классов
        self.models = {
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='linear',
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        }
        
        # Инициализация baseline_model как None
        self.baseline_model = None
    
    def balance_dataset(self, X_train, y_train):
        """Балансировка датасета"""
        print("Balancing dataset...")
        print(f"Before balancing - class distribution:\n{pd.Series(y_train).value_counts()}")
        
        # Применяем SMOTE для балансировки классов
        smote = SMOTE(random_state=42)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_resampled, y_resampled = smote.fit_resample(X_train_vectorized, y_train)
        
        print(f"After balancing - class distribution:\n{pd.Series(y_resampled).value_counts()}")
        
        return X_resampled, y_resampled
    
    def save_results(self, results, name, y_test, y_pred):
        """Сохранение результатов модели"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем метрики в JSON
        metrics_file = f"{self.results_dir}/{name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Сохраняем confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{self.results_dir}/{name}_confusion_matrix_{timestamp}.png")
        plt.close()
        
        # Сохраняем полный отчет
        report = classification_report(y_test, y_pred)
        report_file = f"{self.results_dir}/{name}_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Обучение и оценка моделей"""
        print("Starting baseline models training and evaluation...")
        results = {}
        best_model = None
        best_score = 0
        
        # Балансировка и векторизация текстов
        print("Vectorizing and balancing texts...")
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Обучение модели на сбалансированных данных
            model.fit(X_train_balanced, y_train_balanced)
            
            # Предсказания
            y_pred = model.predict(X_test_vectorized)
            
            # Оценка
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Сохраняем результаты
            model_results = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'model_name': name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Сохраняем результаты и графики
            self.save_results(model_results, name, y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'model': model
            }
            
            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-score: {f1:.4f}")
            
            # Сохраняем лучшую модель
            if f1 > best_score:
                best_score = f1
                best_model = model
                self.baseline_model = model
        
        # Убедимся, что у нас есть baseline_model
        if not hasattr(self, 'baseline_model') or self.baseline_model is None:
            self.baseline_model = best_model
        
        # Сохраняем сравнение всех моделей
        with open(f"{self.results_dir}/all_models_comparison.json", 'w') as f:
            comparison = {name: {'accuracy': res['accuracy'], 'f1_score': res['f1_score']} 
                         for name, res in results.items()}
            json.dump(comparison, f, indent=4)
        
        print("\nBest model:")
        print(f"Score: {best_score:.4f}")
        return self.baseline_model

    def save_model(self, path):
        """Сохранение модели"""
        if self.baseline_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.baseline_model,
                'vectorizer': self.vectorizer
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.baseline_model = data['model']
            self.vectorizer = data['vectorizer']
        print(f"Model loaded from {path}")

def main():
    # Загрузка данных
    print("Loading data...")
    X_train = pd.read_csv('../data/X_train.csv')['cleaned_text']
    X_test = pd.read_csv('../data/X_test.csv')['cleaned_text']
    y_train = pd.read_csv('../data/y_train.csv')['label']
    y_test = pd.read_csv('../data/y_test.csv')['label']
    
    # Создание и обучение моделей
    baseline = BaselineModels()
    best_model = baseline.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Сохраняем модель
    baseline.save_model('../results/baseline/model.pkl')
    
    print("\nBaseline models evaluation completed.")
    print("Check results in the 'results/baseline' directory.")

if __name__ == "__main__":
    main()