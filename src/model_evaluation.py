import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import os
import json
from datetime import datetime
from model_baseline import BaselineModels
from model_advanced import AdvancedModels, TextDataset  # Добавлен импорт TextDataset
from torch.utils.data import Dataset, DataLoader
import lime.lime_text
import nlpaug.augmenter.word as naw
import nltk
nltk.download('wordnet')

class ModelEvaluator:
    def __init__(self):
        """Инициализация оценщика моделей"""
        self.results_dir = '../results/evaluation'
        os.makedirs(self.results_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загружаем данные и модели
        self.load_data()
        self.load_models()
        
    def load_data(self):
        """Загрузка данных"""
        print("Loading data...")
        self.X_test = pd.read_csv('../data/X_test.csv')['cleaned_text']
        self.y_test = pd.read_csv('../data/y_test.csv')['label']
        self.X_train = pd.read_csv('../data/X_train.csv')['cleaned_text']
        self.y_train = pd.read_csv('../data/y_train.csv')['label']
        print("Data loaded successfully")
        
    def load_models(self):
        """Загрузка обученных моделей"""
        print("Loading models...")
        
        # Загружаем базовую модель
        print("Loading baseline model...")
        self.baseline = BaselineModels()
        self.baseline_model = self.baseline.train_and_evaluate(
            self.X_train, 
            self.y_train,
            self.X_test,
            self.y_test
        )
        
        # Загружаем BERT
        print("Loading BERT model...")
        self.advanced = AdvancedModels()
        self.advanced.load_data()
        self.bert_model, self.tokenizer = self.advanced.train_bert()
        
        print("Models loaded successfully")

    def analyze_important_words(self):
        """Анализ важных слов с помощью LIME"""
        print("Analyzing important words...")
        
        # Создаем LIME explainer
        explainer = lime.lime_text.LimeTextExplainer(class_names=['No Stress', 'Stress'])
        
        # Создаем функцию-предсказатель, которая включает векторизацию
        def predictor(texts):
            # Векторизуем тексты
            vec_texts = self.baseline.vectorizer.transform(texts)
            # Получаем вероятности
            return self.baseline_model.predict_proba(vec_texts)
        
        # Анализируем несколько случайных примеров
        sample_indices = np.random.choice(len(self.X_test), 5)
        
        for idx in sample_indices:
            text = self.X_test.iloc[idx]
            try:
                exp = explainer.explain_instance(
                    text, 
                    predictor,
                    num_features=10
                )
                
                # Сохраняем визуализацию
                plt.figure(figsize=(10, 6))
                exp.as_pyplot_figure()
                plt.savefig(f"{self.results_dir}/lime_explanation_{idx}.png")
                plt.close()
                
                # Сохраняем важные слова
                with open(f"{self.results_dir}/important_words_{idx}.json", 'w') as f:
                    json.dump(exp.as_list(), f, indent=4)
                    
            except Exception as e:
                print(f"Error analyzing example {idx}: {str(e)}")
                continue
                
        print("Word importance analysis completed")

    def augment_data(self):
        """Аугментация данных"""
        print("Augmenting data...")
        
        # Используем только синонимы
        aug_syn = naw.SynonymAug(aug_p=0.3)
        
        augmented_texts = []
        augmented_labels = []
        
        print("Generating augmented data...")
        for idx, (text, label) in enumerate(zip(self.X_train, self.y_train)):
            if idx % 100 == 0:
                print(f"Augmenting example {idx}/{len(self.X_train)}")
                
            # Оригинальный текст
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Синонимы
            try:
                aug_text = aug_syn.augment(text)[0]
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            except:
                continue
        
        self.X_train_augmented = pd.Series(augmented_texts)
        self.y_train_augmented = pd.Series(augmented_labels)
        
        print(f"Data augmentation completed. New dataset size: {len(self.X_train_augmented)}")
        
    def create_ensemble(self):
        """Создание и оценка ансамбля моделей"""
        print("Creating ensemble predictions...")
        
        # Получаем вероятности от обеих моделей
        baseline_proba = self.baseline_model.predict_proba(
            self.baseline.vectorizer.transform(self.X_test)
        )[:, 1]
        
        bert_proba = []
        self.bert_model.eval()
        test_dataset = TextDataset(
            self.X_test,
            self.y_test,
            self.tokenizer,
            self.advanced.config['max_length']
        )
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                bert_proba.extend(probs[:, 1].cpu().numpy())
        
        # Создаем ансамбль (среднее предсказаний)
        self.ensemble_proba = (np.array(baseline_proba) + np.array(bert_proba)) / 2
        self.ensemble_preds = (self.ensemble_proba > 0.5).astype(int)
        
        # Оцениваем ансамбль
        ensemble_accuracy = accuracy_score(self.y_test, self.ensemble_preds)
        ensemble_f1 = f1_score(self.y_test, self.ensemble_preds)
        
        print("\nEnsemble Results:")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        print(f"F1-score: {ensemble_f1:.4f}")
        
        # Сохраняем результаты
        results = {
            'accuracy': float(ensemble_accuracy),
            'f1_score': float(ensemble_f1)
        }
        
        with open(f"{self.results_dir}/ensemble_results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
    def plot_confusion_matrices(self):
        """Построение матриц ошибок для всех моделей"""
        print("Plotting confusion matrices...")
        
        # Получаем предсказания базовой модели
        baseline_preds = self.baseline_model.predict(
            self.baseline.vectorizer.transform(self.X_test)
        )
        
        # Получаем предсказания BERT
        bert_preds = []
        self.bert_model.eval()
        test_dataset = TextDataset(
            self.X_test,
            self.y_test,
            self.tokenizer,
            self.advanced.config['max_length']
        )
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.bert_model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                bert_preds.extend(preds.cpu().numpy())
        
        bert_preds = np.array(bert_preds)
        
        # Строим матрицы ошибок
        models = {
            'Baseline': baseline_preds,
            'BERT': bert_preds,
            'Ensemble': self.ensemble_preds
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for (name, preds), ax in zip(models.items(), axes):
            cm = confusion_matrix(self.y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/confusion_matrices.png")
        plt.close()
        print("Confusion matrices saved")
        
    def plot_roc_curves(self):
        """Построение ROC-кривых"""
        print("Plotting ROC curves...")
        plt.figure(figsize=(10, 6))
        
        # Получаем вероятности для базовой модели
        baseline_proba = self.baseline_model.predict_proba(
            self.baseline.vectorizer.transform(self.X_test)
        )[:, 1]
        
        # Получаем вероятности для BERT
        bert_proba = []
        self.bert_model.eval()
        test_dataset = TextDataset(
            self.X_test,
            self.y_test,
            self.tokenizer,
            self.advanced.config['max_length']
        )
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                bert_proba.extend(probs[:, 1].cpu().numpy())
        
        bert_proba = np.array(bert_proba)
        
        models = {
            'Baseline': baseline_proba,
            'BERT': bert_proba,
            'Ensemble': self.ensemble_proba
        }
        
        for name, probs in models.items():
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(f"{self.results_dir}/roc_curves.png")
        plt.close()
        print("ROC curves saved")

    def evaluate(self):
        """Полная оценка и улучшение моделей"""
        print("\nStarting evaluation...")
    
        # 1. Базовые метрики
        print("\nBaseline Model Metrics:")
        baseline_preds = self.baseline_model.predict(
            self.baseline.vectorizer.transform(self.X_test)
        )
        print(f"Accuracy: {accuracy_score(self.y_test, baseline_preds):.4f}")
        print(f"F1-score: {f1_score(self.y_test, baseline_preds):.4f}")
    
        # 2. BERT метрики
        print("\nBERT Model Metrics:")
        self.bert_model.eval()
        bert_preds = []
        test_dataset = TextDataset(
            self.X_test,
            self.y_test,
            self.tokenizer,
            self.advanced.config['max_length']
        )
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.bert_model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                bert_preds.extend(preds.cpu().numpy())
        
        bert_preds = np.array(bert_preds)
        print(f"Accuracy: {accuracy_score(self.y_test, bert_preds):.4f}")
        print(f"F1-score: {f1_score(self.y_test, bert_preds):.4f}")
        
        # 3. Анализ важных слов
        print("\nAnalyzing important words...")
        self.analyze_important_words()
        
        # 4. Аугментация данных
        print("\nPerforming data augmentation...")
        self.augment_data()
        
        # 5. Создание и оценка ансамбля
        print("\nCreating ensemble...")
        self.create_ensemble()
        
        # 6. Построение визуализаций
        print("\nCreating visualizations...")
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        
        # 7. Сохранение моделей (новый код)
        print("\nSaving models...")
        os.makedirs('../results/baseline', exist_ok=True)
        os.makedirs('../results/advanced', exist_ok=True)
        self.baseline.save_model('../results/baseline/model.pkl')
        torch.save(self.bert_model.state_dict(), '../results/advanced/bert_model_best.pt')
        print("Models saved successfully")
        
        print("\nEvaluation and improvement completed.")
        print("Check results in the 'results/evaluation' directory.")


def main():
    print("Starting model evaluation and improvement...")
    evaluator = ModelEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
    