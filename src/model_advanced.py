import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.tokenizer:
            # Для BERT
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Для Word2Vec + LSTM
            return text, torch.tensor(label, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Берем только последний выход
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

class AdvancedModels:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = '../results/advanced'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Гиперпараметры
        self.config = {
            'max_length': 128,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'epochs': 5,
            'lstm_hidden_dim': 256,
            'embedding_dim': 300
        }

    def load_data(self):
        """Загрузка данных"""
        self.train_texts = pd.read_csv('../data/X_train.csv')['cleaned_text']
        self.train_labels = pd.read_csv('../data/y_train.csv')['label']
        self.test_texts = pd.read_csv('../data/X_test.csv')['cleaned_text']
        self.test_labels = pd.read_csv('../data/y_test.csv')['label']

    def train_bert(self):
        """Обучение BERT модели"""
        print("Инициализация BERT...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=2
        ).to(self.device)

        # Создание датасетов
        train_dataset = TextDataset(
            self.train_texts, 
            self.train_labels,
            tokenizer,
            self.config['max_length']
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        # Оптимизатор и scheduler
        optimizer = AdamW(model.parameters(), lr=self.config['learning_rate'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * self.config['epochs']
        )

        # Обучение
        print("Начало обучения BERT...")
        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{self.config["epochs"]}, Loss: {avg_loss:.4f}')

        return model, tokenizer

    def evaluate_model(self, model, tokenizer, is_bert=True):
        """Оценка модели"""
        model.eval()
        test_dataset = TextDataset(
            self.test_texts,
            self.test_labels,
            tokenizer if is_bert else None,
            self.config['max_length']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                if is_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels']

                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                else:
                    texts, labels = batch
                    outputs = model(texts.to(self.device))
                    preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Расчет метрик
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        results = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Сохранение результатов
        self.save_results(results, 'BERT' if is_bert else 'LSTM', all_labels, all_preds)
        
        return results

    def save_results(self, results, model_name, y_true, y_pred):
        """Сохранение результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение метрик
        metrics_file = f"{self.results_dir}/{model_name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Сохранение confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f"{self.results_dir}/{model_name}_confusion_matrix_{timestamp}.png")
        plt.close()

    def save_model(self, path):
        """Сохранение модели"""
        torch.save(self.bert_model.state_dict(), path)

    def load_pretrained_model(self, path):
        """Загрузка предобученной модели"""
        # Устанавливаем имя модели BERT, если оно не установлено
        if not hasattr(self, 'bert_model_name'):
            self.bert_model_name = 'bert-base-uncased'
            
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализируем токенизатор
        if not hasattr(self, 'tokenizer'):
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            
        model = BertForSequenceClassification.from_pretrained(
            self.bert_model_name,
            num_labels=2
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model

def main():
    advanced = AdvancedModels()
    advanced.load_data()
    
    # Обучение и оценка BERT
    print("Обучение BERT модели...")
    bert_model, tokenizer = advanced.train_bert()
    bert_results = advanced.evaluate_model(bert_model, tokenizer)
    
    print("\nРезультаты BERT:")
    print(f"Accuracy: {bert_results['accuracy']:.4f}")
    print(f"F1-score: {bert_results['f1_score']:.4f}")

if __name__ == "__main__":
    main()