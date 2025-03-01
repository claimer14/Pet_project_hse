import pandas as pd
import torch
from torch.utils.data import DataLoader
from model_baseline import BaselineModels
from model_advanced import AdvancedModels, TextDataset
from textblob import TextBlob
import os
import numpy as np
from langdetect import detect
import re

class StressPredictor:
    def __init__(self):
        """Инициализация предсказателя стресса"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загружаем модели
        self.load_models()
        
        # Инициализация ключевых слов
        self.init_keywords()
        
        # Настройка пороговых значений
        self.THRESHOLD = 0.5
        self.SUPPORTED_LANGUAGE = 'en'
        
    def init_keywords(self):
        """Инициализация списков ключевых слов"""
        self.positive_words = {
            'happy', 'great', 'good', 'amazing', 'wonderful', 
            'joy', 'excited', 'relaxed', 'peaceful', 'love',
            'blessed', 'fantastic', 'excellent', 'perfect', 'calm',
            'satisfied', 'content', 'delighted', 'pleased', 'grateful',
            'success', 'successful', 'accomplished', 'achievement', 'proud',
            'fun', 'enjoy', 'enjoyed', 'awesome', 'beautiful', 'best',
            'comfortable', 'confidence', 'energetic', 'enthusiastic',
            'glad', 'happiness', 'harmony', 'inspired', 'motivated',
            'optimistic', 'positive', 'refreshed', 'satisfied', 'thank'
        }
        
        self.negative_words = {
            'stress', 'anxiety', 'worried', 'fear', 'panic',
            'depression', 'overwhelmed', 'tired', 'exhausted', 'nervous',
            'anxious', 'scared', 'terrified', 'upset', 'sad',
            'frustrated', 'angry', 'annoyed', 'concerned', 'troubled',
            'fail', 'failure', 'difficult', 'hard', 'impossible',
            'pressure', 'deadline', 'burden', 'struggle', 'worried',
            'afraid', 'agitated', 'desperate', 'distressed', 'dread',
            'frightened', 'helpless', 'hopeless', 'miserable', 'nightmare',
            'pain', 'panic', 'suffering', 'tense', 'terrible', 'worse',
            'bad', 'die', 'dead', 'kill myself', 'lonely'
        }
    
    def is_english(self, text):
        """Улучшенная проверка английского языка"""
        try:
            # Очищаем текст от специальных символов перед проверкой
            cleaned_text = self.clean_text(text)
            
            # Если текст слишком короткий, считаем его английским
            if len(cleaned_text.split()) <= 2:
                return True
                
            # Проверяем наличие английских слов
            english_words = set(cleaned_text.lower().split()) & (self.positive_words | self.negative_words)
            if english_words:
                return True
                
            # Используем langdetect как последний вариант
            return detect(cleaned_text) == self.SUPPORTED_LANGUAGE
        except:
            # В случае ошибки, проверяем наличие базовых английских символов
            return bool(re.search(r'[a-zA-Z]', text))
    
    def clean_text(self, text):
        """Очистка текста от специальных символов"""
        # Удаляем специальные символы, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        return text
    
    def load_models(self):
        """Загрузка сохраненных моделей"""
        print("Loading models...")
        
        # Загружаем базовую модель
        print("Loading baseline model...")
        self.baseline = BaselineModels()
        self.baseline.load_model('../results/baseline/model.pkl')
        
        # Загружаем BERT
        print("Loading BERT model...")
        self.advanced = AdvancedModels()
        self.advanced.load_data()
        self.bert_model = self.advanced.load_pretrained_model('../results/advanced/bert_model_best.pt')
        self.tokenizer = self.advanced.tokenizer
        
        print("Models loaded successfully")
    
    def has_positive_keywords(self, text):
        """Проверка на позитивные ключевые слова"""
        words = set(text.lower().split())
        matches = words & self.positive_words
        return bool(matches), len(matches)
    
    def has_negative_keywords(self, text):
        """Проверка на негативные ключевые слова"""
        words = set(text.lower().split())
        matches = words & self.negative_words
        return bool(matches), len(matches)
    
    def analyze_sentiment(self, text):
        """Анализ эмоциональной окраски текста"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def get_model_predictions(self, text):
        """Получение предсказаний от моделей"""
        # Предсказание базовой модели
        vec_text = self.baseline.vectorizer.transform([text])
        baseline_prob = self.baseline.baseline_model.predict_proba(vec_text)[0, 1]
        
        # Предсказание BERT
        self.bert_model.eval()
        test_dataset = TextDataset(
            pd.Series([text]),
            pd.Series([0]),
            self.tokenizer,
            self.advanced.config['max_length']
        )
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        with torch.no_grad():
            batch = next(iter(test_loader))
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.bert_model(**inputs)
            bert_prob = torch.softmax(outputs.logits, dim=1)[0, 1].cpu().numpy()
        
        return baseline_prob, bert_prob
    
    def predict(self, text):
        """Комплексное предсказание стресса"""
        # Проверяем язык текста
        if not self.is_english(text):
            print("\nОшибка: Поддерживается только английский язык.")
            print("Error: Only English language is supported.")
            return {
                'error': 'Language not supported',
                'message': 'Only English language is supported'
            }
        
        # Очищаем текст
        text = self.clean_text(text)
        
        # Проверяем, что текст не пустой после очистки
        if not text.strip():
            print("\nОшибка: Текст пуст или содержит только специальные символы.")
            print("Error: Text is empty or contains only special characters.")
            return {
                'error': 'Empty text',
                'message': 'Text is empty after cleaning'
            }
        
        print("\nАнализ текста:", text)
        
        # 1. Получаем предсказания моделей
        baseline_prob, bert_prob = self.get_model_predictions(text)
        
        # 2. Анализируем ключевые слова
        has_positive, pos_count = self.has_positive_keywords(text)
        has_negative, neg_count = self.has_negative_keywords(text)
        
        # 3. Получаем эмоциональную оценку
        sentiment = self.analyze_sentiment(text)
        
        # 4. Корректируем вероятности
        # 4.1 Корректировка на основе ключевых слов
        keyword_factor = 0.5  # Начинаем с нейтральной позиции
        if has_positive:
            keyword_factor -= 0.3 * pos_count  # Сильнее уменьшаем при позитивных словах
        if has_negative:
            keyword_factor += 0.3 * neg_count
        
        # 4.2 Корректировка на основе эмоциональной оценки
        sentiment_factor = (1 - sentiment) / 2
        
        # 5. Вычисляем финальную оценку с обновленными весами
        model_score = (baseline_prob * 0.5 + bert_prob * 0.5)  # Равные веса для моделей
        keyword_score = min(max(keyword_factor, 0), 1)
        
        # Базовая оценка
        final_score = (
            0.3 * model_score +
            0.4 * keyword_score +
            0.3 * sentiment_factor
        )
        
        # Применяем корректировки
        if pos_count > 0 and neg_count == 0:
            # Если есть только позитивные слова
            final_score *= (1 - (0.3 * pos_count))  # Уменьшаем оценку
            
        if sentiment > 0.5:  # Если текст очень позитивный
            final_score *= 0.5  # Значительно уменьшаем оценку
            
        if neg_count > 0:
            # Если есть негативные слова
            final_score = max(final_score, 0.6)  # Минимальный порог при негативе
        
        # Финальные корректировки
        if pos_count > 0 and neg_count == 0 and sentiment > 0:
            # Если текст полностью позитивный
            final_score = min(final_score, 0.4)  # Максимальный порог для позитива
        
        # 6. Определяем наличие стресса и его уровень
        is_stress = final_score > self.THRESHOLD
        
        # Корректируем определение уровня стресса
        if pos_count > 0 and neg_count == 0:
            stress_level = "Низкий"
            is_stress = False
            final_score = min(final_score, 0.4)  # Дополнительное ограничение для позитивных текстов
        else:
            stress_level = "Высокий" if final_score > 0.7 else "Средний" if final_score > 0.5 else "Низкий"
        
        # 7. Выводим подробные результаты
        print("\nПодробные результаты:")
        print(f"1. Предсказания моделей:")
        print(f"   - Базовая модель: {baseline_prob:.2%}")
        print(f"   - BERT модель: {bert_prob:.2%}")
        print(f"2. Анализ ключевых слов:")
        print(f"   - Позитивные слова: {pos_count}")
        print(f"   - Негативные слова: {neg_count}")
        print(f"3. Эмоциональная оценка: {sentiment:.2f} ({sentiment_factor:.2%})")
        print(f"\nИтоговая оценка: {final_score:.2%}")
        print(f"Уровень стресса: {stress_level}")
        print(f"Наличие стресса: {'Да' if is_stress else 'Нет'}")
        
        return {
            'baseline_probability': baseline_prob,
            'bert_probability': bert_prob,
            'keyword_score': keyword_score,
            'sentiment_score': sentiment_factor,
            'final_score': final_score,
            'is_stress': is_stress,
            'stress_level': stress_level
        }

def main():
    predictor = StressPredictor()
    
    print("\nСистема анализа стресса (только английский язык)")
    print("Stress Analysis System (English only)")
    
    # Интерактивный режим
    while True:
        text = input("\nВведите текст для анализа (или 'q' для выхода): ")
        if text.lower() == 'q':
            break
        
        result = predictor.predict(text)
        if result and 'error' in result:
            continue

if __name__ == "__main__":
    main()