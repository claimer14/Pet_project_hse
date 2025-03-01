import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from utils import preprocess_text

class DataAnalyzer:
    def __init__(self, file_path='../data/Mental-Health-Twitter.csv'):
        """Инициализация анализатора данных"""
        try:
            self.df = pd.read_csv(file_path)
            # Выводим информацию о колонках
            print("Доступные колонки в датасете:")
            print(self.df.columns.tolist())
            print("\nПример данных:")
            print(self.df.head())
            
            # Используем правильное название колонки 'post_text'
            text_column = 'post_text'
            if text_column not in self.df.columns:
                raise KeyError(f"Колонка '{text_column}' не найдена в датасете. "
                             f"Доступные колонки: {self.df.columns.tolist()}")
            
            self.df['cleaned_text'] = self.df[text_column].apply(preprocess_text)
            
        except FileNotFoundError:
            print(f"Ошибка: Файл {file_path} не найден!")
            raise
        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def analyze_class_distribution(self):
        """Анализ сбалансированности классов"""
        plt.figure(figsize=(10, 6))
        class_dist = self.df['label'].value_counts()
        sns.barplot(x=class_dist.index, y=class_dist.values)
        plt.title('Распределение классов')
        plt.xlabel('Класс (0 - норма, 1 - стресс)')
        plt.ylabel('Количество')
        plt.savefig('../results/class_distribution.png')
        plt.close()
        
        print("\nРаспределение классов:")
        print(class_dist)
        
    def analyze_text_length(self):
        """Анализ длины текстов"""
        self.df['text_length'] = self.df['cleaned_text'].str.len()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='label', y='text_length', data=self.df)
        plt.title('Распределение длины текстов по классам')
        plt.savefig('../results/text_length_distribution.png')
        plt.close()
        
        print("\nСтатистика длины текстов:")
        print(self.df.groupby('label')['text_length'].describe())
        
    def show_common_words(self, n=20):
        """Анализ частых слов"""
        def get_common_words(texts):
            words = ' '.join(texts).split()
            return pd.DataFrame(Counter(words).most_common(n), 
                              columns=['Слово', 'Частота'])
        
        print("\nЧастые слова в нормальных текстах:")
        print(get_common_words(self.df[self.df['label'] == 0]['cleaned_text']))
        
        print("\nЧастые слова в стрессовых текстах:")
        print(get_common_words(self.df[self.df['label'] == 1]['cleaned_text']))
        
    def show_examples(self):
        """Показать примеры текстов"""
        print("\nПримеры обычных текстов:")
        print(self.df[self.df['label'] == 0]['post_text'].head())  
        
        print("\nПримеры стрессовых текстов:")
        print(self.df[self.df['label'] == 1]['post_text'].head())  

def main():
    analyzer = DataAnalyzer()
    analyzer.analyze_class_distribution()
    analyzer.analyze_text_length()
    analyzer.show_common_words()
    analyzer.show_examples()

if __name__ == "__main__":
    main()