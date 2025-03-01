import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text

class DataPreprocessor:
    def __init__(self, file_path='../data/Mental-Health-Twitter.csv'):
        """Инициализация препроцессора"""
        self.df = pd.read_csv(file_path)
        
    def preprocess_data(self):
        """Очистка и предобработка данных"""
        # Очистка текстов (заменяем 'text' на 'post_text')
        self.df['cleaned_text'] = self.df['post_text'].apply(preprocess_text)
        
        # Подготовка меток
        self.X = self.df['cleaned_text']
        self.y = self.df['label']
        
    def split_data(self, test_size=0.15, val_size=0.15):
        """Разделение на train/val/test"""
        # Сначала отделяем тестовую выборку
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Затем отделяем валидационную выборку
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42, stratify=y_temp
        )
        
        # Сохраняем разделенные данные
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        # Сохраняем разделенные данные в файлы
        for name, data in splits.items():
            data.to_csv(f'../data/{name}.csv', index=False)
            
        return splits

def main():
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()
    splits = preprocessor.split_data()
    
    # Вывод информации о размерах выборок
    print("\nРазмеры выборок:")
    print(f"Тренировочная: {len(splits['X_train'])} примеров")
    print(f"Валидационная: {len(splits['X_val'])} примеров")
    print(f"Тестовая: {len(splits['X_test'])} примеров")

if __name__ == "__main__":
    main()