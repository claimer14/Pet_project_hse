import re
import emoji
import string

def preprocess_text(text):
    """Очистка и предобработка текста"""
    # Приведение к нижнему регистру
    text = str(text).lower()
    
    # Удаление URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Удаление тегов пользователей
    text = re.sub(r'@\w+', '', text)
    
    # Удаление хэштегов
    text = re.sub(r'#\w+', '', text)
    
    # Удаление эмодзи
    text = emoji.replace_emoji(text, '')
    
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Удаление лишних пробелов
    text = ' '.join(text.split())
    
    return text