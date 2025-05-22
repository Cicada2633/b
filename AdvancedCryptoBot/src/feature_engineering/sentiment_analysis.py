import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Убедимся, что необходимые лексиконы VADER загружены при первом импорте.
# SentimentIntensityAnalyzer() сам позаботится об этом при инициализации.

def get_vader_sentiment_scores(text, analyzer):
    """
    Анализирует тональность текста с использованием VADER.

    :param text: Текст для анализа.
    :type text: str
    :param analyzer: Экземпляр SentimentIntensityAnalyzer.
    :type analyzer: vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer
    :return: Словарь с оценками тональности ('neg', 'neu', 'pos', 'compound').
    :rtype: dict
    """
    if not isinstance(text, str):
        # Если текст не строка (например, float/NaN), возвращаем нейтральные значения или None
        # print(f"Предупреждение: Входной текст не является строкой: {text}. Возвращаем нейтральные оценки.")
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    try:
        # Получение словаря оценок
        scores = analyzer.polarity_scores(text)
        return scores
    except Exception as e:
        print(f"Ошибка при анализе тональности текста '{text[:50]}...': {e}")
        # Возвращаем нейтральные значения в случае ошибки
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

def add_sentiment_scores_to_news(news_df):
    """
    Добавляет оценки тональности VADER к DataFrame новостей.

    Анализируется объединенный текст из колонок 'title' и 'description'.

    :param news_df: pandas DataFrame с новостными данными. 
                    Ожидаются колонки 'title' и 'description'.
    :type news_df: pd.DataFrame
    :return: pandas DataFrame с добавленными колонками оценок тональности
             ('sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound').
    :rtype: pd.DataFrame
    """
    if not isinstance(news_df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame.")
    
    if news_df.empty:
        # print("Предупреждение: DataFrame новостей пуст. Оценки тональности не будут добавлены.")
        # Добавляем пустые колонки, если DataFrame пуст, чтобы схема была консистентной
        news_df['sentiment_neg'] = []
        news_df['sentiment_neu'] = []
        news_df['sentiment_pos'] = []
        news_df['sentiment_compound'] = []
        return news_df

    analyzer = SentimentIntensityAnalyzer()
    
    # Создаем копию для безопасного изменения
    df_with_sentiment = news_df.copy()

    # Функция для применения к каждой строке DataFrame
    def calculate_row_sentiment(row):
        text_for_analysis = ""
        title = row.get('title', '')
        description = row.get('description', '')
        
        if pd.notna(title) and isinstance(title, str):
            text_for_analysis += title
        
        if pd.notna(description) and isinstance(description, str):
            if text_for_analysis: # Если есть заголовок, добавляем пробел
                text_for_analysis += " " + description
            else:
                text_for_analysis += description
        
        if not text_for_analysis: # Если оба поля пустые или не строки
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0} # Нейтральные оценки
            
        return get_vader_sentiment_scores(text_for_analysis, analyzer)

    # Применение функции и создание новых колонок
    # apply возвращает Series из словарей, которую нужно будет развернуть в колонки
    sentiment_scores_list = df_with_sentiment.apply(calculate_row_sentiment, axis=1)
    
    # Преобразование списка словарей в DataFrame и объединение
    sentiment_df = pd.DataFrame(sentiment_scores_list.tolist(), index=df_with_sentiment.index)
    
    # Переименование колонок для ясности
    sentiment_df.rename(columns={
        'neg': 'sentiment_neg',
        'neu': 'sentiment_neu',
        'pos': 'sentiment_pos',
        'compound': 'sentiment_compound'
    }, inplace=True)
    
    df_with_sentiment = pd.concat([df_with_sentiment, sentiment_df], axis=1)
    
    return df_with_sentiment

if __name__ == '__main__':
    print("--- Демонстрация модуля sentiment_analysis ---")

    # Пример DataFrame с новостями
    sample_news_data = {
        'publishedAt': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                                       '2023-01-01 12:00:00', '2023-01-01 13:00:00',
                                       '2023-01-01 14:00:00']),
        'title': [
            "Bitcoin Surges to New Highs!", 
            "Ethereum Price Drops Sharply", 
            "Crypto Market Remains Stable and Neutral",
            "Analysts Predict Bullish Future for Altcoins",
            None # Пример с отсутствующим заголовком
        ],
        'description': [
            "Positive market sentiment drives Bitcoin price up significantly.",
            "Regulatory concerns lead to a sudden fall in Ethereum's value. Very bad news.",
            "The overall cryptocurrency market shows little movement today. Observers are neutral.",
            "Several altcoins show promising growth potential according to experts. This is great!",
            "This article has no description, only a missing title." # Описание без заголовка
        ],
        'url': ['url1', 'url2', 'url3', 'url4', 'url5'],
        'source_name': ['SourceA', 'SourceB', 'SourceC', 'SourceD', 'SourceE']
    }
    news_articles_df = pd.DataFrame(sample_news_data)

    print("\nИсходный DataFrame новостей:")
    print(news_articles_df[['title', 'description']])

    # Демонстрация get_vader_sentiment_scores с SentimentIntensityAnalyzer
    analyzer_instance = SentimentIntensityAnalyzer()
    sample_text_positive = "This is a great and wonderful piece of news!"
    sample_text_negative = "This is terrible, awful, and very bad news."
    
    print(f"\nАнализ тональности для позитивного текста: '{sample_text_positive}'")
    print(get_vader_sentiment_scores(sample_text_positive, analyzer_instance))
    
    print(f"\nАнализ тональности для негативного текста: '{sample_text_negative}'")
    print(get_vader_sentiment_scores(sample_text_negative, analyzer_instance))

    # Демонстрация add_sentiment_scores_to_news
    print("\nПрименение add_sentiment_scores_to_news() к DataFrame новостей:")
    df_with_sentiments = add_sentiment_scores_to_news(news_articles_df.copy()) # Используем copy()

    print("\nDataFrame новостей с добавленными оценками тональности:")
    print(df_with_sentiments[['title', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']])

    # Проверка случая с пустым DataFrame
    print("\nТестирование с пустым DataFrame:")
    empty_df = pd.DataFrame(columns=['publishedAt', 'title', 'description', 'url', 'source_name'])
    empty_df_with_sentiments = add_sentiment_scores_to_news(empty_df)
    print("Колонки в DataFrame после обработки пустого DataFrame:")
    print(empty_df_with_sentiments.columns.tolist())
    print(f"Количество строк: {len(empty_df_with_sentiments)}")


    print("\n--- Демонстрация sentiment_analysis завершена ---")
