import pandas as pd
import numpy as np
import os
from datetime import datetime # For dummy data

# --- Attempt to import project-specific modules ---
# This structure assumes that the script is run from a context where
# AdvancedCryptoBot is in the Python path (e.g., by setting PYTHONPATH)
# or that an IDE manages the project structure.

try:
    from AdvancedCryptoBot.src.data_collection.data_caching import load_historical_data_from_csv
    from AdvancedCryptoBot.src.feature_engineering import technical_indicators as ti_module
    from AdvancedCryptoBot.src.feature_engineering import sentiment_analysis as sa_module
except ImportError:
    # Fallback for direct execution or if PYTHONPATH is not set
    # This requires data_caching.py, technical_indicators.py, sentiment_analysis.py
    # to be discoverable, e.g., in the same directory or via sys.path manipulation.
    print("Attempting fallback imports for data_preparer.py...")
    import sys
    # Assuming the script is in AdvancedCryptoBot/src/ml_data_preparation/
    # We need to go up two levels to AdvancedCryptoBot/
    # then down to src/data_collection and src/feature_engineering
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    src_path = os.path.join(project_root, 'src')

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    # Now try importing with the modified path
    from data_collection.data_caching import load_historical_data_from_csv
    from feature_engineering import technical_indicators as ti_module
    from feature_engineering import sentiment_analysis as sa_module


# --- Base directory definitions (relative to this file) ---
# data_preparer.py is in AdvancedCryptoBot/src/ml_data_preparation/
# Data is in AdvancedCryptoBot/data/
DEFAULT_BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
DEFAULT_HIST_DATA_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'historical')
DEFAULT_NEWS_DATA_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'news')


def load_and_prepare_market_data(symbol, timeframe, hist_data_dir=DEFAULT_HIST_DATA_DIR, indicators_module=None):
    """
    Загружает исторические OHLCV данные из CSV, добавляет технические индикаторы
    и устанавливает 'timestamp' в качестве индекса.

    :param symbol: Символ криптовалюты (например, 'BTC-USDT').
    :type symbol: str
    :param timeframe: Таймфрейм (например, '1H').
    :type timeframe: str
    :param hist_data_dir: Директория, где хранятся CSV файлы с историческими данными.
    :type hist_data_dir: str
    :param indicators_module: Модуль для добавления технических индикаторов 
                              (должен иметь функцию add_all_indicators).
    :type indicators_module: module
    :return: pandas DataFrame с рыночными данными и индикаторами, или None.
    :rtype: pd.DataFrame or None
    """
    print(f"Загрузка рыночных данных для {symbol} ({timeframe}) из {hist_data_dir}...")
    market_df = load_historical_data_from_csv(symbol, timeframe, base_dir=hist_data_dir)

    if market_df is None or market_df.empty:
        print(f"Рыночные данные для {symbol} ({timeframe}) не найдены или пусты.")
        return None

    if 'timestamp' not in market_df.columns:
        print("Ошибка: колонка 'timestamp' отсутствует в рыночных данных.")
        return None
        
    # Преобразование timestamp в datetime, если это еще не сделано в load_historical_data_from_csv
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
    market_df.set_index('timestamp', inplace=True)
    market_df.sort_index(inplace=True) # Убедимся, что индекс отсортирован

    if indicators_module:
        print("Добавление технических индикаторов...")
        try:
            market_df = indicators_module.add_all_indicators(market_df.copy()) # Используем copy для безопасности
        except Exception as e:
            print(f"Ошибка при добавлении технических индикаторов: {e}")
            # Можно вернуть market_df без индикаторов или None, в зависимости от требований
            # return market_df 
    
    print(f"Рыночные данные для {symbol} ({timeframe}) успешно загружены и подготовлены.")
    return market_df

def load_and_prepare_news_data(news_csv_filename='news_articles.csv', news_data_dir=DEFAULT_NEWS_DATA_DIR, sentiment_module=None):
    """
    Загружает новостные данные из CSV, добавляет оценки тональности и обрабатывает дату публикации.

    :param news_csv_filename: Имя CSV файла с новостями.
    :type news_csv_filename: str
    :param news_data_dir: Директория, где хранится CSV файл с новостями.
    :type news_data_dir: str
    :param sentiment_module: Модуль для добавления оценок тональности 
                             (должен иметь функцию add_sentiment_scores_to_news).
    :type sentiment_module: module
    :return: pandas DataFrame с новостными данными и оценками тональности, или None.
    :rtype: pd.DataFrame or None
    """
    news_csv_path = os.path.join(news_data_dir, news_csv_filename)
    print(f"Загрузка новостных данных из {news_csv_path}...")
    
    if not os.path.exists(news_csv_path):
        print(f"Файл новостей {news_csv_path} не найден.")
        return None
        
    try:
        news_df = pd.read_csv(news_csv_path)
    except pd.errors.EmptyDataError:
        print(f"Файл новостей {news_csv_path} пуст.")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла новостей {news_csv_path}: {e}")
        return None

    if news_df.empty:
        print(f"Файл новостей {news_csv_path} не содержит данных.")
        return news_df # Возвращаем пустой DataFrame, чтобы последующие шаги могли его обработать

    if 'publishedAt' not in news_df.columns:
        print("Ошибка: колонка 'publishedAt' отсутствует в новостных данных.")
        return None # Или news_df, если хотим продолжить без дат

    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'], errors='coerce')
    news_df.dropna(subset=['publishedAt'], inplace=True) # Удаляем строки, где дата не может быть распознана
    news_df.sort_values(by='publishedAt', inplace=True) # Сортируем по дате публикации

    if sentiment_module:
        print("Добавление оценок тональности к новостям...")
        try:
            news_df = sentiment_module.add_sentiment_scores_to_news(news_df.copy()) # Используем copy
        except Exception as e:
            print(f"Ошибка при добавлении оценок тональности: {e}")
            # return news_df # Можно вернуть news_df без оценок или None

    print(f"Новостные данные из {news_csv_path} успешно загружены и подготовлены.")
    return news_df

def align_and_merge_data(market_df, news_df, news_time_window_td=pd.Timedelta(hours=4)):
    """
    Объединяет рыночные данные с агрегированными новостными данными на основе временного окна.

    :param market_df: DataFrame с рыночными данными (индекс 'timestamp').
    :type market_df: pd.DataFrame
    :param news_df: DataFrame с новостными данными (колонка 'publishedAt').
    :type news_df: pd.DataFrame
    :param news_time_window_td: Временное окно для агрегации новостей перед каждой рыночной свечой.
    :type news_time_window_td: pd.Timedelta
    :return: DataFrame рыночных данных с добавленными агрегированными новостными признаками.
    :rtype: pd.DataFrame
    """
    if market_df is None or market_df.empty:
        print("Рыночные данные отсутствуют, объединение невозможно.")
        return market_df
    if news_df is None or news_df.empty:
        print("Новостные данные отсутствуют, будут добавлены пустые новостные признаки.")
        # Добавляем пустые колонки, если нет новостей
        market_df['avg_sentiment_compound'] = 0.0 # или np.nan
        market_df['sum_sentiment_pos'] = 0.0
        market_df['sum_sentiment_neg'] = 0.0
        market_df['news_count'] = 0
        return market_df

    if not isinstance(market_df.index, pd.DatetimeIndex):
        raise ValueError("Индекс market_df должен быть pd.DatetimeIndex.")
    if 'publishedAt' not in news_df.columns or not pd.api.types.is_datetime64_any_dtype(news_df['publishedAt']):
        raise ValueError("news_df должен содержать колонку 'publishedAt' типа datetime.")

    # Убедимся, что news_df отсортирован и имеет 'publishedAt' в качестве индекса для merge_asof
    news_df_indexed = news_df.set_index('publishedAt').sort_index()
    
    # Колонки для агрегации
    sentiment_cols = ['sentiment_compound', 'sentiment_pos', 'sentiment_neg']
    # Убедимся, что эти колонки существуют в news_df_indexed
    missing_sentiment_cols = [col for col in sentiment_cols if col not in news_df_indexed.columns]
    if any(missing_sentiment_cols):
        print(f"Предупреждение: Отсутствуют колонки для анализа тональности: {missing_sentiment_cols}. Агрегация новостей будет неполной.")
        # Заполняем отсутствующие колонки нулями, чтобы агрегация не падала
        for col in missing_sentiment_cols:
            news_df_indexed[col] = 0.0


    # Инициализация колонок для агрегированных новостей в market_df
    market_df['avg_sentiment_compound'] = np.nan
    market_df['sum_sentiment_pos'] = np.nan
    market_df['sum_sentiment_neg'] = np.nan
    market_df['news_count'] = 0

    print(f"Выполняется выравнивание и объединение данных (окно новостей: {news_time_window_td})...")
    # Итеративный подход (более явный, может быть медленным на больших данных)
    for ts_market_candle_end in market_df.index:
        ts_window_start = ts_market_candle_end - news_time_window_td
        
        # Фильтруем новости, опубликованные в интервале (ts_window_start, ts_market_candle_end]
        # Новость должна быть опубликована СТРОГО ПОСЛЕ начала окна и НЕ ПОЗЖЕ конца окна (т.е. конца свечи)
        relevant_news = news_df_indexed[(news_df_indexed.index > ts_window_start) & (news_df_indexed.index <= ts_market_candle_end)]
        
        if not relevant_news.empty:
            market_df.loc[ts_market_candle_end, 'avg_sentiment_compound'] = relevant_news['sentiment_compound'].mean() if 'sentiment_compound' in relevant_news else 0.0
            market_df.loc[ts_market_candle_end, 'sum_sentiment_pos'] = relevant_news['sentiment_pos'].sum() if 'sentiment_pos' in relevant_news else 0.0
            market_df.loc[ts_market_candle_end, 'sum_sentiment_neg'] = relevant_news['sentiment_neg'].sum() if 'sentiment_neg' in relevant_news else 0.0
            market_df.loc[ts_market_candle_end, 'news_count'] = len(relevant_news)
        else: # Если нет новостей в окне, оставляем NaN для средних/сумм, 0 для количества
            market_df.loc[ts_market_candle_end, 'avg_sentiment_compound'] = 0.0 # или np.nan
            market_df.loc[ts_market_candle_end, 'sum_sentiment_pos'] = 0.0
            market_df.loc[ts_market_candle_end, 'sum_sentiment_neg'] = 0.0
            market_df.loc[ts_market_candle_end, 'news_count'] = 0
            
    print("Объединение данных завершено.")
    return market_df

def define_target_variable(merged_df, price_column='close', forecast_horizon=1, type='classification'):
    """
    Определяет целевую переменную для задачи машинного обучения.

    :param merged_df: DataFrame с объединенными признаками.
    :type merged_df: pd.DataFrame
    :param price_column: Колонка с ценой для определения целевой переменной (например, 'close').
    :type price_column: str
    :param forecast_horizon: Горизонт прогнозирования (количество периодов вперед).
    :type forecast_horizon: int
    :param type: Тип целевой переменной: 'classification' (рост/падение) или 'regression' (процентное изменение).
    :type type: str
    :return: DataFrame с добавленной колонкой 'target' и удаленными строками с NaN в 'target'.
    :rtype: pd.DataFrame
    """
    if merged_df is None or merged_df.empty:
        print("DataFrame для определения целевой переменной пуст.")
        return merged_df
        
    if price_column not in merged_df.columns:
        raise ValueError(f"Колонка '{price_column}' не найдена в DataFrame.")

    print(f"Определение целевой переменной (тип: {type}, горизонт: {forecast_horizon})...")
    df_with_target = merged_df.copy()

    if type == 'classification':
        # 1 если цена выросла, 0 если упала или осталась той же
        future_price = df_with_target[price_column].shift(-forecast_horizon)
        df_with_target['target'] = (future_price > df_with_target[price_column]).astype(int)
    elif type == 'regression':
        # Процентное изменение цены
        future_price = df_with_target[price_column].shift(-forecast_horizon)
        df_with_target['target'] = (future_price / df_with_target[price_column]) - 1
    else:
        raise ValueError("Параметр 'type' должен быть 'classification' или 'regression'.")

    # Удаление строк, где 'target' равен NaN (это последние 'forecast_horizon' строк)
    df_with_target.dropna(subset=['target'], inplace=True)
    
    print("Целевая переменная определена.")
    return df_with_target

# --- Функции для создания dummy данных (для if __name__ == '__main__') ---
def create_dummy_market_data(filepath, symbol, timeframe):
    print(f"Создание dummy рыночных данных: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=100, freq='H') # 100 часов данных
    data = {
        'timestamp': timestamps,
        'open': np.random.uniform(100, 200, size=100),
        'high': np.random.uniform(200, 300, size=100),
        'low': np.random.uniform(50, 100, size=100),
        'close': np.random.uniform(100, 200, size=100),
        'volume': np.random.uniform(1000, 5000, size=100)
    }
    # Гарантируем, что high >= open/close и low <= open/close
    df = pd.DataFrame(data)
    df['high'] = df[['high', 'open', 'close']].max(axis=1) + np.random.uniform(0,10, size=100) # high > max(o,c)
    df['low'] = df[['low', 'open', 'close']].min(axis=1) - np.random.uniform(0,10, size=100)   # low < min(o,c)
    df.loc[df['low'] < 0, 'low'] = 0 # low не может быть < 0
    
    df.to_csv(filepath, index=False)

def create_dummy_news_data(filepath):
    print(f"Создание dummy новостных данных: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'publishedAt': pd.to_datetime([
            '2023-01-01 00:30:00', '2023-01-01 03:15:00', '2023-01-01 08:00:00', 
            '2023-01-01 12:00:00', '2023-01-01 15:45:00', '2023-01-02 01:00:00',
            '2023-01-02 05:30:00', '2023-01-02 10:00:00', '2023-01-02 18:00:00',
            '2023-01-03 03:00:00', '2023-01-03 11:00:00', '2023-01-03 20:00:00',
            '2023-01-04 04:00:00', '2023-01-04 14:00:00', '2023-01-05 00:00:00' 
        ]),
        'title': [
            "Great Bitcoin News!", "Ethereum Falls Slightly", "Market is Stable",
            "Altcoin Rally Expected", "Bad News for Crypto", "Bitcoin to the moon!",
            "Ethereum uncertainty persists", "Neutral news today", "Positive outlook for blockchain",
            "Crypto regulations tighten", "New DeFi project launched", "Experts are optimistic",
            "Bearish sentiment grows", "Tech giant invests in Web3", "Market crash fear"
        ],
        'description': [
            "Very positive developments for BTC.", "ETH price corrected downwards.", "No major changes observed.",
            "Many altcoins are showing bullish signs.", "This is not good for the market.", "BTC price is skyrocketing",
            "ETH is volatile.", "General news, nothing special.", "Blockchain tech is evolving.",
            "New regulations impact exchanges.", "Innovative DeFi platform goes live.", "Analysts predict further growth.",
            "Market sentiment turns negative.", "Big company enters the Web3 space.", "Fear of a major market downturn."
        ],
        'url': [f'http://example.com/news{i}' for i in range(15)],
        'source_name': [f'Source{chr(65+i%3)}' for i in range(15)] # SourceA, SourceB, SourceC
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    print("--- Демонстрация модуля data_preparer ---")

    # --- 1. Подготовка Dummy Данных ---
    dummy_symbol = 'DUMMY-COIN-USDT'
    dummy_timeframe = '1H'
    
    dummy_market_csv_path = os.path.join(DEFAULT_HIST_DATA_DIR, f"{dummy_symbol}_{dummy_timeframe}.csv")
    dummy_news_csv_path = os.path.join(DEFAULT_NEWS_DATA_DIR, 'dummy_news_aggregated.csv')

    if not os.path.exists(dummy_market_csv_path):
        create_dummy_market_data(dummy_market_csv_path, dummy_symbol, dummy_timeframe)
    
    if not os.path.exists(dummy_news_csv_path):
        create_dummy_news_data(dummy_news_csv_path)

    # --- 2. Загрузка и подготовка рыночных данных ---
    print("\n--- Шаг 1: Загрузка и обработка рыночных данных ---")
    market_data_df = load_and_prepare_market_data(
        symbol=dummy_symbol, 
        timeframe=dummy_timeframe,
        hist_data_dir=DEFAULT_HIST_DATA_DIR, # Убедимся, что путь корректный
        indicators_module=ti_module # Передаем импортированный модуль
    )
    if market_data_df is not None:
        print("\nРыночные данные с индикаторами (первые 5 строк):")
        print(market_data_df.head())
        print("\nРыночные данные с индикаторами (последние 5 строк):")
        print(market_data_df.tail())
        print(f"Количество строк в рыночных данных: {len(market_data_df)}")
    else:
        print("Не удалось загрузить рыночные данные.")
        # Выход, если основные данные не загружены
        exit() 

    # --- 3. Загрузка и подготовка новостных данных ---
    print("\n--- Шаг 2: Загрузка и обработка новостных данных ---")
    news_data_df = load_and_prepare_news_data(
        news_csv_filename='dummy_news_aggregated.csv', # Имя файла, созданного create_dummy_news_data
        news_data_dir=DEFAULT_NEWS_DATA_DIR,
        sentiment_module=sa_module # Передаем импортированный модуль
    )
    if news_data_df is not None:
        print("\nНовостные данные с оценками тональности (первые 5 строк):")
        print(news_data_df.head())
        print(f"Количество строк в новостных данных: {len(news_data_df)}")
    else:
        print("Не удалось загрузить новостные данные. Продолжаем без них.")
        # Создаем пустой DataFrame с нужными колонками, если новости не загрузились,
        # чтобы align_and_merge_data не упал
        news_data_df = pd.DataFrame(columns=['publishedAt', 'sentiment_compound', 'sentiment_pos', 'sentiment_neg'])


    # --- 4. Объединение данных ---
    print("\n--- Шаг 3: Объединение рыночных и новостных данных ---")
    # Убедимся, что market_data_df не None перед объединением
    if market_data_df is not None:
        merged_features_df = align_and_merge_data(
            market_df=market_data_df, 
            news_df=news_data_df, # news_data_df может быть пустым DataFrame, если новости не загрузились
            news_time_window_td=pd.Timedelta(hours=24) # Окно для новостей - 24 часа
        )
        print("\nОбъединенные данные (первые 5 строк):")
        print(merged_features_df.head())
        print("\nОбъединенные данные (последние 5 строк):")
        print(merged_features_df.tail())
        print(f"Количество строк в объединенных данных: {len(merged_features_df)}")
        print("Колонки в объединенных данных:", merged_features_df.columns.tolist())

    else:
        print("Рыночные данные отсутствуют, невозможно выполнить объединение и определение цели.")
        exit()

    # --- 5. Определение целевой переменной ---
    print("\n--- Шаг 4: Определение целевой переменной ---")
    final_ml_df_classification = define_target_variable(
        merged_df=merged_features_df.copy(), # Используем copy
        price_column='close', 
        forecast_horizon=5, # Прогнозируем на 5 часов вперед
        type='classification'
    )
    print("\nФинальный DataFrame для классификации (первые 5 строк):")
    print(final_ml_df_classification.head())
    print("\nФинальный DataFrame для классификации (последние 5 строк):")
    # Убедимся, что 'target' есть в колонках перед выводом
    if 'target' in final_ml_df_classification.columns:
        print(final_ml_df_classification[['close', 'avg_sentiment_compound', 'news_count', 'target']].tail())
    else:
        print(final_ml_df_classification.tail()) # Если 'target' не добавилась по какой-то причине
    print(f"Количество строк в данных для классификации: {len(final_ml_df_classification)}")


    final_ml_df_regression = define_target_variable(
        merged_df=merged_features_df.copy(), # Используем copy
        price_column='close', 
        forecast_horizon=5, 
        type='regression'
    )
    print("\nФинальный DataFrame для регрессии (последние 5 строк):")
    if 'target' in final_ml_df_regression.columns:
        print(final_ml_df_regression[['close', 'avg_sentiment_compound', 'news_count', 'target']].tail())
    else:
        print(final_ml_df_regression.tail())
    print(f"Количество строк в данных для регрессии: {len(final_ml_df_regression)}")
    
    print("\n--- Демонстрация data_preparer завершена ---")
