import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np # For NaN handling and dummy data generation

# --- Project Module Imports (Relative) ---
from ..ml_data_preparation import data_preparer as dp_module
from ..feature_engineering import technical_indicators as ti_module
from ..feature_engineering import sentiment_analysis as sa_module
# For dummy data creation, if needed directly (though data_preparer should handle it)
# from ..ml_data_preparation.data_preparer import create_dummy_market_data, create_dummy_news_data


# --- Base directory definitions (relative to this file) ---
# model_trainer.py is in AdvancedCryptoBot/src/ml_model/
# Models are saved in AdvancedCryptoBot/models/
DEFAULT_MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
# Data for __main__ is in AdvancedCryptoBot/data/
DEFAULT_MAIN_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
DEFAULT_MAIN_HIST_DIR = os.path.join(DEFAULT_MAIN_DATA_DIR, 'historical')
DEFAULT_MAIN_NEWS_DIR = os.path.join(DEFAULT_MAIN_DATA_DIR, 'news')


def train_model(X, y, model_type='random_forest_classifier', model_params=None, test_size=0.2, random_state=42):
    """
    Обучает модель машинного обучения на предоставленных данных.

    :param X: pandas DataFrame с признаками.
    :type X: pd.DataFrame
    :param y: pandas Series с целевой переменной.
    :type y: pd.Series
    :param model_type: Тип модели для обучения (пока только 'random_forest_classifier').
    :type model_type: str
    :param model_params: Словарь параметров для модели. Если None, используются стандартные.
    :type model_params: dict or None
    :param test_size: Доля данных для тестовой выборки.
    :type test_size: float
    :param random_state: Random state для воспроизводимости.
    :type random_state: int
    :return: Обученная модель.
    :rtype: BaseEstimator (scikit-learn model)
    """
    print(f"Разделение данных на обучающую и тестовую выборки (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False) # shuffle=False для временных рядов

    print(f"Инициализация модели: {model_type}...")
    if model_type == 'random_forest_classifier':
        if model_params is None:
            model_params = {'n_estimators': 100, 'random_state': random_state, 'class_weight': 'balanced'}
        model = RandomForestClassifier(**model_params)
    # TODO: Добавить другие типы моделей по мере необходимости (например, RandomForestRegressor, GradientBoostingClassifier)
    # elif model_type == 'random_forest_regressor':
    #     if model_params is None:
    #         model_params = {'n_estimators': 100, 'random_state': random_state}
    #     model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    print("Обучение модели...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Ошибка во время обучения модели: {e}")
        print("Проверьте данные X_train и y_train на наличие NaN или нечисловых значений.")
        return None

    print("Создание прогнозов на тестовой выборке...")
    y_pred = model.predict(X_test)

    print("\n--- Оценка Модели ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность (Accuracy): {accuracy:.4f}")
    
    print("\nОтчет по классификации (Classification Report):")
    try:
        print(classification_report(y_test, y_pred, zero_division=0))
    except Exception as e:
        print(f"Не удалось сгенерировать отчет по классификации: {e}")

    return model

def save_model(model, base_dir=DEFAULT_MODEL_SAVE_DIR, filename='trained_model.joblib'):
    """
    Сохраняет обученную модель в файл с использованием joblib.

    :param model: Обученная модель.
    :type model: object
    :param base_dir: Директория для сохранения модели.
    :type base_dir: str
    :param filename: Имя файла для сохранения модели.
    :type filename: str
    """
    if model is None:
        print("Модель не предоставлена для сохранения.")
        return

    try:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, filename)
        joblib.dump(model, filepath)
        print(f"Модель успешно сохранена в: {filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении модели в {filepath}: {e}")

def load_model(filepath):
    """
    Загружает модель из файла с использованием joblib.

    :param filepath: Путь к файлу с сохраненной моделью.
    :type filepath: str
    :return: Загруженная модель или None в случае ошибки.
    :rtype: object or None
    """
    try:
        if not os.path.exists(filepath):
            print(f"Файл модели не найден по пути: {filepath}")
            return None
        model = joblib.load(filepath)
        print(f"Модель успешно загружена из: {filepath}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели из {filepath}: {e}")
        return None

def make_prediction(model, current_features_df):
    """
    Делает предсказания с использованием обученной модели.

    :param model: Обученная модель scikit-learn.
    :type model: object
    :param current_features_df: pandas DataFrame с признаками для предсказания.
                                 Колонки должны совпадать с теми, на которых обучалась модель.
    :type current_features_df: pd.DataFrame
    :return: Кортеж (predictions, probabilities).
             predictions: массив предсказанных классов/значений.
             probabilities: массив вероятностей для каждого класса (для классификаторов).
                            None для регрессоров или если метод predict_proba отсутствует.
    :rtype: tuple (np.ndarray, np.ndarray or None)
    """
    if model is None:
        print("Модель не загружена, предсказание невозможно.")
        return None, None
    if current_features_df is None or current_features_df.empty:
        print("Данные для предсказания отсутствуют.")
        return None, None

    print("Создание прогнозов для текущих данных...")
    try:
        predictions = model.predict(current_features_df)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(current_features_df)
            
        return predictions, probabilities
    except Exception as e:
        print(f"Ошибка при создании прогноза: {e}")
        print("Убедитесь, что current_features_df содержит те же колонки (и в том же порядке), что и обучающие данные, и не содержит NaN.")
        return None, None

if __name__ == '__main__':
    print("--- Демонстрация модуля model_trainer ---")

    # --- Helper functions to create dummy data if missing ---
    def create_dummy_market_data_for_trainer(filepath, symbol, timeframe, num_rows=100):
        print(f"Создание dummy рыночных данных для model_trainer: {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=num_rows, freq='H')
        data = {
            'timestamp': timestamps,
            'open': np.random.uniform(100, 200, size=num_rows),
            'high': np.random.uniform(200, 300, size=num_rows), # Will be adjusted
            'low': np.random.uniform(50, 100, size=num_rows),   # Will be adjusted
            'close': np.random.uniform(100, 200, size=num_rows),
            'volume': np.random.uniform(1000, 5000, size=num_rows)
        }
        df = pd.DataFrame(data)
        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = df[['high', 'open', 'close']].max(axis=1) + np.random.uniform(0, 10, size=num_rows)
        df['low'] = df[['low', 'open', 'close']].min(axis=1) - np.random.uniform(0, 10, size=num_rows)
        df['low'] = np.maximum(0, df['low']) # Ensure low is not negative
        
        df.to_csv(filepath, index=False)
        print(f"Dummy рыночные данные сохранены: {filepath}")

    def create_dummy_news_data_for_trainer(filepath, num_rows=30, market_start_time_str='2023-01-01 00:00:00', market_periods=100, market_freq='H'):
        print(f"Создание dummy новостных данных для model_trainer: {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Generate news timestamps that overlap with market data
        market_start_time = pd.to_datetime(market_start_time_str)
        market_end_time = market_start_time + pd.Timedelta(hours=(market_periods -1)) # Approx end time of market data
        
        # Generate news timestamps within this range
        news_timestamps = []
        for _ in range(num_rows):
            random_offset_hours = np.random.randint(0, market_periods -1)
            random_offset_minutes = np.random.randint(0, 59)
            news_time = market_start_time + pd.Timedelta(hours=random_offset_hours, minutes=random_offset_minutes)
            news_timestamps.append(news_time)
        news_timestamps.sort()

        data = {
            'publishedAt': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in news_timestamps], # Store as string, like real CSV
            'title': [f"Dummy News Title {i+1}" for i in range(num_rows)],
            'description': [f"Dummy news description for article {i+1}. Market is very interesting today." for i in range(num_rows)],
            'url': [f'http://example.com/news_trainer_{i+1}' for i in range(num_rows)],
            'source_name': [f'TestSource{chr(65+i%3)}' for i in range(num_rows)]
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Dummy новостные данные сохранены: {filepath}")

    # --- 0. Параметры и подготовка путей ---
    dummy_symbol = 'DUMMY-COIN-USDT'
    dummy_timeframe = '1H'
    num_market_data_rows = 100 # Достаточно для индикаторов с периодом 50

    model_filename = f"{dummy_symbol.lower().replace('-', '_')}_{dummy_timeframe.lower()}_rf_classifier.joblib"
    full_model_path = os.path.join(DEFAULT_MODEL_SAVE_DIR, model_filename)

    # Пути к dummy данным
    dummy_market_data_path = os.path.join(DEFAULT_MAIN_HIST_DIR, f"{dummy_symbol.upper()}_{dummy_timeframe.upper()}.csv")
    dummy_news_data_path = os.path.join(DEFAULT_MAIN_NEWS_DIR, "dummy_news_aggregated_for_trainer.csv") # Unique name

    # --- 1. Проверка и создание Dummy Данных при необходимости ---
    if not os.path.exists(dummy_market_data_path):
        print(f"Файл dummy рыночных данных не найден. Создание: {dummy_market_data_path}")
        create_dummy_market_data_for_trainer(dummy_market_data_path, dummy_symbol.upper(), dummy_timeframe.upper(), num_rows=num_market_data_rows)
    else:
        print(f"Найден существующий файл dummy рыночных данных: {dummy_market_data_path}")
        
    if not os.path.exists(dummy_news_data_path):
        print(f"Файл dummy новостных данных не найден. Создание: {dummy_news_data_path}")
        create_dummy_news_data_for_trainer(dummy_news_data_path, num_rows=30, market_periods=num_market_data_rows)
    else:
        print(f"Найден существующий файл dummy новостных данных: {dummy_news_data_path}")
        
    # --- 2. Загрузка и подготовка данных с использованием data_preparer ---
    print("\n[Main] Шаг 1 (model_trainer): Загрузка и обработка рыночных данных...")
    market_df = dp_module.load_and_prepare_market_data(
        symbol=dummy_symbol.upper(), # Ensure uppercase for consistency
        timeframe=dummy_timeframe.upper(),
        hist_data_dir=DEFAULT_MAIN_HIST_DIR,
        indicators_module=ti_module
    )

    print("\n[Main] Шаг 2 (model_trainer): Загрузка и обработка новостных данных...")
    news_df = dp_module.load_and_prepare_news_data(
        news_csv_filename=os.path.basename(dummy_news_data_path), # Используем имя файла из определенного выше пути
        news_data_dir=DEFAULT_MAIN_NEWS_DIR, # Каталог тот же
        sentiment_module=sa_module
    )
    
    if market_df is None or market_df.empty:
        print("[Main] Ошибка: Не удалось загрузить или подготовить рыночные данные. Демонстрация прервана.")
        # Попытка удалить потенциально "плохие" dummy файлы, чтобы они пересоздались при следующем запуске
        if os.path.exists(dummy_market_data_path): os.remove(dummy_market_data_path)
        exit()
    
    # Если новости не загрузились, создаем пустой DataFrame, чтобы объединение не упало
    if news_df is None:
        news_df = pd.DataFrame(columns=['publishedAt']) 


    print("\n[Main] Шаг 3: Объединение рыночных и новостных данных...")
    merged_df = dp_module.align_and_merge_data(
        market_df=market_df, 
        news_df=news_df,
        news_time_window_td=pd.Timedelta(hours=24)
    )

    print("\n[Main] Шаг 4: Определение целевой переменной...")
    final_df = dp_module.define_target_variable(
        merged_df=merged_df, 
        price_column='close', 
        forecast_horizon=1, # Прогноз на 1 шаг вперед
        type='classification'
    )

    if final_df is None or final_df.empty or len(final_df) < 50:
        print("[Main] Ошибка: Данные для обучения отсутствуют, пусты или содержат слишком мало строк (<50) после обработки.")
        print(f"Размер final_df: {len(final_df) if final_df is not None else 'None'}")
        exit()

    print(f"\n[Main] Размер финального DataFrame для ML: {final_df.shape}")
    print("[Main] Колонки в final_df:", final_df.columns.tolist())
    
    # --- 2. Подготовка X и y для модели ---
    print("\n[Main] Шаг 5: Подготовка X (признаки) и y (цель)...")
    
    # Удаляем строки с NaN, которые могли остаться от индикаторов или объединения
    final_df_cleaned = final_df.copy()
    # Колонки, которые точно не являются признаками
    cols_to_drop_for_X = ['target'] 
    # Если 'timestamp' является индексом, его не нужно удалять из колонок
    # Если 'timestamp' обычная колонка, ее нужно удалить:
    # if 'timestamp' in final_df_cleaned.columns:
    #    cols_to_drop_for_X.append('timestamp')
    
    # Исключаем нечисловые колонки, которые могли случайно попасть (кроме целевой)
    potential_feature_cols = [col for col in final_df_cleaned.columns if col not in cols_to_drop_for_X]
    X = final_df_cleaned[potential_feature_cols].select_dtypes(include=np.number)
    y = final_df_cleaned['target']

    # Обработка NaN в X: сначала ffill, потом bfill, потом dropna для оставшихся (если есть)
    # Это важно, т.к. многие модели не могут работать с NaN
    print(f"[Main] Размер X до обработки NaN: {X.shape}")
    X_original_index = X.index # Сохраняем индекс
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    # Проверяем, есть ли еще NaN после ffill/bfill (например, если весь столбец NaN)
    if X.isnull().any().any():
        print("[Main] Обнаружены NaN в X даже после ffill/bfill. Удаление строк с NaN...")
        # Удаляем строки, где X все еще содержит NaN
        rows_before_dropna = len(X)
        X = X.dropna()
        print(f"[Main] Удалено {rows_before_dropna - len(X)} строк из X из-за NaN.")
    
    # Выравниваем y с X (если из X удалялись строки)
    y = y[X.index] 
    print(f"[Main] Размер X после обработки NaN: {X.shape}")
    print(f"[Main] Размер y после выравнивания с X: {y.shape}")

    if X.empty or y.empty or len(X) < 20: # Убедимся, что осталось достаточно данных
        print("[Main] Ошибка: Недостаточно данных после очистки NaN для обучения модели.")
        exit()

    # --- 3. Обучение модели ---
    print("\n[Main] Шаг 6: Обучение модели...")
    trained_classifier = train_model(X, y, model_type='random_forest_classifier')

    if trained_classifier:
        print("\n[Main] Модель успешно обучена.")
        # --- 4. Сохранение модели ---
        print("\n[Main] Шаг 7: Сохранение модели...")
        save_model(trained_classifier, filename=model_filename)

        # --- 5. Загрузка модели ---
        print("\n[Main] Шаг 8: Загрузка модели...")
        loaded_classifier = load_model(full_model_path)

        if loaded_classifier:
            print("\n[Main] Модель успешно загружена.")
            # --- 6. Подготовка данных для предсказания (например, последняя доступная строка) ---
            print("\n[Main] Шаг 9: Подготовка данных для предсказания...")
            # current_X_sample = X.tail(1).copy() # Берем последнюю строку из X (уже очищенную)
            
            # Более реалистичный сценарий: взять последнюю строку из *исходного* final_df,
            # обработать ее так же, как X, и предсказать.
            # Но для простоты демонстрации возьмем из X.
            
            if not X.empty:
                current_X_sample = X.iloc[[-1]] # Берем последнюю строку, сохраняя DataFrame формат
                print(f"Данные для предсказания (1 строка):\n{current_X_sample}")

                # --- 7. Создание предсказания ---
                print("\n[Main] Шаг 10: Создание предсказания...")
                predictions, probabilities = make_prediction(loaded_classifier, current_X_sample)

                if predictions is not None:
                    print(f"\nПредсказание для последней строки данных: {predictions[0]}")
                    if probabilities is not None:
                        print(f"Вероятности классов: {probabilities[0]}")
            else:
                print("[Main] Не удалось подготовить образец для предсказания, так как X пуст.")
        else:
            print("[Main] Ошибка: Не удалось загрузить модель для предсказания.")
    else:
        print("[Main] Ошибка: Модель не была обучена.")

    print("\n--- Демонстрация model_trainer завершена ---")
