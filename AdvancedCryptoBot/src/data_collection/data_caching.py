import json
import os
import hashlib
import pandas as pd
from datetime import datetime, timedelta

# Определяем базовые пути относительно этого файла
# __file__ указывает на data_caching.py
# os.path.dirname(__file__) -> .../AdvancedCryptoBot/src/data_collection
# os.path.join(os.path.dirname(__file__), '..', '..', 'data') -> .../AdvancedCryptoBot/data
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
DEFAULT_CACHE_DIR = os.path.join(BASE_DATA_DIR, 'cache')
DEFAULT_HISTORICAL_DIR = os.path.join(BASE_DATA_DIR, 'historical')

# Убедимся, что директории существуют при загрузке модуля
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
os.makedirs(DEFAULT_HISTORICAL_DIR, exist_ok=True)

def generate_cache_key(prefix, params):
    """
    Генерирует уникальный ключ для кэширования на основе префикса и параметров.

    :param prefix: Строковый префикс (например, 'coingecko_top_n').
    :type prefix: str
    :param params: Словарь параметров API запроса.
    :type params: dict
    :return: Уникальное имя файла для кэша.
    :rtype: str
    """
    # Сортируем параметры для обеспечения консистентности ключа
    sorted_params = sorted(params.items())
    # Преобразуем параметры в строку
    params_str = "_".join([f"{k}_{v}" for k, v in sorted_params])
    
    # Используем хэш для сокращения длины ключа, если параметров много
    if len(params_str) > 100: # Произвольный порог
        params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()
        filename = f"{prefix}_{params_hash}.json"
    else:
        filename = f"{prefix}_{params_str}.json"
    
    # Заменяем недопустимые символы в имени файла, если они есть
    filename = filename.replace('/', '_').replace(':', '_')
    return filename

def save_to_cache(cache_key, data, cache_dir=DEFAULT_CACHE_DIR):
    """
    Сохраняет данные в кэш-файл вместе с временной меткой.

    :param cache_key: Имя файла для кэша (сгенерированное generate_cache_key).
    :type cache_key: str
    :param data: Данные для кэширования (могут быть list, dict, pd.DataFrame).
    :type data: any
    :param cache_dir: Директория для сохранения кэша.
    :type cache_dir: str
    """
    filepath = os.path.join(cache_dir, cache_key)
    
    data_to_store = data
    if isinstance(data, pd.DataFrame):
        # Сериализуем DataFrame в JSON (orient='table' сохраняет схему)
        data_to_store = data.to_json(orient='table', date_format='iso', default_handler=str)
    
    cache_content = {
        'timestamp': datetime.now().isoformat(),
        'data': data_to_store
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_content, f, ensure_ascii=False, indent=4)
        # print(f"Данные сохранены в кэш: {filepath}") # Для отладки
    except IOError as e:
        print(f"Ошибка при сохранении данных в кэш {filepath}: {e}")
    except TypeError as e:
        print(f"Ошибка сериализации при сохранении данных в кэш {filepath}: {e}")

def load_from_cache(cache_key, cache_dir=DEFAULT_CACHE_DIR, is_dataframe=False):
    """
    Загружает данные из кэш-файла.

    :param cache_key: Имя файла кэша.
    :type cache_key: str
    :param cache_dir: Директория кэша.
    :type cache_dir: str
    :param is_dataframe: Указывает, являются ли кэшированные данные DataFrame.
    :type is_dataframe: bool
    :return: Кортеж (cached_data, cache_timestamp_iso) или (None, None).
    :rtype: tuple
    """
    filepath = os.path.join(cache_dir, cache_key)
    if not os.path.exists(filepath):
        return None, None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_content = json.load(f)
        
        cached_data_raw = cache_content.get('data')
        cache_timestamp_iso = cache_content.get('timestamp')

        if cached_data_raw is None or cache_timestamp_iso is None:
            # print(f"Файл кэша {filepath} имеет неверный формат.") # Для отладки
            return None, None

        if is_dataframe:
            # Десериализуем DataFrame из JSON
            # Убедимся, что cached_data_raw это строка JSON, как ее сохраняет to_json
            if isinstance(cached_data_raw, str):
                cached_data = pd.read_json(cached_data_raw, orient='table')
            else: # Если данные хранились не как строка (старый формат или ошибка)
                print(f"Предупреждение: данные DataFrame в кэше {filepath} не являются строкой JSON.")
                # Попытка восстановить, если это был dict из to_dict() - менее предпочтительно
                cached_data = pd.DataFrame(cached_data_raw)

            # Восстановление datetime для колонки timestamp, если она есть и была строкой
            if 'timestamp' in cached_data.columns:
                 try:
                    cached_data['timestamp'] = pd.to_datetime(cached_data['timestamp'])
                 except Exception: # Если 'timestamp' не может быть преобразован, оставляем как есть
                    pass 
            elif cached_data.index.name == 'timestamp' and isinstance(cached_data.index, pd.DatetimeIndex):
                pass # Индекс уже datetime

        else:
            cached_data = cached_data_raw
            
        # print(f"Данные загружены из кэша: {filepath}") # Для отладки
        return cached_data, cache_timestamp_iso
        
    except (IOError, json.JSONDecodeError, ValueError) as e:
        print(f"Ошибка при загрузке или парсинге кэша {filepath}: {e}")
        return None, None

def is_cache_valid(cache_timestamp_iso, cache_duration_minutes):
    """
    Проверяет, действителен ли кэш на основе его временной метки и длительности.

    :param cache_timestamp_iso: Временная метка сохранения кэша в формате ISO.
    :type cache_timestamp_iso: str
    :param cache_duration_minutes: Длительность валидности кэша в минутах.
    :type cache_duration_minutes: int
    :return: True, если кэш действителен, иначе False.
    :rtype: bool
    """
    if not cache_timestamp_iso:
        return False
    
    try:
        cache_time = datetime.fromisoformat(cache_timestamp_iso)
        expiration_time = cache_time + timedelta(minutes=cache_duration_minutes)
        return datetime.now() < expiration_time
    except ValueError:
        print(f"Неверный формат временной метки ISO: {cache_timestamp_iso}")
        return False

def save_historical_data_to_csv(df, okx_symbol, timeframe, base_dir=DEFAULT_HISTORICAL_DIR):
    """
    Сохраняет исторические данные OHLCV в CSV файл, объединяя с существующими данными.

    :param df: pandas DataFrame с данными OHLCV. Колонка 'timestamp' должна быть datetime.
    :type df: pd.DataFrame
    :param okx_symbol: Символ OKX (например, 'BTC-USDT').
    :type okx_symbol: str
    :param timeframe: Таймфрейм (например, '1H').
    :type timeframe: str
    :param base_dir: Базовая директория для сохранения CSV.
    :type base_dir: str
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        # print(f"Нет данных для сохранения для {okx_symbol} ({timeframe}).") # Для отладки
        return

    if 'timestamp' not in df.columns:
        print(f"Ошибка: колонка 'timestamp' отсутствует в DataFrame для {okx_symbol} ({timeframe}).")
        return
    
    # Убедимся, что timestamp это datetime объект
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Ошибка преобразования 'timestamp' в datetime для {okx_symbol} ({timeframe}): {e}")
        return

    filename = f"{okx_symbol.replace('/', '_')}_{timeframe}.csv" # Заменяем / на _ для имен файлов
    filepath = os.path.join(base_dir, filename)
    
    combined_df = df.copy() # Начинаем с новых данных

    if os.path.exists(filepath):
        try:
            existing_df = pd.read_csv(filepath, parse_dates=['timestamp'])
            if not existing_df.empty:
                # Объединяем существующие и новые данные
                combined_df = pd.concat([existing_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"Файл CSV {filepath} пуст. Будет перезаписан.")
        except Exception as e:
            print(f"Ошибка при чтении существующего CSV файла {filepath}: {e}. Файл будет перезаписан.")

    # Удаляем дубликаты по 'timestamp', оставляя последнее вхождение (самые свежие данные)
    combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    # Сортируем данные по времени в возрастающем порядке
    combined_df.sort_values(by='timestamp', ascending=True, inplace=True)
    
    try:
        combined_df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S') # Стандартный формат даты
        # print(f"Исторические данные сохранены/обновлены: {filepath}") # Для отладки
    except IOError as e:
        print(f"Ошибка при сохранении исторических данных в CSV {filepath}: {e}")

def load_historical_data_from_csv(okx_symbol, timeframe, base_dir=DEFAULT_HISTORICAL_DIR):
    """
    Загружает исторические данные OHLCV из CSV файла.

    :param okx_symbol: Символ OKX (например, 'BTC-USDT').
    :type okx_symbol: str
    :param timeframe: Таймфрейм (например, '1H').
    :type timeframe: str
    :param base_dir: Базовая директория, где хранятся CSV файлы.
    :type base_dir: str
    :return: pandas DataFrame или None, если файл не найден или пуст.
    :rtype: pd.DataFrame or None
    """
    filename = f"{okx_symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        # print(f"Файл исторических данных не найден: {filepath}") # Для отладки
        return None
        
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        if df.empty:
            # print(f"Файл исторических данных {filepath} пуст.") # Для отладки
            return None
        # print(f"Исторические данные загружены из: {filepath}") # Для отладки
        return df
    except pd.errors.EmptyDataError:
        # print(f"Файл CSV {filepath} пуст (EmptyDataError).") # Для отладки
        return None
    except Exception as e:
        print(f"Ошибка при загрузке исторических данных из CSV {filepath}: {e}")
        return None

if __name__ == '__main__':
    # --- Тестирование функций кэширования ---
    print("--- Тестирование кэширования ---")
    # 1. generate_cache_key
    params_cg = {'n': 10}
    cg_key = generate_cache_key('coingecko_top_n', params_cg)
    print(f"CoinGecko cache key: {cg_key}")

    params_okx = {'symbol': 'BTC-USDT', 'timeframe': '1H', 'limit': 100}
    okx_key = generate_cache_key('okx_historical', params_okx)
    print(f"OKX cache key: {okx_key}")

    # 2. save_to_cache & load_from_cache (list)
    sample_list_data = ['BTC', 'ETH', 'USDT']
    save_to_cache(cg_key, sample_list_data)
    loaded_list_data, list_ts = load_from_cache(cg_key)
    print(f"Загруженные list данные: {loaded_list_data}, Timestamp: {list_ts}")

    # 3. save_to_cache & load_from_cache (DataFrame)
    sample_df_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 11:00:00']),
        'open': [100, 102], 'high': [105, 106], 'low': [99, 101], 'close': [102, 105], 'volume': [1000, 1200]
    })
    save_to_cache(okx_key, sample_df_data) # is_dataframe=True не нужен при сохранении
    loaded_df_data, df_ts = load_from_cache(okx_key, is_dataframe=True) # нужен при загрузке
    print(f"Загруженные DataFrame данные:\n{loaded_df_data}")
    print(f"DataFrame Timestamp: {df_ts}")
    if loaded_df_data is not None:
        print(f"Типы данных в загруженном DataFrame:\n{loaded_df_data.dtypes}")


    # 4. is_cache_valid
    valid_now = is_cache_valid(df_ts, 60) # Кэш должен быть валиден (60 минут)
    print(f"Кэш валиден (60 мин)? {valid_now}")
    
    past_timestamp = (datetime.now() - timedelta(minutes=120)).isoformat()
    valid_expired = is_cache_valid(past_timestamp, 60) # Кэш должен быть невалиден
    print(f"Кэш валиден (просрочен на 120 мин, годен 60 мин)? {valid_expired}")

    # --- Тестирование сохранения/загрузки исторических данных CSV ---
    print("\n--- Тестирование CSV для исторических данных ---")
    csv_symbol = 'TEST-USDT'
    csv_timeframe = '1min'

    # Удаляем тестовый CSV, если он существует, для чистоты теста
    test_csv_path = os.path.join(DEFAULT_HISTORICAL_DIR, f"{csv_symbol}_{csv_timeframe}.csv")
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    # Первая порция данных
    df1 = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00']),
        'open': [1.0, 1.1], 'high': [1.2, 1.3], 'low': [0.9, 1.0], 'close': [1.1, 1.2], 'volume': [100, 110]
    })
    print(f"\nСохранение df1 для {csv_symbol}_{csv_timeframe}:")
    save_historical_data_to_csv(df1, csv_symbol, csv_timeframe)
    loaded_df1 = load_historical_data_from_csv(csv_symbol, csv_timeframe)
    print("Загружено после сохранения df1:\n", loaded_df1)

    # Вторая порция данных (с перекрытием и новыми данными)
    df2 = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 00:01:00', '2023-01-01 00:02:00']), # 00:01:00 дублируется
        'open': [1.15, 1.25], 'high': [1.35, 1.45], 'low': [1.05, 1.15], 'close': [1.25, 1.35], 'volume': [115, 125]
    })
    print(f"\nСохранение df2 для {csv_symbol}_{csv_timeframe} (с обновлением дубликата и добавлением нового):")
    save_historical_data_to_csv(df2, csv_symbol, csv_timeframe)
    loaded_df2 = load_historical_data_from_csv(csv_symbol, csv_timeframe)
    print("Загружено после сохранения df2 (ожидается 3 строки, 00:01:00 обновлена):\n", loaded_df2)

    # Третья порция данных (более старые данные)
    df3 = pd.DataFrame({
        'timestamp': pd.to_datetime(['2022-12-31 23:59:00']),
        'open': [0.8], 'high': [0.9], 'low': [0.7], 'close': [0.85], 'volume': [90]
    })
    print(f"\nСохранение df3 для {csv_symbol}_{csv_timeframe} (более старые данные):")
    save_historical_data_to_csv(df3, csv_symbol, csv_timeframe)
    loaded_df3 = load_historical_data_from_csv(csv_symbol, csv_timeframe)
    print("Загружено после сохранения df3 (ожидается 4 строки, отсортировано):\n", loaded_df3)
    
    # Проверка, что timestamp действительно datetime
    if loaded_df3 is not None:
        print(f"\nТипы данных в загруженном историческом DataFrame:\n{loaded_df3.dtypes}")

    # Очистка тестового файла
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"\nТестовый файл {test_csv_path} удален.")
    
    print("\nТестирование завершено.")
