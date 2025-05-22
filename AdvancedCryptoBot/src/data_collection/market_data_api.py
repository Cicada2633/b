import requests
import pandas as pd
from datetime import datetime
import time # Для демонстрации кэширования

# Импорт функций кэширования из соседнего модуля
from .data_caching import (
    generate_cache_key,
    save_to_cache,
    load_from_cache,
    is_cache_valid,
    save_historical_data_to_csv,
    load_historical_data_from_csv
)

# Попытка импортировать OKX SDK, но не делать его обязательным
try:
    from okx.MarketData import MarketAPI as OkxMarketAPI
    OKX_SDK_AVAILABLE = True
except ImportError:
    OKX_SDK_AVAILABLE = False
    OkxMarketAPI = None # Заглушка, если SDK недоступен

# --- CoinGecko Functions ---
COINGECKO_CACHE_DURATION_MINUTES = 180 # 3 часа

def get_top_n_coins_coingecko(n=10):
    """
    Получает топ N криптовалют по рыночной капитализации с CoinGecko API, используя кэширование.

    :param n: Количество криптовалют для получения (по умолчанию 10).
    :type n: int
    :return: Список символов криптовалют в верхнем регистре (например, ['BTC', 'ETH', 'USDT', ...]).
             Возвращает пустой список в случае ошибки.
    :rtype: list[str]
    """
    # URL API CoinGecko для получения рыночных данных
    url = "https://api.coingecko.com/api/v3/coins/markets"
    
    """
    api_params = { # Параметры, которые влияют на результат API
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': n,
        'page': 1,
        'sparkline': 'false'
    }
    
    cache_key_params = {'n': n} # Параметры для ключа кэша (могут быть подмножеством api_params)
    cache_key = generate_cache_key('coingecko_top_n', cache_key_params)
    
    # Попытка загрузить из кэша
    cached_data, cache_ts = load_from_cache(cache_key, is_dataframe=False) # Данные не DataFrame
    if cached_data is not None and is_cache_valid(cache_ts, COINGECKO_CACHE_DURATION_MINUTES):
        print(f"CoinGecko: Данные для top {n} монет загружены из кэша.")
        return cached_data

    print(f"CoinGecko: Запрос топ {n} монет из API (кэш отсутствует или невалиден).")
    # URL API CoinGecko для получения рыночных данных
    url = "https://api.coingecko.com/api/v3/coins/markets"

    try:
        # Выполнение GET-запроса к API
        response = requests.get(url, params=api_params)
        response.raise_for_status()
        data = response.json()
        symbols = [coin['symbol'].upper() for coin in data]
        
        # Сохранение в кэш
        save_to_cache(cache_key, symbols)
        return symbols

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к CoinGecko API: {e}")
        return []
    except ValueError as e:
        print(f"Ошибка при парсинге JSON ответа от CoinGecko API: {e}")
        return []
    except KeyError as e:
        print(f"Ошибка: в ответе API отсутствует ожидаемый ключ: {e}")
        return []

# --- OKX Functions ---
OKX_OPERATIONAL_CACHE_DURATION_MINUTES = 15 # 15 минут для операционного кэша

def format_symbol_for_okx(symbol, quote_currency='USDT'):
    """
    Форматирует символ базовой криптовалюты и котировочной валюты для OKX API.

    :param symbol: Символ базовой криптовалюты (например, 'BTC', 'ETH').
    :type symbol: str
    :param quote_currency: Символ котировочной валюты (например, 'USDT', 'USDC', 'BTC').
                           По умолчанию 'USDT'.
    :type quote_currency: str
    :return: Строка отформатированного символа для OKX (например, 'BTC-USDT').
    :rtype: str
    """
    return f"{symbol.upper()}-{quote_currency.upper()}"

def get_okx_historical_data(okx_symbol, timeframe, limit=100, use_sdk_if_available=False):
    """
    Получает исторические данные OHLCV для указанного символа и таймфрейма с OKX,
    используя операционный кэш и сохраняя данные в CSV для долгосрочного хранения.

    Предпочтительно использует прямые HTTP запросы.
    Может использовать OKX SDK, если use_sdk_if_available=True и SDK установлен.

    :param okx_symbol: Отформатированный символ для OKX (например, 'BTC-USDT').
    :type okx_symbol: str
    :param timeframe: Таймфрейм свечей (например, '1m', '5m', '1H', '1D').
                      OKX использует 'bar' параметр для этого (например, '1H', '4H', '1D').
    :type timeframe: str
    :param limit: Количество свечей для получения (OKX может иметь макс. лимит, например, 100 или 300).
    :type limit: int
    :param use_sdk_if_available: Если True и OKX SDK доступен, использовать SDK. По умолчанию False.
    :type use_sdk_if_available: bool
    :return: pandas DataFrame с колонками ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
             или None в случае ошибки. Данные отсортированы по timestamp (новые сначала, как отдает API).
    :rtype: pd.DataFrame or None
    """
    
    cache_key_params = {'instId': okx_symbol, 'bar': timeframe, 'limit': limit}
    # Добавляем флаг SDK в ключ кэша, т.к. результаты могут отличаться или иметь разные ошибки
    if use_sdk_if_available and OKX_SDK_AVAILABLE:
        cache_key_params['sdk'] = True
    
    cache_key = generate_cache_key('okx_historical', cache_key_params)

    # Попытка загрузить из операционного кэша
    cached_df, cache_ts = load_from_cache(cache_key, is_dataframe=True)
    if cached_df is not None and is_cache_valid(cache_ts, OKX_OPERATIONAL_CACHE_DURATION_MINUTES):
        print(f"OKX: Данные для {okx_symbol} ({timeframe}, limit {limit}) загружены из операционного кэша.")
        # Важно: если данные из кэша, они уже были сохранены в CSV при первом получении.
        # Нет необходимости повторно вызывать save_historical_data_to_csv здесь.
        return cached_df

    print(f"OKX: Запрос данных для {okx_symbol} ({timeframe}, limit {limit}) из API (кэш отсутствует или невалиден).")
    
    # Колонки для DataFrame в соответствии с требованиями
    # columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_quote_currency'] # Убрал, т.к. не используется
    df_to_return = None

    if use_sdk_if_available and OKX_SDK_AVAILABLE:
        # --- Метод с использованием OKX Python SDK ---
        print("OKX: Попытка использования OKX SDK...")
        market_api = OkxMarketAPI(flag='0') # '0' для реальной торговли, '1' для демо
        try:
            result = market_api.get_history_candlesticks(
                instId=okx_symbol, bar=timeframe, limit=str(limit)
            )
            if result['code'] == '0' and result.get('data'):
                data = result['data']
                if not data:
                    print(f"OKX SDK: Нет данных для {okx_symbol} с таймфреймом {timeframe}.")
                    # Возвращаем пустой DataFrame, чтобы кэшировать этот "отсутствие данных" результат
                    df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    df_data = [[item[0], item[1], item[2], item[3], item[4], item[5]] for item in data]
                    df = pd.DataFrame(df_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df_to_return = df
            else:
                print(f"Ошибка от OKX SDK: {result.get('msg', 'Неизвестная ошибка SDK')}")
                # Не возвращаем None сразу, чтобы кэшировать ошибку (пустой DataFrame)
                df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        except Exception as e:
            print(f"Исключение при использовании OKX SDK: {e}")
            df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    else: # --- Метод с использованием прямого HTTP запроса (requests) ---
        print("OKX: Использование прямого HTTP запроса...")
        url = "https://www.okx.com/api/v5/market/history-candles"
        params = {'instId': okx_symbol, 'bar': timeframe, 'limit': str(limit)}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            if data_json.get('code') == '0' and data_json.get('data'):
                ohlcv_data = data_json['data']
                if not ohlcv_data:
                    print(f"OKX API: Нет данных для {okx_symbol} с таймфреймом {timeframe}.")
                    df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                else:

            # Структура ответа OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            # Нам нужны первые 6: ts, o, h, l, c, vol
            # В документации указано 7 полей для history-candles, но по факту их может быть больше.
            # ts, o, h, l, c, vol, volCcy (объем в базовой валюте), volCcyQuote (объем в котировочной валюте)
            # Мы берем 'vol' (объем в базовой валюте) и 'volCcyQuote' (объем в котировочной валюте)
            # Структура ответа OKX /api/v5/market/history-candles:
            # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            # Индексы:
            # 0: timestamp (мс)
            # 1: open
            # 2: high
            # 3: low
            # 4: close
            # 5: vol (Объем торгов в базовой валюте или в контрактах в зависимости от типа инструмента. Для SPOT это базовая валюта)
            # 6: volCcy (Объем торгов в котировочной валюте для SPOT рынков (например USDT для BTC-USDT). Для деривативов это базовая валюта.)
            # 7: volCcyQuote (Объем торгов в котировочной валюте. Только применимо к SWAP и FUTURES.)
            # 8: confirm (0 = свеча не закрыта, 1 = свеча закрыта) - не всегда есть, зависит от свечи

            # Нам нужны ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    processed_data = []
                    for candle_data in ohlcv_data:
                        if len(candle_data) >= 6:
                            processed_data.append([
                                candle_data[0], candle_data[1], candle_data[2],
                                candle_data[3], candle_data[4], candle_data[5]
                            ])
                        else:
                            print(f"Предупреждение: получен неполный набор данных для свечи: {candle_data}")
                    
                    df = pd.DataFrame(processed_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df_to_return = df
            else:
                error_message = data_json.get('msg', 'Неизвестная ошибка API')
                if data_json.get('code') == '51001': # Parameter instId error
                    print(f"Ошибка от OKX API для {okx_symbol}: {error_message} (пара не существует/неактивна?).")
                else:
                    print(f"Ошибка от OKX API ({data_json.get('code')}): {error_message}")
                df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        except requests.exceptions.Timeout:
            print(f"Тайм-аут при запросе к OKX API для {okx_symbol}.")
            df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к OKX API для {okx_symbol}: {e}")
            df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except ValueError as e: # JSONDecodeError
            print(f"Ошибка при парсинге JSON ответа от OKX API для {okx_symbol}: {e}")
            df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e: # Catch-all for other unexpected errors
            print(f"Непредвиденная ошибка при получении данных с OKX для {okx_symbol} (HTTP): {e}")
            df_to_return = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # После получения данных (или ошибки, приводящей к пустому df_to_return)
    if df_to_return is not None:
        # Сохраняем в операционный кэш (даже если это пустой DataFrame из-за ошибки API)
        save_to_cache(cache_key, df_to_return) # is_dataframe=True учтено в save_to_cache
        
        # Если данные не пустые, сохраняем их в долгосрочное хранилище CSV
        if not df_to_return.empty:
            save_historical_data_to_csv(df_to_return, okx_symbol, timeframe)
            print(f"OKX: Данные для {okx_symbol} ({timeframe}) также сохранены/обновлены в CSV.")
        
        # Если df_to_return пустой из-за ошибки API, и мы не хотим возвращать пустой DataFrame,
        # а хотим вернуть None, чтобы указать на сбой, можно сделать так:
        if df_to_return.empty and not (use_sdk_if_available and OKX_SDK_AVAILABLE and result['code'] == '0' and not result.get('data') ) and not (not (use_sdk_if_available and OKX_SDK_AVAILABLE) and data_json.get('code') == '0' and not data_json.get('data')): # Проверяем, что это не случай "нет данных" от API
             return None # Ошибка получения данных

        return df_to_return
    
    return None # Если что-то пошло совсем не так

if __name__ == '__main__':
    print("--- Демонстрация функций API с кэшированием ---")

    # --- CoinGecko кэширование ---
    print("\n--- Тестирование CoinGecko get_top_n_coins_coingecko ---")
    print("Первый вызов (n=7):")
    top_7_coins = get_top_n_coins_coingecko(7)
    if top_7_coins: print(f"Топ 7: {top_7_coins}")

    print("\nВторой вызов (n=7, должен быть из кэша):")
    start_time = time.time()
    top_7_coins_cached = get_top_n_coins_coingecko(7)
    end_time = time.time()
    if top_7_coins_cached: print(f"Топ 7 (кэш): {top_7_coins_cached}")
    print(f"Время выполнения второго вызова: {end_time - start_time:.4f} сек")

    print("\nТретий вызов (n=3, должен быть из API, новый ключ кэша):")
    top_3_coins = get_top_n_coins_coingecko(3)
    if top_3_coins: print(f"Топ 3: {top_3_coins}")

    # --- OKX кэширование и CSV ---
    print("\n--- Тестирование OKX get_okx_historical_data ---")
    btc_pair = format_symbol_for_okx('BTC') # BTC-USDT
    eth_pair = format_symbol_for_okx('ETH') # ETH-USDT
    timeframe_1h = '1H'
    timeframe_5m = '5m' # Другой таймфрейм для разнообразия

    # 1. Получение BTC-USDT 1H (первый вызов, из API, сохранение в кэш и CSV)
    print(f"\n1. Получение {btc_pair} {timeframe_1h} (limit 5) - первый вызов:")
    btc_h1_data = get_okx_historical_data(btc_pair, timeframe_1h, limit=5)
    if btc_h1_data is not None and not btc_h1_data.empty:
        print(f"Данные для {btc_pair} ({timeframe_1h}):\n{btc_h1_data.head()}")
    elif btc_h1_data is not None and btc_h1_data.empty :
        print(f"Для {btc_pair} ({timeframe_1h}) API вернул пустой набор данных.")
    else:
        print(f"Не удалось получить данные для {btc_pair} ({timeframe_1h}).")

    # 2. Повторное получение BTC-USDT 1H (должно быть из операционного кэша)
    print(f"\n2. Получение {btc_pair} {timeframe_1h} (limit 5) - второй вызов (из кэша):")
    start_time = time.time()
    btc_h1_data_cached = get_okx_historical_data(btc_pair, timeframe_1h, limit=5)
    end_time = time.time()
    if btc_h1_data_cached is not None and not btc_h1_data_cached.empty:
        print(f"Данные из кэша для {btc_pair} ({timeframe_1h}):\n{btc_h1_data_cached.head()}")
    print(f"Время выполнения второго вызова: {end_time - start_time:.4f} сек")

    # 3. Получение ETH-USDT 5m (новый символ/таймфрейм, из API, сохранение)
    print(f"\n3. Получение {eth_pair} {timeframe_5m} (limit 3) - первый вызов:")
    eth_5m_data = get_okx_historical_data(eth_pair, timeframe_5m, limit=3)
    if eth_5m_data is not None and not eth_5m_data.empty:
        print(f"Данные для {eth_pair} ({timeframe_5m}):\n{eth_5m_data.head()}")

    # 4. Загрузка из CSV для проверки
    print(f"\n4. Загрузка сохраненных данных {btc_pair} ({timeframe_1h}) из CSV:")
    loaded_btc_csv = load_historical_data_from_csv(btc_pair, timeframe_1h)
    if loaded_btc_csv is not None and not loaded_btc_csv.empty:
        print(f"Загружено из CSV для {btc_pair} ({timeframe_1h}):\n{loaded_btc_csv.tail()}") # tail, т.к. сохраняем отсортированным
    else:
        print(f"Не удалось загрузить CSV для {btc_pair} ({timeframe_1h}) или файл пуст.")
        
    print(f"\n5. Загрузка сохраненных данных {eth_pair} ({timeframe_5m}) из CSV:")
    loaded_eth_csv = load_historical_data_from_csv(eth_pair, timeframe_5m)
    if loaded_eth_csv is not None and not loaded_eth_csv.empty:
        print(f"Загружено из CSV для {eth_pair} ({timeframe_5m}):\n{loaded_eth_csv.tail()}")
    else:
        print(f"Не удалось загрузить CSV для {eth_pair} ({timeframe_5m}) или файл пуст.")

    # Демонстрация случая, когда API возвращает пустые данные (например, для неторгуемой пары или нового листинга)
    # Для этого можно использовать пару, которая вряд ли имеет данные, или очень маленький limit
    print(f"\n6. Попытка получить данные для несуществующей пары XYZ-USDT (ожидается пустой DataFrame или None):")
    non_existent_pair = "XYZ-USDT"
    bad_data = get_okx_historical_data(non_existent_pair, timeframe_1h, limit=5)
    if bad_data is None:
        print(f"Получен None для {non_existent_pair}, что указывает на ошибку API (не 'нет данных').")
    elif bad_data.empty:
        print(f"Получен пустой DataFrame для {non_existent_pair}, как и ожидалось (API вернул 'нет данных' или ошибка была обработана как пустой DF).")
        # Проверим, что в CSV не сохранилось
        loaded_bad_csv = load_historical_data_from_csv(non_existent_pair, timeframe_1h)
        if loaded_bad_csv is None:
            print(f"CSV для {non_existent_pair} не существует или пуст, что корректно.")
        else:
            print(f"ОШИБКА: CSV для {non_existent_pair} был создан, хотя не должен был.")
    else:
        print(f"Данные для {non_existent_pair}:\n{bad_data}")

    print("\n--- Демонстрация завершена ---")
