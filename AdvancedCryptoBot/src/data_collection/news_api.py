import requests
import pandas as pd
from datetime import datetime, timezone # Добавлен timezone
import os
import json # Для отладки и сохранения/загрузки сложных структур в кэш, если потребуется
import feedparser # Для RSS
import time # для time.mktime при конвертации struct_time

# Импорт функций кэширования из соседнего модуля
# Предполагается, что data_caching.py находится в той же директории
try:
    from .data_caching import (
        generate_cache_key,
        save_to_cache,
        load_from_cache,
        is_cache_valid
    )
except ImportError:
    # Если запускается как скрипт, а не как часть пакета
    print("Не удалось выполнить относительный импорт data_caching. Попытка прямого импорта...")
    from data_caching import (
        generate_cache_key,
        save_to_cache,
        load_from_cache,
        is_cache_valid
    )


# Определяем базовые пути относительно этого файла для сохранения CSV
# .../AdvancedCryptoBot/src/data_collection/news_api.py
# .../AdvancedCryptoBot/data/news/
DEFAULT_NEWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'news'))
os.makedirs(DEFAULT_NEWS_DIR, exist_ok=True) # Убедимся, что директория существует

NEWSAPI_CACHE_DURATION_MINUTES = 60 # 1 час
RSS_CACHE_DURATION_MINUTES = 45 # Кэш для RSS фидов, можно сделать короче, т.к. они чаще обновляются

# Словарь RSS-лент
RSS_FEEDS = {
    'Cointelegraph': 'https://cointelegraph.com/rss',
    'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'Decrypt': 'https://decrypt.co/feed',
    # 'TheBlock': 'https://www.theblock.co/rss.xml', # TheBlock часто требует подписку или спец. доступ
    'NullTX': 'https://nulltx.com/feed/', # Еще один источник для примера
    'CryptoSlate': 'https://cryptoslate.com/feed/'
}


def fetch_crypto_news_newsapi(api_key, keywords, n_articles=20, language='en'):
    """
    Получает новости о криптовалютах с NewsAPI, используя кэширование.

    :param api_key: Ключ API для NewsAPI.
    :type api_key: str
    :param keywords: Ключевые слова для поиска (например, "bitcoin OR ethereum").
    :type keywords: str
    :param n_articles: Целевое количество статей (максимум 100 для NewsAPI).
    :type n_articles: int
    :param language: Язык новостей (например, 'en').
    :type language: str
    :return: Список словарей, где каждый словарь представляет статью.
             Возвращает пустой список в случае ошибки.
    :rtype: list[dict]
    """
    # Параметры для ключа кэша
    cache_key_params = {
        'keywords': keywords,
        'n_articles': n_articles,
        'language': language,
        'source': 'newsapi' # Чтобы отличать от других возможных источников новостей
    }
    cache_key = generate_cache_key('crypto_news', cache_key_params)

    # Попытка загрузить из кэша
    # Новости это список словарей, не DataFrame на этом этапе
    cached_articles, cache_ts = load_from_cache(cache_key, is_dataframe=False) 
    if cached_articles is not None and is_cache_valid(cache_ts, NEWSAPI_CACHE_DURATION_MINUTES):
        print(f"NewsAPI: Новости для '{keywords}' (до {n_articles} статей) загружены из кэша.")
        # Важно: Убедимся, что 'publishedAt' в кэше уже datetime объекты или конвертируем их
        for article in cached_articles:
            if isinstance(article.get('publishedAt'), str):
                try:
                    article['publishedAt'] = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                except ValueError:
                     article['publishedAt'] = pd.to_datetime(article['publishedAt'], errors='coerce') # Более общая попытка
        return cached_articles

    print(f"NewsAPI: Запрос новостей для '{keywords}' (до {n_articles} статей) из API (кэш отсутствует или невалиден).")
    
    base_url = "https://newsapi.org/v2/everything"
    # NewsAPI pageSize имеет максимум 100
    page_size = min(n_articles, 100)

    params = {
        'q': keywords,
        'language': language,
        'pageSize': page_size,
        'sortBy': 'publishedAt', # или 'relevancy', 'popularity'
        'apiKey': api_key
    }

    extracted_articles = []

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Вызовет исключение для HTTP-ошибок (4xx, 5xx)
        
        news_data = response.json()
        
        if news_data.get('status') == 'ok':
            articles_raw = news_data.get('articles', [])
            
            for article_raw in articles_raw:
                # Проверка наличия основных полей
                if not all(k in article_raw for k in ['title', 'description', 'url', 'publishedAt']) or \
                   not article_raw.get('source') or not article_raw['source'].get('name'):
                    # print(f"Пропуск статьи из-за отсутствия ключевых полей: {article_raw.get('title')}")
                    continue

                try:
                    # Преобразование publishedAt в datetime
                    # NewsAPI возвращает UTC время, например, "2023-10-26T10:30:00Z"
                    # datetime.fromisoformat ожидает, что Z будет заменен на +00:00 для Python < 3.11
                    published_at_dt = datetime.fromisoformat(article_raw['publishedAt'].replace('Z', '+00:00'))
                except ValueError as ve:
                    print(f"Ошибка конвертации даты для статьи: {article_raw.get('title')}. Дата: {article_raw['publishedAt']}. Ошибка: {ve}")
                    published_at_dt = None # или pd.NaT, если планируем сразу в DataFrame
                
                extracted_articles.append({
                    'title': article_raw['title'],
                    'description': article_raw['description'],
                    'url': article_raw['url'],
                    'publishedAt': published_at_dt, # Уже datetime объект
                    'source_name': article_raw['source']['name']
                })
            
            # Сохранение в кэш (datetime объекты будут корректно обработаны json.dump если они ISO строки)
            # Для кэша лучше сохранять строки ISO, чтобы избежать проблем с форматами при загрузке
            articles_to_cache = []
            for art in extracted_articles:
                art_copy = art.copy()
                if isinstance(art_copy['publishedAt'], datetime):
                    art_copy['publishedAt'] = art_copy['publishedAt'].isoformat()
                articles_to_cache.append(art_copy)

            save_to_cache(cache_key, articles_to_cache) # is_dataframe=False по умолчанию
            return extracted_articles # Возвращаем с datetime объектами
            
        else:
            error_msg = news_data.get('message', 'Неизвестная ошибка от NewsAPI')
            print(f"Ошибка от NewsAPI: {error_msg} (Код: {news_data.get('code')})")
            return []

    except requests.exceptions.Timeout:
        print(f"Тайм-аут при запросе к NewsAPI для '{keywords}'.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к NewsAPI для '{keywords}': {e}")
        return []
    except ValueError as e: # Ошибка парсинга JSON
        print(f"Ошибка при парсинге JSON ответа от NewsAPI для '{keywords}': {e}")
        return []
    except Exception as e:
        print(f"Непредвиденная ошибка при получении новостей с NewsAPI для '{keywords}': {e}")
        return []

def save_news_to_csv(news_articles_list, filename='news_articles.csv', base_dir=DEFAULT_NEWS_DIR):
    """
    Сохраняет список новостных статей в CSV файл, добавляя новые и удаляя дубликаты.

    :param news_articles_list: Список словарей статей (из fetch_crypto_news_newsapi).
    :type news_articles_list: list[dict]
    :param filename: Имя CSV файла.
    :type filename: str
    :param base_dir: Базовая директория для сохранения CSV.
    :type base_dir: str
    """
    if not news_articles_list:
        # print("Список новостей пуст, сохранение в CSV не требуется.") # Для отладки
        return

    # Убедимся, что директория существует (на всякий случай, хотя создается при импорте)
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename)

    new_df = pd.DataFrame(news_articles_list)
    
    # Важно: Убедимся, что 'publishedAt' в new_df является datetime объектом перед операциями
    if 'publishedAt' in new_df.columns:
        new_df['publishedAt'] = pd.to_datetime(new_df['publishedAt'], errors='coerce')

    # Колонки для CSV в нужном порядке
    columns_order = ['publishedAt', 'title', 'description', 'url', 'source_name']
    new_df = new_df[columns_order] # Переупорядочиваем/выбираем нужные колонки

    combined_df = new_df

    if os.path.exists(filepath):
        try:
            existing_df = pd.read_csv(filepath, parse_dates=['publishedAt'])
            if not existing_df.empty:
                # Объединяем существующие и новые данные
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"Файл CSV {filepath} пуст. Будет перезаписан новыми данными.")
        except Exception as e:
            print(f"Ошибка при чтении существующего CSV файла {filepath}: {e}. Файл будет перезаписан.")
    
    # Удаляем дубликаты по 'url', оставляя первое вхождение (самые старые данные, если были дубликаты).
    # Для новостей обычно лучше 'first', так как контент статьи не меняется.
    combined_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    
    # Сортируем данные по 'publishedAt' в убывающем порядке (самые новые сначала)
    combined_df.sort_values(by='publishedAt', ascending=False, inplace=True)
    
    try:
        combined_df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Новости сохранены/обновлены в: {filepath} ({len(combined_df)} статей)")
    except IOError as e:
        print(f"Ошибка при сохранении новостей в CSV {filepath}: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка при сохранении новостей в CSV {filepath}: {e}")


if __name__ == '__main__':
    print("--- Демонстрация модуля news_api ---")
    
    # ВНИМАНИЕ: Для производственного использования ключ API должен быть защищен,
    # например, через переменные окружения или конфигурационные файлы.
    NEWSAPI_KEY = "e3318f2c424c49218e1792c2fcf22863" # Ключ предоставлен в задании
    
    if not NEWSAPI_KEY:
        print("Ключ NEWSAPI_KEY не установлен. Демонстрация не может быть выполнена.")
    else:
        keywords_query = (
            "bitcoin OR ethereum OR solana OR cardano OR ripple OR dogecoin OR tether OR "
            "binance coin OR polkadot OR avalanche OR cryptocurrency OR crypto OR web3 OR defi OR nft"
        )
        
        # --- Тестирование fetch_crypto_news_newsapi ---
        print("\n1. Получение новостей (первый вызов, n=5):")
        articles1 = fetch_crypto_news_newsapi(NEWSAPI_KEY, keywords_query, n_articles=5, language='en')
        if articles1:
            print(f"  Получено {len(articles1)} статей.")
            for i, article in enumerate(articles1[:2]): # Печатаем первые 2
                print(f"  Статья {i+1}: {article['title']} ({article['source_name']}) - {article['publishedAt']}")
        else:
            print("  Новости не получены.")

        print("\n2. Получение новостей (второй вызов, n=5, должен быть из кэша):")
        articles2 = fetch_crypto_news_newsapi(NEWSAPI_KEY, keywords_query, n_articles=5, language='en')
        if articles2:
            print(f"  Получено {len(articles2)} статей из кэша.")
            # Проверим, что данные те же (например, по URL первой статьи)
            if articles1 and articles2 and articles1[0]['url'] == articles2[0]['url']:
                print("  Данные из кэша совпадают с первым вызовом (проверено по URL).")
        else:
            print("  Новости не получены (ошибка кэширования?).")

        # --- Тестирование save_news_to_csv и последующей загрузки ---
        if articles1: # Если первый вызов был успешен
            csv_filename = "demo_crypto_news.csv" # Используем отдельный файл для демонстрации
            # Удаляем старый демо-файл, если он есть, для чистоты теста
            demo_filepath = os.path.join(DEFAULT_NEWS_DIR, csv_filename)
            if os.path.exists(demo_filepath):
                os.remove(demo_filepath)
                print(f"\nУдален старый файл {demo_filepath} для чистоты теста.")

            print(f"\n3. Сохранение {len(articles1)} статей в {csv_filename}:")
            save_news_to_csv(articles1, filename=csv_filename) # base_dir по умолчанию

            # Добавим еще немного "новых" статей (могут быть дубликатами, если API их вернул снова)
            print("\n4. Попытка получить еще немного статей (n=3) для демонстрации добавления:")
            # Изменим запрос немного, чтобы получить другие статьи (или те же, если API так решит)
            more_articles = fetch_crypto_news_newsapi(NEWSAPI_KEY, "blockchain finance", n_articles=3, language='en')
            if more_articles:
                print(f"  Получено {len(more_articles)} 'новых' статей.")
                print(f"  Сохранение этих статей в тот же {csv_filename} (с объединением и дедупликацией):")
                save_news_to_csv(more_articles, filename=csv_filename)
            else:
                print("  'Новые' статьи не получены.")

            print(f"\n5. Загрузка новостей из {csv_filename} для проверки:")
            try:
                loaded_df = pd.read_csv(demo_filepath, parse_dates=['publishedAt'])
                print(f"  Успешно загружено {len(loaded_df)} статей из {csv_filename}.")
                if not loaded_df.empty:
                    print("  Последние 5 загруженных статей (отсортированы по publishedAt desc):")
                    print(loaded_df[['publishedAt', 'title', 'source_name']].head())
                    
                    # Проверка на дубликаты по URL
                    if loaded_df['url'].duplicated().any():
                        print("  ПРЕДУПРЕЖДЕНИЕ: В CSV файле обнаружены дубликаты URL!")
                    else:
                        print("  Проверка на дубликаты URL в CSV: Дубликатов не найдено.")
                
                # Очистка: удаляем демо-файл
                # os.remove(demo_filepath)
                # print(f"\nТестовый файл {demo_filepath} удален.")

            except FileNotFoundError:
                print(f"  Ошибка: файл {demo_filepath} не найден.")
            except pd.errors.EmptyDataError:
                print(f"  Ошибка: файл {demo_filepath} пуст.")
            except Exception as e:
                print(f"  Ошибка при чтении CSV: {e}")
        else:
            print("\nПропуск тестирования сохранения в CSV, так как новости не были получены.")
            
    print("\n--- Демонстрация news_api завершена ---")


def fetch_news_from_rss(feed_url, source_name):
    """
    Получает новости из указанного RSS-фида, используя кэширование.

    :param feed_url: URL RSS-фида.
    :type feed_url: str
    :param source_name: Имя источника новостей (например, 'Cointelegraph').
    :type source_name: str
    :return: Список словарей, где каждый словарь представляет статью.
             Формат словаря: {'publishedAt': datetime, 'title': str, 
                              'description': str, 'url': str, 'source_name': str}
             Возвращает пустой список в случае ошибки.
    :rtype: list[dict]
    """
    cache_key_params = {'url': feed_url, 'source': 'rss'}
    cache_key = generate_cache_key(f"rss_{source_name.lower().replace(' ', '_')}", cache_key_params)

    # Попытка загрузить из кэша
    cached_articles, cache_ts = load_from_cache(cache_key, is_dataframe=False)
    if cached_articles is not None and is_cache_valid(cache_ts, RSS_CACHE_DURATION_MINUTES):
        print(f"RSS: Новости из '{source_name}' ({feed_url}) загружены из кэша.")
        # Конвертируем 'publishedAt' обратно в datetime, если это строка
        for article in cached_articles:
            if isinstance(article.get('publishedAt'), str):
                try:
                    article['publishedAt'] = datetime.fromisoformat(article['publishedAt'])
                except ValueError: # На случай, если формат немного отличается, но pd.to_datetime может справиться
                    article['publishedAt'] = pd.to_datetime(article['publishedAt'], errors='coerce').tz_localize('UTC')

        return cached_articles

    print(f"RSS: Запрос новостей из '{source_name}' ({feed_url}) (кэш отсутствует или невалиден).")
    
    extracted_articles = []
    try:
        # Установка user-agent может помочь с некоторыми фидами
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # feedparser сам делает запрос, но если бы мы делали через requests:
        # response = requests.get(feed_url, headers=headers, timeout=15)
        # feed = feedparser.parse(response.content)
        
        feed = feedparser.parse(feed_url, agent=headers.get('User-Agent')) # agent - это User-Agent для feedparser

        if feed.bozo: # Если bozo установлен в 1, значит, фид не был хорошо сформирован
            bozo_exception = feed.get("bozo_exception", "Неизвестная ошибка парсинга RSS")
            print(f"Ошибка парсинга RSS-фида {feed_url} от {source_name}: {bozo_exception}")
            # Иногда фид все равно содержит данные, можно попытаться их извлечь
            # return [] # Строгий вариант - не возвращать ничего при ошибке парсинга

        for entry in feed.entries:
            title = entry.get('title')
            link = entry.get('link')
            # Описание может быть в 'summary' или 'description'
            description = entry.get('summary') or entry.get('description')
            
            published_parsed_time = entry.get('published_parsed') # struct_time
            published_iso = entry.get('published') # Строка даты, если есть

            published_at_dt = None

            if published_parsed_time:
                try:
                    # struct_time может быть наивным UTC или содержать смещение.
                    # time.mktime предполагает, что struct_time в локальном времени.
                    # datetime.fromtimestamp(..., tz=timezone.utc) создаст datetime в UTC.
                    # Это самый распространенный случай для RSS.
                    timestamp_utc = time.mktime(published_parsed_time)
                    published_at_dt = datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)
                except Exception as e_parsed:
                    # print(f"Ошибка конвертации published_parsed_time для '{title}': {e_parsed}")
                    pass # Пробуем следующий метод

            if not published_at_dt and published_iso:
                try:
                    # Попытка распарсить строку даты, если она есть
                    dt_object = pd.to_datetime(published_iso, errors='coerce')
                    if pd.notna(dt_object):
                        if dt_object.tzinfo is None: # Если наивный datetime
                            published_at_dt = dt_object.tz_localize('UTC') # Предполагаем UTC
                        else: # Если уже с таймзоной
                            published_at_dt = dt_object.astimezone(timezone.utc) # Конвертируем в UTC
                except Exception as e_iso:
                    # print(f"Ошибка конвертации published_iso для '{title}': {e_iso}")
                    pass

            if not title or not link or not published_at_dt:
                # print(f"Пропуск записи из-за отсутствия title, link или published_at: {title[:30] if title else 'N/A'}")
                continue
            
            # Простой способ убрать HTML из описания, если нужно (не идеально, но для начала)
            if description:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(description, "html.parser")
                    description = soup.get_text()
                except Exception: # Если BeautifulSoup не установлен или ошибка
                    pass 
                description = description[:500] # Ограничим длину описания

            extracted_articles.append({
                'title': title,
                'description': description if description else '',
                'url': link,
                'publishedAt': published_at_dt, # datetime объект (UTC)
                'source_name': source_name
            })

        # Сохранение в кэш. Конвертируем datetime в ISO строку.
        articles_to_cache = []
        for art in extracted_articles:
            art_copy = art.copy()
            if isinstance(art_copy['publishedAt'], datetime):
                art_copy['publishedAt'] = art_copy['publishedAt'].isoformat()
            articles_to_cache.append(art_copy)
        
        save_to_cache(cache_key, articles_to_cache)
        return extracted_articles

    except Exception as e:
        print(f"Общая ошибка при получении или обработке RSS-фида {feed_url} от {source_name}: {e}")
        return []

if __name__ == '__main__':
    print("--- Демонстрация модуля news_api ---")
    
    # ВНИМАНИЕ: Для производственного использования ключ API должен быть защищен,
    # например, через переменные окружения или конфигурационные файлы.
    NEWSAPI_KEY = "e3318f2c424c49218e1792c2fcf22863" # Ключ предоставлен в задании
    
    # Общий список для всех новостей
    all_fetched_articles = []
    csv_filename = "demo_crypto_news_aggregated.csv" # Единый файл для всех источников
    demo_filepath = os.path.join(DEFAULT_NEWS_DIR, csv_filename)

    # Удаляем старый демо-файл, если он есть, для чистоты теста
    if os.path.exists(demo_filepath):
        os.remove(demo_filepath)
        print(f"\nУдален старый агрегированный файл {demo_filepath} для чистоты теста.")

    if not NEWSAPI_KEY:
        print("Ключ NEWSAPI_KEY не установлен. Демонстрация NewsAPI будет пропущена.")
    else:
        keywords_query = (
            "bitcoin OR ethereum OR solana OR cardano OR ripple OR dogecoin OR tether OR "
            "binance coin OR polkadot OR avalanche OR cryptocurrency OR crypto OR web3 OR defi OR nft"
        )
        
        # --- Тестирование fetch_crypto_news_newsapi ---
        print("\n--- NewsAPI Источник ---")
        print("1. Получение новостей NewsAPI (n=5):")
        articles1 = fetch_crypto_news_newsapi(NEWSAPI_KEY, keywords_query, n_articles=5, language='en')
        if articles1:
            print(f"  NewsAPI: Получено {len(articles1)} статей.")
            all_fetched_articles.extend(articles1)
            for i, article in enumerate(articles1[:1]): # Печатаем первую
                print(f"  Пример NewsAPI: {article['title']} ({article['source_name']}) - {article['publishedAt']}")
        else:
            print("  NewsAPI: Новости не получены.")

        print("\n2. Получение новостей NewsAPI (n=5, должен быть из кэша):")
        articles2 = fetch_crypto_news_newsapi(NEWSAPI_KEY, keywords_query, n_articles=5, language='en')
        # Не добавляем в all_fetched_articles, т.к. это те же данные для теста кэша
        if articles2: print(f"  NewsAPI: Получено {len(articles2)} статей из кэша.")

    # --- Тестирование RSS-фидов ---
    print("\n--- RSS Источники ---")
    for source, url in RSS_FEEDS.items():
        print(f"\nЗапрос RSS из: {source} ({url})")
        rss_articles = fetch_news_from_rss(url, source)
        if rss_articles:
            print(f"  {source}: Получено {len(rss_articles)} статей.")
            all_fetched_articles.extend(rss_articles)
            for i, article in enumerate(rss_articles[:1]): # Печатаем первую
                print(f"  Пример {source}: {article['title']} - {article['publishedAt']}")
        else:
            print(f"  {source}: Новости не получены или ошибка.")
        
        print(f"Запрос RSS из: {source} ({url}) - второй раз (из кэша)")
        rss_articles_cached = fetch_news_from_rss(url, source)
        if rss_articles_cached: print(f"  {source} (кэш): Получено {len(rss_articles_cached)} статей.")


    # --- Сохранение всех новостей в один CSV ---
    if all_fetched_articles:
        print(f"\n--- Сохранение всех ({len(all_fetched_articles)}) новостей в {csv_filename} ---")
        save_news_to_csv(all_fetched_articles, filename=csv_filename)

        print(f"\n--- Загрузка агрегированных новостей из {csv_filename} для проверки ---")
        try:
            loaded_df = pd.read_csv(demo_filepath, parse_dates=['publishedAt'])
            print(f"  Успешно загружено {len(loaded_df)} статей из {csv_filename}.")
            if not loaded_df.empty:
                print("  Последние 5 загруженных статей (отсортированы по publishedAt desc):")
                print(loaded_df[['publishedAt', 'title', 'source_name']].head())
                
                # Проверка на дубликаты по URL
                if loaded_df['url'].duplicated().any():
                    print("  ПРЕДУПРЕЖДЕНИЕ: В CSV файле обнаружены дубликаты URL!")
                else:
                    print("  Проверка на дубликаты URL в CSV: Дубликатов не найдено.")
            
            # Очистка: можно оставить файл для последующего просмотра или удалить
            # if os.path.exists(demo_filepath):
            #     os.remove(demo_filepath)
            #     print(f"\nТестовый файл {demo_filepath} удален.")

        except FileNotFoundError:
            print(f"  Ошибка: файл {demo_filepath} не найден.")
        except pd.errors.EmptyDataError:
            print(f"  Ошибка: файл {demo_filepath} пуст.")
        except Exception as e:
            print(f"  Ошибка при чтении CSV: {e}")
    else:
        print("\nНет новостей для сохранения в CSV.")
            
    print("\n--- Демонстрация news_api (включая RSS) завершена ---")
