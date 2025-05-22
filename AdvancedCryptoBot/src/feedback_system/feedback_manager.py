import pandas as pd
import os
import json
from datetime import datetime

# Определение пути к файлу логов относительно текущего файла
# feedback_manager.py находится в AdvancedCryptoBot/src/feedback_system/
# Данные должны быть в AdvancedCryptoBot/data/
# Таким образом, ../../data/signals_log.csv
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_LOG_FILE = os.path.join(_BASE_DIR, '..', '..', 'data', 'signals_log.csv')

LOG_COLUMNS = [
    'signal_id', 'timestamp_generated', 'coin_symbol', 'signal_type', 
    'entry_price', 'stop_loss', 'take_profit', 'confidence', 
    'model_features_json', 'predicted_class', 'predicted_probability', 
    'outcome_timestamp', 'outcome_price_at_tp', 'outcome_price_at_sl', 
    'actual_outcome', 'feedback_notes'
]

def initialize_signals_log():
    """
    Инициализирует файл лога сигналов.
    Проверяет, существует ли файл SIGNALS_LOG_FILE.
    Если нет, создает его с LOG_COLUMNS в качестве заголовка.
    """
    log_dir = os.path.dirname(SIGNALS_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Создана директория для логов: {log_dir}")

    if not os.path.exists(SIGNALS_LOG_FILE):
        df = pd.DataFrame(columns=LOG_COLUMNS)
        df.to_csv(SIGNALS_LOG_FILE, index=False, encoding='utf-8')
        print(f"Файл лога сигналов инициализирован: {SIGNALS_LOG_FILE}")
    else:
        print(f"Файл лога сигналов уже существует: {SIGNALS_LOG_FILE}")

def log_generated_signal(signal_id, timestamp_generated, coin_symbol, signal_type, 
                         entry_price, stop_loss, take_profit, confidence, 
                         model_features_dict, predicted_class, predicted_probability):
    """
    Логирует сгенерированный торговый сигнал в CSV файл.

    :param signal_id: Уникальный идентификатор сигнала.
    :param timestamp_generated: Временная метка генерации сигнала (datetime object or ISO string).
    :param coin_symbol: Символ криптовалюты.
    :param signal_type: Тип сигнала (BUY, SELL, LONG, SHORT).
    :param entry_price: Цена входа.
    :param stop_loss: Уровень стоп-лосса.
    :param take_profit: Уровень тейк-профита.
    :param confidence: Уверенность модели в сигнале.
    :param model_features_dict: Словарь с признаками, использованными для предсказания.
    :param predicted_class: Предсказанный класс (например, 1 для UP, 0 для DOWN).
    :param predicted_probability: Вероятность предсказанного класса.
    """
    try:
        # Преобразование словаря признаков в JSON строку
        # Обработка datetime объектов в словаре признаков, если они есть
        def json_serializable_features(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            # Добавьте другие типы, если необходимо, например, np.int64 -> int
            if hasattr(obj, 'dtype') and ('int64' in str(obj.dtype) or 'float64' in str(obj.dtype)):
                 return str(obj) # Преобразовать в строку, чтобы избежать проблем с JSON
            return str(obj) # Общий случай, если что-то не сериализуется

        model_features_json = json.dumps(model_features_dict, default=json_serializable_features)
        
        # Преобразование timestamp_generated в ISO строку, если это datetime объект
        if isinstance(timestamp_generated, (datetime, pd.Timestamp)):
            ts_generated_str = timestamp_generated.isoformat()
        else:
            ts_generated_str = str(timestamp_generated)


        new_signal_data = {
            'signal_id': [signal_id],
            'timestamp_generated': [ts_generated_str],
            'coin_symbol': [coin_symbol],
            'signal_type': [signal_type],
            'entry_price': [entry_price],
            'stop_loss': [stop_loss],
            'take_profit': [take_profit],
            'confidence': [confidence],
            'model_features_json': [model_features_json],
            'predicted_class': [predicted_class],
            'predicted_probability': [predicted_probability],
            'outcome_timestamp': [pd.NA], # Заполнители для будущих данных
            'outcome_price_at_tp': [pd.NA],
            'outcome_price_at_sl': [pd.NA],
            'actual_outcome': [pd.NA],
            'feedback_notes': ['']
        }
        
        new_row_df = pd.DataFrame(new_signal_data, columns=LOG_COLUMNS)
        
        # Добавление строки в CSV. Создание файла с заголовком, если он не существует.
        file_exists = os.path.exists(SIGNALS_LOG_FILE)
        new_row_df.to_csv(SIGNALS_LOG_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8')
        
        print(f"Сигнал {signal_id} для {coin_symbol} успешно залогирован.")

    except Exception as e:
        print(f"Ошибка при логировании сигнала {signal_id} для {coin_symbol}: {e}")
        # Можно добавить более детальное логирование ошибки, если нужно

def add_manual_feedback(signal_id, actual_outcome, feedback_notes="", outcome_timestamp=None):
    """
    Добавляет ручную обратную связь к залогированному сигналу.
    (Это базовый пример, для реального использования может потребоваться более надежная реализация)

    :param signal_id: ID сигнала для обновления.
    :param actual_outcome: Фактический исход сигнала (например, 'TP_HIT', 'SL_HIT', 'MANUAL_CLOSE_PROFIT', 'MANUAL_CLOSE_LOSS').
    :param feedback_notes: Текстовые примечания к обратной связи.
    :param outcome_timestamp: Временная метка исхода сигнала. Если None, используется текущее время.
    """
    try:
        if not os.path.exists(SIGNALS_LOG_FILE):
            print(f"Ошибка: Файл лога сигналов {SIGNALS_LOG_FILE} не найден. Невозможно добавить обратную связь.")
            return

        signals_df = pd.read_csv(SIGNALS_LOG_FILE, encoding='utf-8')
        
        # Ищем сигнал по ID. Убедимся, что signal_id в CSV и передаваемый signal_id одного типа (например, оба строки)
        signals_df['signal_id'] = signals_df['signal_id'].astype(str)
        signal_index = signals_df[signals_df['signal_id'] == str(signal_id)].index

        if not signal_index.empty:
            idx = signal_index[0] # Берем первый найденный индекс
            
            signals_df.loc[idx, 'actual_outcome'] = actual_outcome
            signals_df.loc[idx, 'feedback_notes'] = feedback_notes
            
            if outcome_timestamp:
                signals_df.loc[idx, 'outcome_timestamp'] = outcome_timestamp.isoformat() if isinstance(outcome_timestamp, datetime) else str(outcome_timestamp)
            else:
                signals_df.loc[idx, 'outcome_timestamp'] = datetime.now().isoformat()
            
            # Сохраняем обновленный DataFrame обратно в CSV
            signals_df.to_csv(SIGNALS_LOG_FILE, index=False, encoding='utf-8')
            print(f"Обратная связь для сигнала {signal_id} успешно добавлена/обновлена.")
        else:
            print(f"Ошибка: Сигнал с ID {signal_id} не найден в логе.")

    except pd.errors.EmptyDataError:
        print(f"Ошибка: Файл лога сигналов {SIGNALS_LOG_FILE} пуст.")
    except Exception as e:
        print(f"Ошибка при добавлении обратной связи для сигнала {signal_id}: {e}")


if __name__ == '__main__':
    print("--- Демонстрация модуля feedback_manager ---")

    # 1. Инициализация файла логов
    initialize_signals_log()

    # 2. Логирование нескольких тестовых сигналов
    print("\nЛогирование тестовых сигналов...")
    
    # Сигнал 1
    dummy_features1 = {'SMA_20': 24000, 'RSI_14': 65.5, 'news_sentiment_compound': 0.25}
    # Используем datetime объекты для timestamp_generated
    ts1 = datetime(2023, 10, 26, 10, 0, 0) 
    log_generated_signal(
        signal_id="SIG20231026100000_BTCUSDT",
        timestamp_generated=ts1,
        coin_symbol="BTC-USDT",
        signal_type="BUY",
        entry_price=25000.0,
        stop_loss=24500.0,
        take_profit=26500.0,
        confidence=0.75,
        model_features_dict=dummy_features1,
        predicted_class=1,
        predicted_probability=0.75
    )

    # Сигнал 2
    dummy_features2 = {'SMA_20': 1750, 'RSI_14': 30.1, 'news_sentiment_compound': -0.5}
    ts2 = pd.Timestamp.now(tz='UTC') # Использование pd.Timestamp с таймзоной
    log_generated_signal(
        signal_id=f"SIG{ts2.strftime('%Y%m%d%H%M%S%f')}_ETHUSDT", # Более уникальный ID с микросекундами
        timestamp_generated=ts2,
        coin_symbol="ETH-USDT",
        signal_type="SELL",
        entry_price=1800.50,
        stop_loss=1850.25,
        take_profit=1700.00,
        confidence=0.68,
        model_features_dict=dummy_features2,
        predicted_class=0,
        predicted_probability=0.68
    )
    
    # Сигнал 3 (для демонстрации того, что файл не перезаписывается, а добавляется)
    dummy_features3 = {'SMA_20': 1.05, 'RSI_14': 55.0, 'news_sentiment_compound': 0.05}
    ts3 = "2023-10-26T12:30:00Z" # ISO строка для timestamp_generated
    log_generated_signal(
        signal_id="SIG20231026123000_ADAUSDT",
        timestamp_generated=ts3,
        coin_symbol="ADA-USDT",
        signal_type="BUY",
        entry_price=1.0,
        stop_loss=0.95,
        take_profit=1.15,
        confidence=0.55,
        model_features_dict=dummy_features3,
        predicted_class=1,
        predicted_probability=0.55
    )

    # 3. Демонстрация добавления обратной связи
    print("\nДобавление обратной связи для сигнала SIG20231026100000_BTCUSDT...")
    add_manual_feedback(
        signal_id="SIG20231026100000_BTCUSDT",
        actual_outcome="TP_HIT", # Тейк-профит достигнут
        feedback_notes="Рынок пошел по прогнозу, хороший вход.",
        outcome_timestamp=datetime(2023, 10, 26, 14, 30, 0) # Время исхода
    )

    print("\nДобавление обратной связи для несуществующего сигнала...")
    add_manual_feedback(
        signal_id="NONEXISTENT_SIGNAL_ID",
        actual_outcome="SL_HIT",
        feedback_notes="Это не должно сработать."
    )

    # 4. Печать содержимого файла лога
    print(f"\nСодержимое файла {SIGNALS_LOG_FILE}:")
    try:
        if os.path.exists(SIGNALS_LOG_FILE):
            log_df_content = pd.read_csv(SIGNALS_LOG_FILE, encoding='utf-8')
            # Для лучшего отобратия JSON и длинных строк
            with pd.option_context('display.max_colwidth', 100, 'display.max_rows', None, 'display.width', 1000):
                print(log_df_content)
        else:
            print("Файл лога не найден.")
    except pd.errors.EmptyDataError:
        print("Файл лога пуст.")
    except Exception as e:
        print(f"Ошибка при чтении файла лога: {e}")
        
    print("\n--- Демонстрация feedback_manager завершена ---")
