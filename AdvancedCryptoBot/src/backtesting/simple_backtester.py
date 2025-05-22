import pandas as pd
import numpy as np
import os
from datetime import datetime # For dummy data

# --- Project Module Imports ---
try:
    from AdvancedCryptoBot.src.data_collection import market_data_api as market_api # Not directly used in backtester, but good for context
    from AdvancedCryptoBot.src.data_collection import news_api as news_api_module
    from AdvancedCryptoBot.src.data_collection import data_caching as data_caching # For loading data
    from AdvancedCryptoBot.src.feature_engineering import technical_indicators as ti_module
    from AdvancedCryptoBot.src.feature_engineering import sentiment_analysis as sa_module
    from AdvancedCryptoBot.src.ml_data_preparation import data_preparer as dp_module
    from AdvancedCryptoBot.src.ml_model import model_trainer as mt_module # For loading model
    from AdvancedCryptoBot.src.trading_logic import signal_generator as sg_module
except ImportError:
    print("Attempting fallback imports for simple_backtester.py...")
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    src_path = os.path.join(project_root, 'src')

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # from data_collection import market_data_api as market_api
    from data_collection import news_api as news_api_module
    from data_collection import data_caching as data_caching
    from feature_engineering import technical_indicators as ti_module
    from feature_engineering import sentiment_analysis as sa_module
    from ml_data_preparation import data_preparer as dp_module
    from ml_model import model_trainer as mt_module
    from trading_logic import signal_generator as sg_module


# --- Base directory definitions (for __main__) ---
_BASE_DIR_MAIN = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR_MAIN = os.path.join(_BASE_DIR_MAIN, '..', '..', 'data')
DEFAULT_MODELS_DIR_MAIN = os.path.join(_BASE_DIR_MAIN, '..', '..', 'models')


def run_backtest(config, model):
    """
    Запускает упрощенное историческое тестирование (бэктестинг).

    :param config: Словарь конфигурации бэктестинга.
    :type config: dict
    :param model: Предварительно загруженная обученная модель машинного обучения.
    :type model: object
    :return: Список словарей, представляющих совершенные сделки.
    :rtype: list[dict]
    """
    print(f"Запуск бэктестинга для символа: {config['symbol']}, таймфрейм: {config['timeframe']}")

    # 1. Загрузка исторических OHLCV данных
    print("Шаг 1: Загрузка исторических OHLCV данных...")
    ohlcv_df = data_caching.load_historical_data_from_csv(
        config['symbol'], 
        config['timeframe'], 
        base_dir=config['historical_data_dir']
    )
    if ohlcv_df is None or ohlcv_df.empty:
        print(f"Ошибка: Не удалось загрузить или данные OHLCV пусты для {config['symbol']}-{config['timeframe']}.")
        return []
    # Убедимся, что timestamp является индексом и отсортирован
    if 'timestamp' in ohlcv_df.columns:
         ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])
         ohlcv_df.set_index('timestamp', inplace=True)
    ohlcv_df.sort_index(inplace=True)


    # 2. Добавление технических индикаторов
    print("Шаг 2: Добавление технических индикаторов...")
    market_df_indicators = ti_module.add_all_indicators(ohlcv_df.copy())
    if market_df_indicators.empty:
        print("Ошибка: DataFrame пуст после добавления индикаторов.")
        return []

    # 3. Загрузка и обработка новостных данных (опционально)
    print("Шаг 3: Загрузка и обработка новостных данных...")
    news_df_processed = pd.DataFrame() # Пустой DataFrame по умолчанию
    if config.get('news_data_path') and os.path.exists(config['news_data_path']):
        try:
            news_df_raw = pd.read_csv(config['news_data_path'])
            if not news_df_raw.empty and 'publishedAt' in news_df_raw.columns:
                news_df_raw['publishedAt'] = pd.to_datetime(news_df_raw['publishedAt'], errors='coerce')
                news_df_raw.dropna(subset=['publishedAt'], inplace=True)
                news_df_processed = sa_module.add_sentiment_scores_to_news(news_df_raw.copy())
                print(f"Загружено и обработано {len(news_df_processed)} новостных статей.")
            else:
                print("Файл новостей пуст или не содержит колонку 'publishedAt'.")
        except Exception as e:
            print(f"Ошибка при загрузке или обработке новостей: {e}")
    else:
        print("Путь к файлу новостей не указан или файл не существует. Бэктестинг без новостных признаков.")

    # 4. Выравнивание и объединение данных
    print("Шаг 4: Выравнивание и объединение рыночных и новостных данных...")
    features_df = dp_module.align_and_merge_data(
        market_df_indicators.copy(), 
        news_df_processed.copy() if not news_df_processed.empty else pd.DataFrame(), # Передаем пустой DF если новостей нет
        news_time_window_td=pd.Timedelta(hours=config.get("news_lookback_hours", 4))
    )
    if features_df is None or features_df.empty:
        print("Ошибка: DataFrame пуст после объединения данных.")
        return []
    
    # Убедимся, что индекс features_df это DatetimeIndex
    if not isinstance(features_df.index, pd.DatetimeIndex):
        print("Предупреждение: Индекс features_df не является DatetimeIndex. Попытка преобразования...")
        features_df.index = pd.to_datetime(features_df.index)


    # 5. Подготовка признаков для предсказания
    print("Шаг 5: Подготовка признаков для предсказания...")
    if not hasattr(model, 'feature_names_in_'):
        print("Ошибка: Модель не имеет атрибута 'feature_names_in_'. Невозможно определить признаки для предсказания.")
        # Пытаемся угадать признаки (все числовые, кроме 'open', 'high', 'low', 'close', 'volume', 'target' если есть)
        # Это НЕ НАДЕЖНО. Модель должна предоставлять список признаков.
        excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        model_feature_names = [col for col in features_df.select_dtypes(include=np.number).columns if col not in excluded_cols]
        print(f"Предупреждение: Используются предполагаемые имена признаков: {model_feature_names}")
    else:
        model_feature_names = model.feature_names_in_
    
    # Убедимся, что все ожидаемые моделью признаки присутствуют в features_df
    missing_features = [col for col in model_feature_names if col not in features_df.columns]
    if missing_features:
        print(f"Ошибка: Отсутствуют необходимые признаки в features_df: {missing_features}")
        # Добавляем отсутствующие колонки с NaN, чтобы ffill/bfill могли сработать
        for col in missing_features:
            features_df[col] = np.nan
            
    X_for_prediction = features_df[model_feature_names].copy()
    # Обработка NaN: сначала ffill (заполнение вперед), потом bfill (заполнение назад)
    X_for_prediction.fillna(method='ffill', inplace=True)
    X_for_prediction.fillna(method='bfill', inplace=True)
    
    # Сохраняем индексы до удаления NaN для синхронизации с features_df
    valid_indices_after_nan_fill = X_for_prediction.dropna().index
    X_for_prediction = X_for_prediction.loc[valid_indices_after_nan_fill]
    features_df_synced = features_df.loc[valid_indices_after_nan_fill] # Синхронизируем основной DF

    if X_for_prediction.empty:
        print("Ошибка: Нет данных для предсказания после обработки NaN.")
        return []
    
    # Начальный индекс для итерации (после прогрева индикаторов и обработки NaN)
    # Например, если самый длинный индикатор требует 50 периодов, а NaN обработка удалила еще несколько.
    # Мы можем просто начать с первой доступной строки в X_for_prediction.
    start_index_loc = 0 # Начинаем с первой доступной строки в X_for_prediction

    # 6. Инициализация переменных для бэктестинга
    trades = []
    current_trade = None
    print(f"Шаг 6 & 7: Итерация по данным и симуляция торговли (с индекса {start_index_loc}, всего {len(X_for_prediction)} свечей)...")

    # 7. Итерация по данным и симуляция торговли
    # Итерируем до предпоследней строки, чтобы иметь следующую свечу для SL/TP проверки
    for i in range(start_index_loc, len(X_for_prediction) -1): 
        # Время открытия текущей свечи (для которой мы могли бы войти в сделку)
        # Используем features_df_synced для цен, т.к. X_for_prediction может содержать только признаки
        current_candle_timestamp = X_for_prediction.index[i] 
        
        # Данные текущей свечи, на которой может произойти SL/TP или открытие новой сделки
        current_open_price = features_df_synced.loc[current_candle_timestamp, 'open']
        current_candle_high = features_df_synced.loc[current_candle_timestamp, 'high']
        current_candle_low = features_df_synced.loc[current_candle_timestamp, 'low']
        # close_price = features_df_synced.loc[current_candle_timestamp, 'close'] # Не используется для SL/TP на H/L

        # Проверка существующей сделки
        if current_trade:
            trade_closed_this_candle = False
            # Проверка Take Profit
            if current_trade['type'] in ['BUY', 'LONG']:
                if current_candle_high >= current_trade['take_profit']:
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'TP'
                    trade_closed_this_candle = True
            elif current_trade['type'] in ['SELL', 'SHORT']:
                if current_candle_low <= current_trade['take_profit']:
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'TP'
                    trade_closed_this_candle = True
            
            # Проверка Stop Loss (если TP не был достигнут)
            if not trade_closed_this_candle:
                if current_trade['type'] in ['BUY', 'LONG']:
                    if current_candle_low <= current_trade['stop_loss']:
                        current_trade['exit_price'] = current_trade['stop_loss']
                        current_trade['outcome'] = 'SL'
                        trade_closed_this_candle = True
                elif current_trade['type'] in ['SELL', 'SHORT']:
                    if current_candle_high >= current_trade['stop_loss']:
                        current_trade['exit_price'] = current_trade['stop_loss']
                        current_trade['outcome'] = 'SL'
                        trade_closed_this_candle = True
            
            if trade_closed_this_candle:
                current_trade['exit_time'] = current_candle_timestamp # Выход на текущей свече
                current_trade['status'] = 'CLOSED'
                trades.append(current_trade.copy())
                # print(f"Trade closed: {current_trade}")
                current_trade = None

        # Если нет активной сделки, пытаемся открыть новую
        if current_trade is None:
            # Используем признаки *предыдущей* свечи (i-1) для предсказания открытия на *текущей* свече (i)
            # Убедимся, что i-1 >= 0
            if i == 0 and start_index_loc == 0 : # Не можем использовать i-1 для первой строки
                # print("Пропуск первой свечи, т.к. нет предыдущих признаков для предсказания.")
                continue 
            
            # Индекс для признаков (сдвинутый)
            # Если i=0, то prev_features_idx будет X_for_prediction.index[-1], что неверно.
            # Мы должны брать признаки i-ой свечи, чтобы предсказать для (i+1)-ой.
            # Но в цикле мы на свече 'i', и решение о входе принимаем на ее 'open'.
            # Значит, признаки должны быть с (i-1)-ой свечи.
            
            features_for_pred_idx = X_for_prediction.index[i-1] if i > 0 else X_for_prediction.index[i] # Для i=0, берем текущие (менее реалистично)

            features_for_pred = X_for_prediction.loc[[features_for_pred_idx]] # Двойные скобки для DataFrame
            
            # Получение ATR с предыдущей свечи (той же, что и признаки)
            atr_col_name = 'ATRr_14' # Убедитесь, что это имя колонки ATR
            atr_value = np.nan
            if atr_col_name in features_df_synced.columns: # features_df_synced, т.к. ATR там
                atr_value = features_df_synced.loc[features_for_pred_idx, atr_col_name]
            
            if pd.isna(atr_value) and config['signal_config']['stop_loss_type'] == 'atr':
                # print(f"Предупреждение: ATR is NaN для {features_for_pred_idx}, пропуск генерации сигнала.")
                continue

            prediction, probability = mt_module.make_prediction(model, features_for_pred)

            if prediction is not None and probability is not None:
                pred_class = prediction[0]
                pred_prob = probability[0][pred_class] if len(probability[0]) > pred_class else probability[0][0]

                # Цена открытия текущей свечи (i) используется как цена входа
                signal = sg_module.generate_signal(
                    pred_class, pred_prob, current_open_price, 
                    config['signal_config'], atr_value=atr_value
                )

                if signal:
                    current_trade = signal
                    current_trade['entry_time'] = current_candle_timestamp
                    # current_trade['entry_price'] уже установлен в generate_signal на основе current_open_price
                    current_trade['status'] = 'OPEN'
                    # print(f"Trade opened: {current_trade}")
            # else:
                # print(f"Предсказание не удалось для {current_candle_timestamp}")

    print("Шаг 8: Бэктестинг завершен.")
    return trades


def calculate_backtest_metrics(trades_list):
    """
    Рассчитывает метрики производительности на основе списка сделок.

    :param trades_list: Список словарей, где каждый словарь представляет закрытую сделку.
    :type trades_list: list[dict]
    :return: Словарь с рассчитанными метриками.
    :rtype: dict
    """
    if not trades_list:
        return {
            "total_trades": 0, "profitable_trades": 0, "loss_making_trades": 0,
            "win_rate": 0, "total_pnl_absolute": 0, "average_profit": 0, "average_loss": 0,
            "profit_factor": 0, "max_drawdown": 0 # Max drawdown - сложнее, пока 0
        }

    total_trades = len(trades_list)
    profitable_trades = 0
    loss_making_trades = 0
    total_pnl_absolute = 0
    total_profit = 0
    total_loss = 0

    for trade in trades_list:
        if 'entry_price' not in trade or 'exit_price' not in trade or \
           trade['entry_price'] is None or trade['exit_price'] is None:
            # print(f"Пропуск сделки из-за отсутствия цен: {trade.get('signal_id', 'N/A')}")
            total_trades -=1 # Уменьшаем общее количество, если сделка некорректна
            continue

        pnl = 0
        if trade['type'] in ['BUY', 'LONG']:
            pnl = trade['exit_price'] - trade['entry_price']
        elif trade['type'] in ['SELL', 'SHORT']:
            pnl = trade['entry_price'] - trade['exit_price']
        
        trade['pnl'] = pnl # Добавляем PnL к сделке для возможного анализа
        total_pnl_absolute += pnl

        if pnl > 0:
            profitable_trades += 1
            total_profit += pnl
        elif pnl < 0:
            loss_making_trades += 1
            total_loss += abs(pnl) # Суммируем абсолютные значения убытков

    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    average_profit = (total_profit / profitable_trades) if profitable_trades > 0 else 0
    average_loss = (total_loss / loss_making_trades) if loss_making_trades > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf') # Inf если нет убытков

    metrics = {
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "loss_making_trades": loss_making_trades,
        "win_rate_percent": round(win_rate, 2),
        "total_pnl_absolute": round(total_pnl_absolute, 4), # Округление до 4 знаков
        "average_profit": round(average_profit, 4),
        "average_loss": round(average_loss, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 'inf'
        # "max_drawdown": "Not implemented" # Реализация максимальной просадки требует отслеживания баланса
    }
    return metrics


if __name__ == '__main__':
    print("--- Демонстрация модуля simple_backtester ---")

    # Определяем пути к данным и модели для __main__
    # Данные должны быть созданы main_controller.py или data_preparer.py в их __main__
    dummy_symbol_main = 'DUMMY-COIN-USDT'
    dummy_timeframe_main = '1H'
    
    # Путь к CSV с историческими данными
    hist_data_dir_main = os.path.join(DEFAULT_DATA_DIR_MAIN, 'historical')
    dummy_hist_csv_path = os.path.join(hist_data_dir_main, f"{dummy_symbol_main}_{dummy_timeframe_main}.csv")

    # Путь к CSV с новостями (агрегированный)
    news_data_dir_main = os.path.join(DEFAULT_DATA_DIR_MAIN, 'news')
    dummy_news_csv_path = os.path.join(news_data_dir_main, 'dummy_news_aggregated.csv') # Из data_preparer

    # Путь к модели
    model_filename_main = f"{dummy_symbol_main.lower().replace('-', '_')}_{dummy_timeframe_main.lower()}_rf_classifier.joblib"
    model_path_main = os.path.join(DEFAULT_MODELS_DIR_MAIN, model_filename_main)

    # --- Проверка наличия dummy данных и модели ---
    if not os.path.exists(dummy_hist_csv_path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Файл dummy исторических данных не найден: {dummy_hist_csv_path}")
        print("Пожалуйста, запустите main_controller.py или data_preparer.py для его создания.")
        # Для теста можно создать минимальный файл, но лучше использовать сгенерированный
        # dp_module.create_dummy_market_data(dummy_hist_csv_path, dummy_symbol_main, dummy_timeframe_main)
        # exit()
    
    if not os.path.exists(dummy_news_csv_path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Файл dummy новостей не найден: {dummy_news_csv_path}")
        # dp_module.create_dummy_news_data(dummy_news_csv_path)
        # exit()

    if not os.path.exists(model_path_main):
        print(f"ОШИБКА: Файл модели не найден: {model_path_main}")
        print("Пожалуйста, запустите model_trainer.py для создания dummy модели.")
        exit()

    # --- Конфигурация для бэктестинга ---
    BACKTEST_CONFIG = {
        "symbol": dummy_symbol_main,
        "timeframe": dummy_timeframe_main,
        "historical_data_dir": hist_data_dir_main,
        "news_data_path": dummy_news_csv_path, # Путь к агрегированным новостям
        "signal_config": { # Такая же, как в main_controller
            "confidence_threshold": 0.51, 
            "risk_reward_ratio": 1.5,
            "stop_loss_type": 'atr', 
            "stop_loss_value": 1.5, 
            "signal_type": 'spot',   
            "price_precision": 4     
        },
        "news_lookback_hours": 24 # Как в main_controller
    }

    # Загрузка модели
    print(f"\nЗагрузка модели из: {model_path_main}")
    model_object = mt_module.load_model(model_path_main)

    if model_object:
        print("Модель успешно загружена.")
        # Запуск бэктестинга
        print("\nЗапуск бэктестинга...")
        backtest_trades = run_backtest(BACKTEST_CONFIG, model_object)

        if backtest_trades:
            print(f"\nБэктестинг завершен. Совершено сделок: {len(backtest_trades)}")
            print("Первые 3 сделки (если есть):")
            for trade in backtest_trades[:3]:
                print(trade)
            
            # Расчет и вывод метрик
            print("\nРасчет метрик бэктестинга...")
            metrics = calculate_backtest_metrics(backtest_trades)
            print("\nМетрики бэктестинга:")
            for key, value in metrics.items():
                print(f"  {key.replace('_', ' ').capitalize()}: {value}")
        else:
            print("\nБэктестинг не сгенерировал сделок или произошла ошибка.")
    else:
        print(f"Не удалось загрузить модель для бэктестинга из {model_path_main}.")

    print("\n--- Демонстрация simple_backtester завершена ---")
