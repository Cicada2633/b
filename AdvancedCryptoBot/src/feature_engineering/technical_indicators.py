import pandas as pd
import pandas_ta as ta # Библиотека для технических индикаторов

def add_sma(df, length=20):
    """
    Добавляет простую скользящую среднюю (SMA) в DataFrame.

    :param df: pandas DataFrame с OHLCV данными. Ожидается колонка 'close'.
    :type df: pd.DataFrame
    :param length: Длина периода для SMA.
    :type length: int
    :return: pandas DataFrame с добавленной колонкой SMA.
    :rtype: pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame.")
    if 'close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close'.")
    if df['close'].isnull().all(): # Если все значения NaN, SMA тоже будет NaN
        df[f'SMA_{length}'] = float('nan')
        return df
        
    try:
        # Расчет SMA с использованием rolling mean
        df[f'SMA_{length}'] = df['close'].rolling(window=length, min_periods=1).mean() # min_periods=1 для начальных значений
    except Exception as e:
        print(f"Ошибка при расчете SMA_{length}: {e}")
        df[f'SMA_{length}'] = float('nan') # В случае ошибки заполняем NaN
    return df

def add_ema(df, length=20):
    """
    Добавляет экспоненциальную скользящую среднюю (EMA) в DataFrame.

    :param df: pandas DataFrame с OHLCV данными. Ожидается колонка 'close'.
    :type df: pd.DataFrame
    :param length: Длина периода для EMA.
    :type length: int
    :return: pandas DataFrame с добавленной колонкой EMA.
    :rtype: pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame.")
    if 'close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close'.")
    if df['close'].isnull().all():
        df[f'EMA_{length}'] = float('nan')
        return df

    try:
        # Расчет EMA с использованием ewm
        df[f'EMA_{length}'] = df['close'].ewm(span=length, adjust=False, min_periods=1).mean() # min_periods=1 для начальных значений
    except Exception as e:
        print(f"Ошибка при расчете EMA_{length}: {e}")
        df[f'EMA_{length}'] = float('nan')
    return df

def add_rsi(df, length=14):
    """
    Добавляет индекс относительной силы (RSI) в DataFrame.

    :param df: pandas DataFrame с OHLCV данными. Ожидается колонка 'close'.
    :type df: pd.DataFrame
    :param length: Длина периода для RSI.
    :type length: int
    :return: pandas DataFrame с добавленной колонкой RSI.
    :rtype: pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame.")
    if 'close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close'.")
    if df['close'].isnull().all() or len(df) < length : # RSI требует достаточно данных
        df[f'RSI_{length}'] = float('nan')
        return df
        
    try:
        # Расчет RSI с использованием pandas_ta
        rsi_series = ta.rsi(df['close'], length=length)
        if rsi_series is not None:
            df[f'RSI_{length}'] = rsi_series
        else:
            df[f'RSI_{length}'] = float('nan')
    except Exception as e:
        print(f"Ошибка при расчете RSI_{length}: {e}")
        df[f'RSI_{length}'] = float('nan')
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """
    Добавляет схождение/расхождение скользящих средних (MACD) в DataFrame.

    :param df: pandas DataFrame с OHLCV данными. Ожидается колонка 'close'.
    :type df: pd.DataFrame
    :param fast: Период быстрой EMA.
    :type fast: int
    :param slow: Период медленной EMA.
    :type slow: int
    :param signal: Период сигнальной линии EMA.
    :type signal: int
    :return: pandas DataFrame с добавленными колонками MACD (линия, гистограмма, сигнал).
    :rtype: pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame.")
    if 'close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close'.")
    if df['close'].isnull().all() or len(df) < slow: # MACD требует достаточно данных, особенно для медленной EMA
        df[f'MACD_{fast}_{slow}_{signal}'] = float('nan')
        df[f'MACDh_{fast}_{slow}_{signal}'] = float('nan')
        df[f'MACDs_{fast}_{slow}_{signal}'] = float('nan')
        return df

    try:
        # Расчет MACD с использованием pandas_ta
        # Возвращает DataFrame с колонками: MACD_<fast>_<slow>_<signal>, MACDh_<fast>_<slow>_<signal>, MACDs_<fast>_<slow>_<signal>
        macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        
        if macd_df is not None and not macd_df.empty:
            # pandas_ta может возвращать имена колонок в разных форматах в зависимости от версии
            # Стандартные имена: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # Проверим наличие и переименуем, если нужно, или используем как есть
            
            # Имена колонок, которые pandas-ta обычно создает
            macd_line_col = f'MACD_{fast}_{slow}_{signal}'
            macd_hist_col = f'MACDh_{fast}_{slow}_{signal}' # Гистограмма
            macd_signal_col = f'MACDs_{fast}_{slow}_{signal}' # Сигнальная линия

            if macd_line_col in macd_df.columns:
                df[macd_line_col] = macd_df[macd_line_col]
            else: # Обработка случая, если имя колонки другое (например, без суффикса)
                # Это маловероятно для стандартных вызовов, но для надежности
                if 'MACD' in macd_df.columns: # Общее имя
                     df[macd_line_col] = macd_df['MACD'] 
                else: # Если совсем не найдено
                     df[macd_line_col] = float('nan')
            
            if macd_hist_col in macd_df.columns:
                df[macd_hist_col] = macd_df[macd_hist_col]
            elif 'MACDh' in macd_df.columns:
                 df[macd_hist_col] = macd_df['MACDh']
            else:
                 df[macd_hist_col] = float('nan')

            if macd_signal_col in macd_df.columns:
                df[macd_signal_col] = macd_df[macd_signal_col]
            elif 'MACDs' in macd_df.columns:
                 df[macd_signal_col] = macd_df['MACDs']
            else:
                 df[macd_signal_col] = float('nan')
        else:
            df[f'MACD_{fast}_{slow}_{signal}'] = float('nan')
            df[f'MACDh_{fast}_{slow}_{signal}'] = float('nan')
            df[f'MACDs_{fast}_{slow}_{signal}'] = float('nan')
            
    except Exception as e:
        print(f"Ошибка при расчете MACD ({fast},{slow},{signal}): {e}")
        df[f'MACD_{fast}_{slow}_{signal}'] = float('nan')
        df[f'MACDh_{fast}_{slow}_{signal}'] = float('nan')
        df[f'MACDs_{fast}_{slow}_{signal}'] = float('nan')
    return df

def add_all_indicators(df):
    """
    Добавляет набор технических индикаторов в DataFrame.

    :param df: pandas DataFrame с OHLCV данными.
    :type df: pd.DataFrame
    :return: pandas DataFrame с добавленными индикаторами.
    :rtype: pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Предупреждение: Входной DataFrame пуст или не является DataFrame. Индикаторы не будут добавлены.")
        return df
    if 'close' not in df.columns:
        print("Предупреждение: Колонка 'close' отсутствует. Многие индикаторы не могут быть рассчитаны.")
        # Можно вернуть df или возбудить ошибку, в зависимости от желаемого поведения
        # return df 

    df_with_indicators = df.copy() # Работаем с копией, чтобы не изменять оригинальный DataFrame напрямую

    # Добавление SMA
    df_with_indicators = add_sma(df_with_indicators, length=20)
    df_with_indicators = add_sma(df_with_indicators, length=50)
    
    # Добавление EMA
    df_with_indicators = add_ema(df_with_indicators, length=20)
    df_with_indicators = add_ema(df_with_indicators, length=50)
    
    # Добавление RSI
    df_with_indicators = add_rsi(df_with_indicators, length=14)
    
    # Добавление MACD
    df_with_indicators = add_macd(df_with_indicators, fast=12, slow=26, signal=9)
    
    # Можно добавить другие индикаторы из pandas_ta, например:
    # ATR (Average True Range)
    try:
        if all(col in df_with_indicators.columns for col in ['high', 'low', 'close']):
            atr_series = ta.atr(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], length=14)
            if atr_series is not None:
                 df_with_indicators['ATR_14'] = atr_series
            else:
                 df_with_indicators['ATR_14'] = float('nan')
        else:
            print("Предупреждение: Колонки 'high', 'low', 'close' необходимы для ATR. Индикатор не будет добавлен.")
            df_with_indicators['ATR_14'] = float('nan')
    except Exception as e:
        print(f"Ошибка при расчете ATR_14: {e}")
        df_with_indicators['ATR_14'] = float('nan')

    # Bollinger Bands (пример, pandas_ta возвращает несколько колонок)
    try:
        if 'close' in df_with_indicators.columns:
            bbands = ta.bbands(df_with_indicators['close'], length=20, std=2) # length=период SMA, std=кол-во станд.отклонений
            if bbands is not None and not bbands.empty:
                # bbands возвращает: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
                # (Lower, Middle, Upper, Bandwidth, Percent)
                df_with_indicators = pd.concat([df_with_indicators, bbands], axis=1)
        else:
            print("Предупреждение: Колонка 'close' необходима для Bollinger Bands. Индикатор не будет добавлен.")
    except Exception as e:
        print(f"Ошибка при расчете Bollinger Bands: {e}")
        # Если нужно добавить пустые колонки в случае ошибки
        for col_suffix in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']:
             df_with_indicators[col_suffix] = float('nan')
             
    return df_with_indicators


if __name__ == '__main__':
    print("--- Демонстрация модуля technical_indicators ---")
    
    # Создание примерного DataFrame (имитация OHLCV данных)
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                     '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                     '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                     '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
                                     '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
                                     '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-29', '2023-01-30']),
        'open': [10, 11, 12, 13, 14, 15, 14, 16, 17, 18, 19, 20, 21, 20, 19, 18, 19, 20, 22, 23, 24, 25, 23, 22, 20, 21, 22, 23, 24, 25],
        'high': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 22, 21, 20, 21, 22, 23, 24, 25, 26, 25, 24, 23, 22, 23, 24, 25, 26],
        'low':  [9, 10, 11, 12, 13, 14, 13, 15, 16, 17, 18, 19, 20, 19, 18, 17, 18, 19, 21, 22, 23, 24, 22, 21, 19, 20, 21, 22, 23, 24],
        'close':[10, 11.5, 12.2, 13.8, 14.2, 15.5, 13.9, 16.1, 17.5, 18.3, 19.9, 20.1, 21.5, 20.3, 19.2, 18.5, 19.3, 20.7, 22.8, 23.1, 24.5, 25.3, 23.5, 22.1, 20.9, 21.2, 22.8, 23.5, 24.2, 25.8],
        'volume':[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
    }
    sample_ohlcv_df = pd.DataFrame(data)
    sample_ohlcv_df.set_index('timestamp', inplace=True) # Установка timestamp как индекса, если это типично для ваших данных

    print("Исходный DataFrame:")
    print(sample_ohlcv_df.head())

    # Демонстрация одной функции, например, add_rsi
    # df_with_rsi = add_rsi(sample_ohlcv_df.copy(), length=14) # Используем copy(), чтобы не изменять sample_ohlcv_df
    # print("\nDataFrame с RSI(14):")
    # print(df_with_rsi.tail()) # tail, чтобы увидеть значения RSI, где они уже рассчитаны

    # Демонстрация add_all_indicators
    print("\nПрименение add_all_indicators():")
    df_with_all_indicators = add_all_indicators(sample_ohlcv_df.copy()) # Используем copy()
    
    print("\nDataFrame со всеми добавленными индикаторами (последние 5 строк):")
    print(df_with_all_indicators.tail())
    
    print("\nКолонки в DataFrame после добавления индикаторов:")
    print(df_with_all_indicators.columns.tolist())

    # Проверка на NaN значения в конце DataFrame для индикаторов
    print("\nПроверка NaN в последних строках для некоторых индикаторов:")
    last_row_indicators = df_with_all_indicators.iloc[-1][['SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'ATR_14', 'BBM_20_2.0']]
    print(last_row_indicators)
    if last_row_indicators.isnull().any():
        print("Одно или несколько значений индикаторов в последней строке являются NaN (это может быть ожидаемо для сложных индикаторов или коротких данных).")
    else:
        print("Все проверенные индикаторы в последней строке имеют значения.")

    print("\n--- Демонстрация technical_indicators завершена ---")
