import numpy as np # Для np.nan и, возможно, других числовых операций

def round_price(price, precision=6):
    """
    Округляет цену до заданной точности.

    :param price: Цена для округления.
    :type price: float
    :param precision: Количество знаков после запятой.
    :type precision: int
    :return: Округленная цена.
    :rtype: float
    """
    if price is None or np.isnan(price):
        return None
    return round(price, precision)

def generate_signal(prediction, probability, current_price, config, atr_value=None):
    """
    Генерирует торговый сигнал на основе предсказания модели и правил управления рисками.

    :param prediction: Предсказанный класс (например, 1 для UP, 0 для DOWN).
    :type prediction: int
    :param probability: Вероятность, связанная с предсказанием.
    :type probability: float
    :param current_price: Текущая цена для входа.
    :type current_price: float
    :param config: Словарь конфигурации, содержащий:
                   'confidence_threshold' (float),
                   'risk_reward_ratio' (float),
                   'stop_loss_type' (str, 'percentage' или 'atr'),
                   'stop_loss_value' (float),
                   'signal_type' (str, 'spot' или 'futures', по умолчанию 'spot').
    :type config: dict
    :param atr_value: Текущее значение Average True Range (ATR). Обязательно, если stop_loss_type == 'atr'.
    :type atr_value: float, optional
    :return: Словарь с деталями сигнала или None, если сигнал не сгенерирован.
    :rtype: dict or None
    """
    # Константы для предсказаний
    UP_SIGNAL = 1  # Предполагаем, что 1 означает рост цены
    DOWN_SIGNAL = 0 # Предполагаем, что 0 означает падение цены

    # Проверка порога уверенности
    if probability < config.get('confidence_threshold', 0.5): # 0.5 - значение по умолчанию, если не указано
        print(f"Сигнал не сгенерирован: уверенность {probability:.2%} < порога {config.get('confidence_threshold', 0.5):.2%}")
        return None

    signal_output = {}
    signal_type_config = config.get('signal_type', 'spot') # 'spot' по умолчанию

    # Определяем точность округления для финансовых значений, можно вынести в config
    price_precision = config.get('price_precision', 6) # Например, 6 знаков для большинства криптопар

    if prediction == UP_SIGNAL:
        signal_output['type'] = 'LONG' if signal_type_config == 'futures' else 'BUY'
        signal_output['entry_price'] = round_price(current_price, price_precision)
        
        # Расчет Stop-Loss
        if config['stop_loss_type'] == 'percentage':
            sl = current_price * (1 - config['stop_loss_value'])
        elif config['stop_loss_type'] == 'atr':
            if atr_value is None or atr_value <= 0:
                print("Предупреждение: ATR значение не предоставлено или некорректно для stop_loss_type='atr'. Сигнал не сгенерирован.")
                return None
            sl = current_price - (atr_value * config['stop_loss_value'])
        else:
            print(f"Предупреждение: Неверный тип stop_loss_type: {config['stop_loss_type']}. Сигнал не сгенерирован.")
            return None
        
        signal_output['stop_loss'] = round_price(sl, price_precision)
        
        # Расчет Take-Profit
        if signal_output['stop_loss'] is not None: # Убедимся, что SL рассчитан
            risk_amount = current_price - signal_output['stop_loss']
            if risk_amount <= 0: # Если SL выше или равен цене входа для LONG
                print(f"Предупреждение: Некорректный Stop-Loss ({signal_output['stop_loss']}) для LONG сигнала при цене входа {current_price}. Риск не положительный. Сигнал не сгенерирован.")
                return None
            signal_output['take_profit'] = round_price(current_price + risk_amount * config['risk_reward_ratio'], price_precision)
        else:
            signal_output['take_profit'] = None # Если SL не был рассчитан

    elif prediction == DOWN_SIGNAL:
        signal_output['type'] = 'SHORT' if signal_type_config == 'futures' else 'SELL'
        signal_output['entry_price'] = round_price(current_price, price_precision)

        # Расчет Stop-Loss
        if config['stop_loss_type'] == 'percentage':
            sl = current_price * (1 + config['stop_loss_value'])
        elif config['stop_loss_type'] == 'atr':
            if atr_value is None or atr_value <= 0:
                print("Предупреждение: ATR значение не предоставлено или некорректно для stop_loss_type='atr'. Сигнал не сгенерирован.")
                return None
            sl = current_price + (atr_value * config['stop_loss_value'])
        else:
            print(f"Предупреждение: Неверный тип stop_loss_type: {config['stop_loss_type']}. Сигнал не сгенерирован.")
            return None
            
        signal_output['stop_loss'] = round_price(sl, price_precision)

        # Расчет Take-Profit
        if signal_output['stop_loss'] is not None: # Убедимся, что SL рассчитан
            risk_amount = signal_output['stop_loss'] - current_price
            if risk_amount <= 0: # Если SL ниже или равен цене входа для SHORT
                print(f"Предупреждение: Некорректный Stop-Loss ({signal_output['stop_loss']}) для SHORT сигнала при цене входа {current_price}. Риск не положительный. Сигнал не сгенерирован.")
                return None
            signal_output['take_profit'] = round_price(current_price - risk_amount * config['risk_reward_ratio'], price_precision)
        else:
            signal_output['take_profit'] = None # Если SL не был рассчитан
            
    else:
        print(f"Предупреждение: Неверное значение предсказания: {prediction}. Сигнал не сгенерирован.")
        return None

    signal_output['confidence'] = round(probability, 4) # Округляем уверенность
    signal_output['justification'] = f"AI prediction: {'UP' if prediction == UP_SIGNAL else 'DOWN'} with {probability:.2%} confidence."
    
    # Проверка, что все ключевые значения не None
    if signal_output.get('entry_price') is None or \
       signal_output.get('stop_loss') is None or \
       signal_output.get('take_profit') is None:
        print(f"Предупреждение: Одно из ключевых значений (entry, SL, TP) не было рассчитано. Сигнал для {signal_output.get('type')} отменен.")
        return None
        
    return signal_output


if __name__ == '__main__':
    print("--- Демонстрация модуля signal_generator ---")

    # Примерные конфигурации
    config_percentage_spot = {
        'confidence_threshold': 0.60,
        'risk_reward_ratio': 1.5,
        'stop_loss_type': 'percentage',
        'stop_loss_value': 0.02, # 2%
        'signal_type': 'spot',
        'price_precision': 2 # для примера с ценами типа 100.00
    }

    config_atr_spot = {
        'confidence_threshold': 0.70,
        'risk_reward_ratio': 2.0,
        'stop_loss_type': 'atr',
        'stop_loss_value': 1.5, # 1.5 * ATR
        'signal_type': 'spot',
        'price_precision': 4 
    }
    
    config_atr_futures = {
        'confidence_threshold': 0.65,
        'risk_reward_ratio': 1.8,
        'stop_loss_type': 'atr',
        'stop_loss_value': 2.0, # 2 * ATR
        'signal_type': 'futures',
        'price_precision': 6
    }

    current_market_price = 25000.0
    sample_atr = 300.0

    print("\n--- Тестовые сценарии ---")

    # 1. BUY сигнал (spot) с процентным SL
    print("\n1. Тест: BUY сигнал (spot) с процентным SL")
    prediction_buy = 1
    prob_buy_high_conf = 0.75
    signal1 = generate_signal(prediction_buy, prob_buy_high_conf, current_market_price, config_percentage_spot)
    print(f"   Сгенерированный сигнал 1: {signal1}")

    # 2. SELL сигнал (spot) с ATR SL
    print("\n2. Тест: SELL сигнал (spot) с ATR SL")
    prediction_sell = 0
    prob_sell_high_conf = 0.80
    signal2 = generate_signal(prediction_sell, prob_sell_high_conf, current_market_price, config_atr_spot, atr_value=sample_atr)
    print(f"   Сгенерированный сигнал 2: {signal2}")

    # 3. Сигнал не генерируется из-за низкой уверенности
    print("\n3. Тест: Сигнал не генерируется (низкая уверенность)")
    prob_low_conf = 0.45
    signal3 = generate_signal(prediction_buy, prob_low_conf, current_market_price, config_percentage_spot)
    print(f"   Сгенерированный сигнал 3: {signal3}")

    # 4. Сигнал не генерируется из-за отсутствия ATR при stop_loss_type='atr'
    print("\n4. Тест: Сигнал не генерируется (отсутствует ATR)")
    signal4 = generate_signal(prediction_sell, prob_sell_high_conf, current_market_price, config_atr_spot, atr_value=None)
    print(f"   Сгенерированный сигнал 4: {signal4}")
    
    # 5. Сигнал не генерируется из-за некорректного (нулевого) ATR
    print("\n5. Тест: Сигнал не генерируется (нулевой ATR)")
    signal5 = generate_signal(prediction_sell, prob_sell_high_conf, current_market_price, config_atr_spot, atr_value=0)
    print(f"   Сгенерированный сигнал 5: {signal5}")

    # 6. LONG сигнал (futures) с ATR SL
    print("\n6. Тест: LONG сигнал (futures) с ATR SL")
    current_futures_price = 0.876543
    sample_futures_atr = 0.015000
    signal6 = generate_signal(prediction_buy, prob_buy_high_conf, current_futures_price, config_atr_futures, atr_value=sample_futures_atr)
    print(f"   Сгенерированный сигнал 6: {signal6}")
    
    # 7. SHORT сигнал (futures) с ATR SL
    print("\n7. Тест: SHORT сигнал (futures) с ATR SL (цена ниже, чем SL из-за большого ATR*multiplier)")
    # Демонстрация, когда SL может быть "нелогичным" если риск слишком мал относительно ATR
    config_atr_futures_high_sl = config_atr_futures.copy()
    # config_atr_futures_high_sl['stop_loss_value'] = 0.01 # Очень маленький множитель ATR, чтобы SL был близко
    # Этот тест на самом деле проверяет, что (sl - current_price) > 0 для SHORT
    # Для SHORT: sl = current_price + (atr_value * config['stop_loss_value'])
    # take_profit = current_price - (sl - current_price) * config['risk_reward_ratio']
    # Если (sl - current_price) <= 0, то risk_amount <=0, сигнал не генерируется.
    # Это уже покрыто проверкой risk_amount > 0.
    
    signal7 = generate_signal(prediction_sell, prob_sell_high_conf, current_futures_price, config_atr_futures, atr_value=sample_futures_atr)
    print(f"   Сгенерированный сигнал 7: {signal7}")

    # 8. Неверный тип stop_loss
    print("\n8. Тест: Неверный тип stop_loss_type")
    config_invalid_sl = config_percentage_spot.copy()
    config_invalid_sl['stop_loss_type'] = 'unknown_type'
    signal8 = generate_signal(prediction_buy, prob_buy_high_conf, current_market_price, config_invalid_sl)
    print(f"   Сгенерированный сигнал 8: {signal8}")

    # 9. Неверное значение предсказания
    print("\n9. Тест: Неверное значение предсказания")
    signal9 = generate_signal(3, prob_buy_high_conf, current_market_price, config_percentage_spot)
    print(f"   Сгенерированный сигнал 9: {signal9}")

    # 10. SL выше цены входа для LONG сигнала (из-за слишком большого stop_loss_value для percentage)
    print("\n10. Тест: SL выше цены входа для LONG (процентный SL)")
    config_bad_sl_long = {**config_percentage_spot, 'stop_loss_value': 1.1} # SL = P * (1 - 1.1) = P * -0.1
    signal10 = generate_signal(prediction_buy, prob_buy_high_conf, current_market_price, config_bad_sl_long)
    print(f"   Сгенерированный сигнал 10: {signal10}") # Ожидается None из-за проверки risk_amount > 0

    # 11. SL ниже цены входа для SHORT сигнала (из-за слишком большого stop_loss_value для percentage)
    print("\n11. Тест: SL ниже цены входа для SHORT (процентный SL)")
    config_bad_sl_short = {**config_percentage_spot, 'stop_loss_value': -0.1} # SL = P * (1 + (-0.1)) = P * 0.9
    signal11 = generate_signal(prediction_sell, prob_sell_high_conf, current_market_price, config_bad_sl_short)
    print(f"   Сгенерированный сигнал 11: {signal11}") # Ожидается None из-за проверки risk_amount > 0


    print("\n--- Демонстрация signal_generator завершена ---")
