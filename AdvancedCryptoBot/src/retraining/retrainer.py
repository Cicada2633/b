import pandas as pd
import json
import os
from datetime import datetime # For new model filename timestamp

# --- Project Module Imports ---
try:
    from AdvancedCryptoBot.src.ml_model.model_trainer import train_model, save_model # Removed load_model as not strictly needed for simplified retraining
except ImportError:
    print("Attempting fallback imports for retrainer.py...")
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    src_path = os.path.join(project_root, 'src')

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from ml_model.model_trainer import train_model, save_model

# --- File Paths ---
# retrainer.py is in AdvancedCryptoBot/src/retraining/
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_LOG_FILE = os.path.join(_BASE_DIR, '..', '..', 'data', 'signals_log.csv')
DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, '..', '..', 'models')


def load_and_prepare_feedback_data(log_file_path=SIGNALS_LOG_FILE):
    """
    Загружает данные из лога сигналов, фильтрует их и подготавливает для переобучения.

    :param log_file_path: Путь к CSV файлу лога сигналов.
    :type log_file_path: str
    :return: Кортеж (X_feedback, y_feedback) или (None, None), если нет валидных данных.
             X_feedback: pandas DataFrame с признаками.
             y_feedback: pandas Series с целевой переменной.
    :rtype: tuple (pd.DataFrame, pd.Series) or (None, None)
    """
    print(f"Загрузка данных обратной связи из: {log_file_path}")
    if not os.path.exists(log_file_path):
        print(f"Ошибка: Файл лога сигналов не найден: {log_file_path}")
        return None, None

    try:
        log_df = pd.read_csv(log_file_path)
    except pd.errors.EmptyDataError:
        print(f"Предупреждение: Файл лога сигналов пуст: {log_file_path}")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении файла лога сигналов {log_file_path}: {e}")
        return None, None

    if log_df.empty:
        print("Файл лога сигналов не содержит данных.")
        return None, None

    # Фильтрация по наличию actual_outcome
    # Ожидаемые значения для actual_outcome: 'TP_HIT_UP', 'SL_HIT_UP', 'TP_HIT_DOWN', 'SL_HIT_DOWN'
    valid_outcomes = ['TP_HIT_UP', 'SL_HIT_UP', 'TP_HIT_DOWN', 'SL_HIT_DOWN']
    feedback_df = log_df[log_df['actual_outcome'].isin(valid_outcomes)].copy() # Используем .copy() для избежания SettingWithCopyWarning

    if feedback_df.empty:
        print("Не найдено записей с валидными 'actual_outcome' для переобучения.")
        return None, None

    print(f"Найдено {len(feedback_df)} записей с валидной обратной связью.")

    # Извлечение признаков из model_features_json
    features_list = []
    for json_str in feedback_df['model_features_json']:
        try:
            features_list.append(json.loads(json_str))
        except json.JSONDecodeError as e:
            print(f"Ошибка декодирования JSON: {e} для строки: {json_str}")
            features_list.append({}) # Добавляем пустой словарь, чтобы сохранить длину
            
    X_feedback = pd.DataFrame(features_list, index=feedback_df.index)
    
    # Убедимся, что все колонки в X_feedback числовые, если нет - пробуем конвертировать или удалить
    # Это важно, т.к. json.loads может вернуть строки для чисел
    for col in X_feedback.columns:
        X_feedback[col] = pd.to_numeric(X_feedback[col], errors='coerce')
    
    # Удаляем строки, где после to_numeric появились NaN в признаках (если были неконвертируемые значения)
    X_feedback.dropna(inplace=True) # Удаляем строки с NaN в признаках
    
    # Выравниваем feedback_df (и, следовательно, y_feedback) с X_feedback
    feedback_df = feedback_df.loc[X_feedback.index]


    if X_feedback.empty:
        print("После обработки признаков DataFrame X_feedback оказался пустым.")
        return None, None

    # Создание целевой переменной y_feedback
    y_list = []
    for index, row in feedback_df.iterrows():
        predicted_class = row['predicted_class']
        actual_outcome = row['actual_outcome']
        
        # Проверяем, что predicted_class - это число (0 или 1)
        try:
            predicted_class = int(float(predicted_class)) # float() для случая "1.0"
        except (ValueError, TypeError):
            print(f"Предупреждение: Некорректное значение predicted_class '{predicted_class}' для signal_id {row.get('signal_id')}. Пропуск.")
            y_list.append(pd.NA) # Используем pd.NA для последующего dropna
            continue

        target_value = pd.NA # Инициализируем как NA
        if predicted_class == 1: # Модель предсказала UP
            if actual_outcome == 'TP_HIT_UP': target_value = 1 # Верно
            elif actual_outcome == 'SL_HIT_UP': target_value = 0 # Неверно
        elif predicted_class == 0: # Модель предсказала DOWN
            if actual_outcome == 'TP_HIT_DOWN': target_value = 0 # Верно
            elif actual_outcome == 'SL_HIT_DOWN': target_value = 1 # Неверно
        
        y_list.append(target_value)

    feedback_df['y_feedback'] = y_list
    feedback_df.dropna(subset=['y_feedback'], inplace=True) # Удаляем строки, где y не был определен
    
    # Обновляем X_feedback, чтобы он соответствовал отфильтрованному feedback_df
    X_feedback = X_feedback.loc[feedback_df.index]
    y_feedback = feedback_df['y_feedback'].astype(int) # Убедимся, что тип int для классификатора

    if X_feedback.empty or y_feedback.empty:
        print("Недостаточно данных для формирования X_feedback или y_feedback после обработки.")
        return None, None

    print(f"Подготовлено {len(X_feedback)} образцов для переобучения.")
    return X_feedback, y_feedback


def retrain_model_with_feedback(X_feedback, y_feedback, original_model_path, 
                                new_model_path_template=None, model_params=None):
    """
    Переобучает модель, используя только данные обратной связи.

    :param X_feedback: pandas DataFrame с признаками из обратной связи.
    :type X_feedback: pd.DataFrame
    :param y_feedback: pandas Series с целевой переменной из обратной связи.
    :type y_feedback: pd.Series
    :param original_model_path: Путь к существующей модели (для определения типа или параметров по умолчанию).
    :type original_model_path: str
    :param new_model_path_template: Шаблон для пути сохранения новой модели. 
                                    Может содержать '{timestamp}'. Если None, перезаписывает original_model_path.
    :type new_model_path_template: str or None
    :param model_params: Параметры для обучения новой модели. Если None, используются параметры по умолчанию.
    :type model_params: dict or None
    :return: Путь к переобученной модели, если успешно, иначе None.
    :rtype: str or None
    """
    print(f"Начало переобучения модели с использованием {len(X_feedback)} образцов обратной связи.")

    if X_feedback is None or y_feedback is None or X_feedback.empty or y_feedback.empty:
        print("Ошибка: Данные для переобучения (X_feedback или y_feedback) отсутствуют или пусты.")
        return None

    # В этой упрощенной версии мы не загружаем старую модель для "дообучения",
    # а обучаем новую модель RandomForestClassifier только на данных обратной связи.
    # train_model сама разделит X_feedback, y_feedback на train/test для оценки.
    
    # Параметры для RandomForestClassifier по умолчанию, если не переданы
    # В реальном сценарии, эти параметры могли бы быть извлечены из старой модели или конфига
    if model_params is None:
        model_params = {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}

    print(f"Обучение новой модели на данных обратной связи с параметрами: {model_params}")
    retrained_model = train_model(X_feedback, y_feedback, model_type='random_forest_classifier', model_params=model_params)

    if retrained_model:
        print("Модель успешно переобучена на данных обратной связи.")
        
        # Определение пути для сохранения
        save_dir = os.path.dirname(original_model_path)
        original_filename = os.path.basename(original_model_path)
        
        if new_model_path_template:
            # Пример: "retrained_model_v{timestamp}.joblib"
            # Или если new_model_path_template это полный путь с {timestamp}
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            if "{timestamp}" in new_model_path_template:
                final_save_path = new_model_path_template.format(timestamp=timestamp_str)
                # Убедимся, что директория существует, если шаблон включает путь
                os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            else: # Если это просто шаблон имени файла
                final_save_path = os.path.join(save_dir, new_model_path_template.format(timestamp=timestamp_str))
        else:
            # Перезаписываем оригинальную модель, добавив суффикс _retrained
            name_part, ext_part = os.path.splitext(original_filename)
            retrained_filename = f"{name_part}_retrained_on_feedback{ext_part}"
            final_save_path = os.path.join(save_dir, retrained_filename)
            print(f"Переобученная модель будет сохранена как: {final_save_path} (оригинал не перезаписан)")

        # save_model ожидает base_dir и filename, если мы не передаем полный путь
        # Если final_save_path это уже полный путь:
        save_model_dir = os.path.dirname(final_save_path)
        save_model_filename = os.path.basename(final_save_path)
        
        save_model(retrained_model, base_dir=save_model_dir, filename=save_model_filename)
        return final_save_path
    else:
        print("Ошибка: Не удалось переобучить модель на данных обратной связи.")
        return None

if __name__ == '__main__':
    print("--- Демонстрация модуля retrainer ---")

    # Предполагается, что signals_log.csv существует и содержит данные.
    # feedback_manager.py в своем __main__ создает такой файл с примерами.
    # Для успешной демонстрации retrainer, убедитесь, что в signals_log.csv
    # есть хотя бы несколько строк с заполненными 'actual_outcome' 
    # (например, 'TP_HIT_UP', 'SL_HIT_UP', 'TP_HIT_DOWN', 'SL_HIT_DOWN')
    # и 'predicted_class' (0 или 1).

    # Пример, как можно добавить больше фидбэка в feedback_manager.py __main__ (если нужно):
    # from AdvancedCryptoBot.src.feedback_system import feedback_manager as fm
    # fm.add_manual_feedback("SIG20231026123000_ADAUSDT", "SL_HIT_UP", "Цена развернулась после сигнала.")
    # fm.add_manual_feedback(
    #    fm.SIGNALS_LOG_FILE.split('/')[-1].replace('.csv','').split('_')[1], # Это неверно, нужен ID сигнала
    #    "TP_HIT_DOWN", 
    #    "Отличный шорт!"
    # ) # Пример, как можно было бы вызвать, но ID сигнала нужен корректный

    print(f"Используется файл лога: {SIGNALS_LOG_FILE}")
    print(f"Модели будут сохраняться в/загружаться из: {DEFAULT_MODEL_DIR}")

    X_feedback_data, y_feedback_data = load_and_prepare_feedback_data(SIGNALS_LOG_FILE)

    if X_feedback_data is not None and not X_feedback_data.empty:
        print(f"\nЗагружено {len(X_feedback_data)} образцов для переобучения.")
        print("Примеры признаков (X_feedback) для переобучения (первые 3 строки):")
        print(X_feedback_data.head(3))
        print("\nПримеры целевых переменных (y_feedback) для переобучения (первые 3 строки):")
        print(y_feedback_data.head(3))

        # Укажем путь к "оригинальной" модели (может быть dummy модель)
        # Имя файла должно соответствовать тому, что используется в model_trainer.py для dummy модели
        # Например, 'dummy_coin_usdt_1h_rf_classifier.joblib'
        dummy_model_name = 'dummy_coin_usdt_1h_rf_classifier.joblib'
        original_model_file_path = os.path.join(DEFAULT_MODEL_DIR, dummy_model_name)

        if not os.path.exists(original_model_file_path):
            print(f"\nПредупреждение: Оригинальный файл модели '{original_model_file_path}' не найден.")
            print("Демонстрация переобучения будет использовать этот путь как основу для сохранения,")
            print("но не сможет использовать его для загрузки параметров старой модели (в текущей упрощенной версии это не требуется).")
            # Можно создать пустую директорию, если ее нет, для save_model
            os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)


        print(f"\nПереобучение модели на основе данных обратной связи. Оригинальный путь модели: {original_model_file_path}")
        
        # Пример использования new_model_path_template
        # new_path_template = os.path.join(DEFAULT_MODEL_DIR, "retrained_feedback_model_v{timestamp}.joblib")
        
        # Для простоты перезапишем "оригинальную" (или создадим новую с суффиксом _retrained_on_feedback)
        retrained_model_path = retrain_model_with_feedback(
            X_feedback_data, 
            y_feedback_data, 
            original_model_file_path,
            new_model_path_template=None # Перезапишет original_model_path с суффиксом
        )

        if retrained_model_path:
            print(f"\nМодель успешно переобучена и сохранена в: {retrained_model_path}")
        else:
            print("\nПереобучение модели не удалось.")
    else:
        print("\nНе найдено достаточных данных обратной связи для переобучения модели.")
        print("Убедитесь, что в файле 'signals_log.csv' есть записи с 'actual_outcome',")
        print("такими как 'TP_HIT_UP', 'SL_HIT_UP', 'TP_HIT_DOWN', 'SL_HIT_DOWN',")
        print("и корректными 'predicted_class'.")

    print("\n--- Демонстрация retrainer завершена ---")
