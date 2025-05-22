import telegram # Основная библиотека
from telegram.ext import Application, CommandHandler, ContextTypes # Для обработчиков команд
import re # Для Markdown экранирования
import asyncio # Для асинхронных операций

# --- Константы ---
TELEGRAM_BOT_TOKEN = "7606239476:AAHpx9jQ5r0UZqIshPWeuqagwAM5yyVC_1s" # Предоставленный токен
ADMIN_CHAT_ID = "799734103" # ID чата администратора, используется как строка

def escape_markdown_v2(text):
    """
    Экранирует специальные символы для Telegram MarkdownV2.

    Символы для экранирования: _ * [ ] ( ) ~ ` > # + - = | { } . !
    Каждый из этих символов должен быть предварен обратным слэшем '\'.

    :param text: Исходная строка.
    :type text: str
    :return: Экранированная строка.
    :rtype: str
    """
    if not isinstance(text, str): # Если на вход пришло не строка (например, None или число)
        return '' # Возвращаем пустую строку или можно вызвать str(text)
        
    # Список символов, которые нужно экранировать в MarkdownV2
    # Обратный слэш \ также является специальным и должен быть экранирован, но re.escape это сделает
    escape_chars = r'_*[]()~`>#+\-=|{}.!'
    # Экранируем каждый символ из списка, добавляя перед ним обратный слэш
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def format_signal_for_telegram(signal_data, coin_symbol):
    """
    Форматирует данные сигнала в сообщение MarkdownV2 для Telegram.

    :param signal_data: Словарь с данными сигнала.
                        Пример: {'type': 'BUY', 'entry_price': 25000.00, 
                                 'stop_loss': 24500.00, 'take_profit': 26500.00,
                                 'confidence': 0.75, 'justification': 'AI prediction based on market trend.'}
    :type signal_data: dict
    :param coin_symbol: Символ криптовалюты (например, 'BTC-USDT').
    :type coin_symbol: str
    :return: Отформатированное сообщение MarkdownV2.
    :rtype: str
    """
    if not signal_data:
        return "Ошибка: Данные сигнала отсутствуют."

    # Экранируем все строковые значения перед вставкой
    esc_coin_symbol = escape_markdown_v2(coin_symbol.replace('-', '')) # #BTCUSDT - без тире для хештега
    signal_type_text = escape_markdown_v2(str(signal_data.get('type', 'N/A')))
    
    # Форматирование цен с нужной точностью и экранирование
    # Точность можно вынести в signal_data или определить глобально
    price_precision = signal_data.get('price_precision', 2 if 'USD' in coin_symbol else 6) 
    
    entry_price_val = signal_data.get('entry_price')
    entry_price_str = f"{entry_price_val:.{price_precision}f}" if isinstance(entry_price_val, (int, float)) else 'N/A'
    esc_entry_price = escape_markdown_v2(entry_price_str)

    stop_loss_val = signal_data.get('stop_loss')
    stop_loss_str = f"{stop_loss_val:.{price_precision}f}" if isinstance(stop_loss_val, (int, float)) else 'N/A'
    esc_stop_loss = escape_markdown_v2(stop_loss_str)

    take_profit_val = signal_data.get('take_profit')
    take_profit_str = f"{take_profit_val:.{price_precision}f}" if isinstance(take_profit_val, (int, float)) else 'N/A'
    esc_take_profit = escape_markdown_v2(take_profit_str)

    confidence_val = signal_data.get('confidence', 0.0)
    confidence_percentage = f"{confidence_val * 100:.2f}%" # Например, "75.00%"
    esc_confidence = escape_markdown_v2(confidence_percentage)
    
    justification_text = escape_markdown_v2(str(signal_data.get('justification', 'No specific justification provided.')))

    # Собираем сообщение с использованием MarkdownV2
    # Используем f-string для удобства, но все динамические части уже экранированы
    message = f"""
*New Signal for #{esc_coin_symbol}* 🪙
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
Type:         *{signal_type_text}*
Entry Price:  `{esc_entry_price}` USD
Stop\-Loss:    `{esc_stop_loss}` USD
Take\-Profit:  `{esc_take_profit}` USD
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
Confidence:   *{esc_confidence}*
Justification: _{justification_text}_
"""
    # Убираем лишние отступы в начале строк, если они есть из-за многострочного f-string
    return '\n'.join([line.lstrip() for line in message.strip().split('\n')])


async def send_telegram_message(bot_token, chat_id, text, parse_mode='MarkdownV2'):
    """
    Асинхронно отправляет сообщение в Telegram.

    :param bot_token: Токен Telegram бота.
    :type bot_token: str
    :param chat_id: ID чата для отправки.
    :type chat_id: str or int
    :param text: Текст сообщения.
    :type text: str
    :param parse_mode: Режим парсинга сообщения ('MarkdownV2' или 'HTML').
    :type parse_mode: str
    """
    bot = telegram.Bot(token=bot_token)
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        print(f"Сообщение успешно отправлено в чат {chat_id}.")
    except telegram.error.TelegramError as e:
        print(f"Ошибка при отправке сообщения в чат {chat_id}: {e}")
        # Дополнительная информация об ошибке, если доступна
        if hasattr(e, 'message'):
            print(f"Сообщение API: {e.message}")
        # Можно также логировать сам текст сообщения, который не удалось отправить (но осторожно с личными данными)
        # print(f"Текст сообщения, вызвавшего ошибку (первые 100 символов): {text[:100]}")
    except Exception as e:
        print(f"Непредвиденная ошибка при отправке сообщения: {e}")

async def start_command_handler(update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /start. Отправляет приветственное сообщение.
    """
    chat_id = update.effective_chat.id
    welcome_message = "Привет\\! Я ваш бот для крипто\\-сигналов\\. Я буду присылать сигналы по мере их генерации\\."
    # Сообщение уже содержит экранирование для MarkdownV2
    await send_telegram_message(TELEGRAM_BOT_TOKEN, chat_id, welcome_message, parse_mode='MarkdownV2')
    # Или можно так:
    # await update.message.reply_text(escape_markdown_v2("Привет! Я ваш бот для крипто-сигналов. Я буду присылать сигналы по мере их генерации."), parse_mode='MarkdownV2')
    print(f"Команда /start обработана для чата {chat_id}.")


async def main_bot_runner():
    """
    Основная функция для настройки и запуска обработчиков команд бота.
    В этой версии polling не запускается, функция служит для демонстрации структуры.
    """
    print("Настройка приложения Telegram бота...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Добавление обработчика команды /start
    start_handler = CommandHandler('start', start_command_handler)
    application.add_handler(start_handler)
    print("Обработчик команды /start добавлен.")

    # Следующие строки закомментированы, так как реальный polling не требуется для этой задачи.
    # Они понадобятся, если бот должен будет постоянно работать и слушать команды.
    # print("Инициализация приложения...")
    # await application.initialize()
    # print("Запуск приложения...")
    # await application.start()
    # print("Запуск polling для получения обновлений...")
    # await application.updater.start_polling()
    # print("Бот запущен и слушает команды. Нажмите Ctrl+C для остановки.")
    # try:
    #     while True:
    #         await asyncio.sleep(3600) # Держать основной поток живым
    # except KeyboardInterrupt:
    #     print("Остановка бота...")
    #     await application.updater.stop()
    #     await application.stop()
    #     print("Бот остановлен.")

    print("Настройка обработчиков команд Telegram бота завершена (polling не запущен в этой версии).")
    # Для этой задачи функция просто настраивает обработчики и завершается.
    # Отправка сигналов будет инициироваться внешне.


if __name__ == '__main__':
    # --- Демонстрация функций ---
    
    # 1. Тестирование escape_markdown_v2
    test_text_md = "Это _тест_ с *жирным* и [ссылкой](http://example.com) и `кодом`."
    escaped_text_md = escape_markdown_v2(test_text_md)
    print(f"Оригинальный текст: {test_text_md}")
    print(f"Экранированный текст: {escaped_text_md}") # Должен быть: Это \_тест\_ с \*жирным\* и \[ссылкой\]\(http://example\.com\) и \`кодом\`\.

    # 2. Создание и форматирование тестового сигнала
    sample_signal = {
        'type': 'BUY', 
        'entry_price': 25000.123456, 
        'stop_loss': 24500.654321, 
        'take_profit': 26500.0,
        'confidence': 0.8578, 
        'justification': 'Сильный бычий тренд подтвержден RSI > 70 и MACD кроссовером. Цена пробила важный уровень сопротивления.',
        'price_precision': 2 # Для USD пары
    }
    coin = "BTC-USD" 
    
    formatted_message = format_signal_for_telegram(sample_signal, coin)
    print(f"\nОтформатированное сообщение для Telegram:\n{formatted_message}")

    # 3. Асинхронная отправка тестового сообщения и запуск main_bot_runner
    async def run_demonstration():
        print("\n--- Асинхронная демонстрация ---")
        # Отправка тестового сигнала администратору
        await send_telegram_message(TELEGRAM_BOT_TOKEN, ADMIN_CHAT_ID, formatted_message)
        
        # Демонстрация настройки обработчиков (без запуска polling)
        await main_bot_runner()

    # Запуск асинхронной демонстрации
    print("\nЗапуск демонстрации (отправка тестового сообщения и настройка main_bot_runner)...")
    asyncio.run(run_demonstration())
    
    print("\n--- Демонстрация модуля bot.py завершена ---")
