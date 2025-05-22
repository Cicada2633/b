import asyncio
import time
import logging
import os
import pandas as pd
import numpy as np # For NaN and feature selection

# --- Project Module Imports ---
# Using try-except for robustness, especially if run from different contexts
try:
    from AdvancedCryptoBot.src.data_collection import market_data_api as market_api
    from AdvancedCryptoBot.src.data_collection import news_api as news_api_module
    # data_caching is used indirectly by market_api and news_api_module
    from AdvancedCryptoBot.src.feature_engineering import technical_indicators as ti_module
    from AdvancedCryptoBot.src.feature_engineering import sentiment_analysis as sa_module
    from AdvancedCryptoBot.src.ml_data_preparation import data_preparer as dp_module
    from AdvancedCryptoBot.src.ml_model import model_trainer as mt_module
    from AdvancedCryptoBot.src.trading_logic import signal_generator as sg_module
    from AdvancedCryptoBot.src.telegram_bot import bot as tg_bot_module
    from AdvancedCryptoBot.src.feedback_system import feedback_manager as fm_module # Feedback system
import uuid # For unique signal IDs
from datetime import datetime # For timestamp_generated

except ImportError as e:
    print(f"ImportError during initial project imports: {e}. Attempting fallback relative imports...")
    # This block allows running the script directly from src/ or a similar context
    # where the AdvancedCryptoBot package is not directly in PYTHONPATH.
    # It adjusts sys.path to include the 'src' directory's parent if needed.
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.abspath(os.path.join(current_script_path, '..', '..')) 
    # (assuming main_controller.py is in src/) -> project_root should be AdvancedCryptoBot
    
    # If src/ is not the project_root, adjust:
    # project_root_path = os.path.abspath(os.path.join(current_script_path, '..')) # If main_controller is in src/
    
    if project_root_path not in os.sys.path:
         os.sys.path.insert(0, project_root_path)

    # Re-attempt imports (now assuming AdvancedCryptoBot is in sys.path)
    from src.data_collection import market_data_api as market_api
    from src.data_collection import news_api as news_api_module
    from src.feature_engineering import technical_indicators as ti_module
    from src.feature_engineering import sentiment_analysis as sa_module
    from src.ml_data_preparation import data_preparer as dp_module
    from src.ml_model import model_trainer as mt_module
    from src.trading_logic import signal_generator as sg_module
    from src.telegram_bot import bot as tg_bot_module
    from src.feedback_system import feedback_manager as fm_module # Feedback system
import uuid # For unique signal IDs
from datetime import datetime # For timestamp_generated

# --- Configuration ---
DEFAULT_CONFIG = {
    "top_n_coins": 3,  # Start with fewer coins for testing
    "timeframe": "1H", # OKX timeframe for analysis
    # Generic model path, assumes model file exists.
    # In a real app, model paths would be more dynamic (e.g., per coin).
    "model_path_template": os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'dummy_coin_usdt_1h_rf_classifier.joblib'),
    "news_api_key": "e3318f2c424c49218e1792c2fcf22863", # Placeholder, use env vars in production
    "news_keywords": "bitcoin OR ethereum OR solana OR cardano OR ripple OR crypto OR cryptocurrency",
    "news_fetch_limit": 20, # Number of articles from NewsAPI
    "rss_feed_urls": news_api_module.RSS_FEEDS if hasattr(news_api_module, 'RSS_FEEDS') else {},
    "signal_config": {
        "confidence_threshold": 0.51, # Adjusted for dummy model (often 0.5 for binary)
        "risk_reward_ratio": 1.5,
        "stop_loss_type": 'atr', 
        "stop_loss_value": 1.5, # For ATR, it's a multiplier
        "signal_type": 'spot',   
        "price_precision": 4     
    },
    "loop_interval_seconds": 300, # 5 minutes
    "max_historical_candles": 200, # For fetching data for indicators
    "news_lookback_hours": 24 # For aligning news with market data
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)


async def process_single_coin(coin_symbol, base_currency, config, model, news_df_processed):
    """
    Processes a single cryptocurrency: fetches data, makes predictions, and sends signals.
    """
    okx_symbol = market_api.format_symbol_for_okx(coin_symbol, base_currency)
    logger.info(f"Processing {okx_symbol}...")

    try:
        # 1. Fetch Market Data
        logger.debug(f"Fetching market data for {okx_symbol}...")
        market_df_raw = await asyncio.to_thread(
            market_api.get_okx_historical_data, 
            okx_symbol, 
            config["timeframe"], 
            limit=config["max_historical_candles"]
        )

        if market_df_raw is None or market_df_raw.empty:
            logger.warning(f"No market data returned for {okx_symbol}. Skipping.")
            return

        # 2. Add Technical Indicators
        logger.debug(f"Adding technical indicators for {okx_symbol}...")
        market_df_indicators = await asyncio.to_thread(ti_module.add_all_indicators, market_df_raw.copy())
        if market_df_indicators.empty: # add_all_indicators might return empty if input is bad
            logger.warning(f"Failed to add indicators or received empty df for {okx_symbol}. Skipping.")
            return
            
        # 3. Align and Merge Data (Feature Preparation)
        logger.debug(f"Aligning and merging data for {okx_symbol}...")
        # Ensure news_df_processed is a copy if it's going to be modified or indexed into
        # dp_module.align_and_merge_data expects news_df to have 'publishedAt'
        current_news_df = news_df_processed.copy() if news_df_processed is not None and not news_df_processed.empty else pd.DataFrame()

        merged_df = await asyncio.to_thread(
            dp_module.align_and_merge_data,
            market_df_indicators.copy(), # Pass a copy
            current_news_df, # Pass the (potentially empty) copy
            news_time_window_td=pd.Timedelta(hours=config.get("news_lookback_hours", 4))
        )

        if merged_df is None or merged_df.empty:
            logger.warning(f"Merged data is empty for {okx_symbol}. Skipping.")
            return
            
        # 4. Prepare current features for prediction (last row)
        logger.debug(f"Preparing current features for {okx_symbol}...")
        
        # Columns model was trained on (excluding target). This needs to be known.
        # For demonstration, we assume the model was trained on all numeric columns available
        # after merging, excluding 'target' and identifiers like 'timestamp' (if it's a column).
        # In a real scenario, feature_names should be saved with the model or defined consistently.
        
        # Get the last row for prediction
        last_row_features = merged_df.iloc[[-1]] # Keep as DataFrame

        # Select only numeric features the model would expect
        # This is a simplified approach; feature names list from training is ideal
        X_predict = last_row_features.select_dtypes(include=np.number)
        
        # Ensure no NaNs in the prediction row (critical for most models)
        if X_predict.isnull().any().any():
            logger.warning(f"NaNs found in the feature row for {okx_symbol} before prediction. Attempting ffill/bfill.")
            # Example: ffill based on the column's own history (not ideal for a single row)
            # For a single row, we might need to fill based on global means or default values,
            # or ensure data prep handles NaNs in the last row robustly.
            # For now, if NaNs exist in the single row, prediction might fail or be unreliable.
            # A simple ffill/bfill on a single row won't do much.
            # Let's assume the model_trainer or model itself can handle some NaNs or they are filled by data_preparer.
            # A more robust way: X_predict = X_predict.fillna(method='ffill').fillna(method='bfill') from training data characteristics
            # For now, we will proceed, but log if NaNs are present.
            nan_cols = X_predict.columns[X_predict.isnull().any()].tolist()
            if nan_cols:
                 logger.warning(f"NaNs still present in prediction features for {okx_symbol} in columns: {nan_cols}. Prediction might be unreliable or fail.")
                 # Option: fill with 0 or mean, or skip prediction
                 X_predict = X_predict.fillna(0) # Simplistic fill with 0 for NaNs

        if X_predict.empty:
            logger.warning(f"Feature set for prediction is empty for {okx_symbol}. Skipping.")
            return

        # 5. Make Prediction
        logger.debug(f"Making prediction for {okx_symbol}...")
        # Ensure model object is valid
        if model is None:
            logger.error(f"Model is not loaded. Cannot make prediction for {okx_symbol}.")
            return

        prediction, probability = await asyncio.to_thread(mt_module.make_prediction, model, X_predict)

        if prediction is None:
            logger.warning(f"Prediction failed for {okx_symbol}.")
            return
            
        # 6. Generate and Send Signal
        pred_class = prediction[0]
        # Probability for the predicted class
        pred_prob = probability[0][pred_class] if probability is not None and len(probability[0]) > pred_class else probability[0][0] if probability is not None else 0.0

        logger.info(f"Prediction for {okx_symbol}: Class={pred_class}, Prob={pred_prob:.2%}")

        current_price = market_df_raw['close'].iloc[-1]
        atr_value_for_signal = None
        if config["signal_config"]["stop_loss_type"] == 'atr':
            # Ensure ATR column exists and is not NaN
            # The column name from ti_module.add_all_indicators is 'ATRr_14' for default ATR(14)
            atr_col_name = 'ATRr_14' # Default name from technical_indicators.py
            if atr_col_name in market_df_indicators.columns and pd.notna(market_df_indicators[atr_col_name].iloc[-1]):
                atr_value_for_signal = market_df_indicators[atr_col_name].iloc[-1]
            else:
                logger.warning(f"ATR value for {atr_col_name} is missing or NaN for {okx_symbol}. Cannot use ATR stop-loss.")
                # Fallback or skip signal: For now, we'll let generate_signal handle missing ATR
        
        logger.debug(f"Generating signal for {okx_symbol} with price={current_price}, ATR={atr_value_for_signal}")
        signal_data = await asyncio.to_thread(
            sg_module.generate_signal, 
            pred_class, 
            pred_prob, 
            current_price, 
            config["signal_config"], 
            atr_value=atr_value_for_signal
        )

        if signal_data:
            logger.info(f"Signal generated for {okx_symbol}: {signal_data}")
            
            # Log the signal before sending to Telegram
            signal_id = str(uuid.uuid4())
            timestamp_generated = pd.Timestamp.now(tz='UTC')
            
            # Ensure model_features_dict is from the exact features used for prediction
            # X_predict is already a DataFrame with one row
            model_features_dict = X_predict.iloc[0].to_dict() 
            
            logger.debug(f"Logging signal {signal_id} for {okx_symbol}...")
            await asyncio.to_thread(
                fm_module.log_generated_signal,
                signal_id=signal_id,
                timestamp_generated=timestamp_generated,
                coin_symbol=okx_symbol,
                signal_type=signal_data['type'], # From signal_data
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                confidence=signal_data['confidence'],
                model_features_dict=model_features_dict,
                predicted_class=pred_class, # From model prediction
                predicted_probability=pred_prob # From model prediction
            )
            
            # Send Telegram message
            message = tg_bot_module.format_signal_for_telegram(signal_data, okx_symbol)
            await tg_bot_module.send_telegram_message(
                tg_bot_module.TELEGRAM_BOT_TOKEN, 
                tg_bot_module.ADMIN_CHAT_ID, 
                message
            )
            logger.info(f"Signal {signal_id} for {okx_symbol} sent to Telegram.")
        else:
            logger.info(f"No signal generated for {okx_symbol} based on current prediction and config.")

    except Exception as e:
        logger.error(f"Error processing {okx_symbol}: {e}", exc_info=True) # exc_info=True for traceback

async def run_main_loop(config):
    """
    Main operational loop for the bot.
    """
    logger.info("Bot starting...")

    # Initialize signals log file
    logger.info("Initializing signals log...")
    await asyncio.to_thread(fm_module.initialize_signals_log)
    
    # Load the model once
    model_path = config["model_path_template"]
    logger.info(f"Loading model from: {model_path}")
    model = await asyncio.to_thread(mt_module.load_model, model_path)
    if model is None:
        logger.critical(f"Failed to load model from {model_path}. Bot cannot operate. Exiting.")
        return
    logger.info("Model loaded successfully.")

    # Initialize Telegram bot command handlers (but don't block/poll here)
    # This sets up handlers but doesn't poll in its current form in bot.py
    # logger.info("Setting up Telegram command handlers...")
    # await tg_bot_module.main_bot_runner() 


    while True:
        logger.info("Starting new processing cycle...")
        
        # 1. Fetch Top Coins
        try:
            logger.debug("Fetching top N coins from CoinGecko...")
            top_coins_symbols = await asyncio.to_thread(market_api.get_top_n_coins_coingecko, n=config["top_n_coins"])
            if not top_coins_symbols:
                logger.warning("No top coins list received from CoinGecko. Skipping this cycle.")
                await asyncio.sleep(config["loop_interval_seconds"])
                continue
            logger.info(f"Top coins to process: {top_coins_symbols}")
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}", exc_info=True)
            await asyncio.sleep(config["loop_interval_seconds"])
            continue

        # 2. Fetch and Process News Data (once per cycle for all coins)
        news_df_processed = pd.DataFrame() # Default to empty if no news
        try:
            logger.debug("Fetching news articles (NewsAPI and RSS)...")
            news_api_articles = await asyncio.to_thread(
                news_api_module.fetch_crypto_news_newsapi,
                config["news_api_key"],
                config["news_keywords"],
                n_articles=config["news_fetch_limit"]
            )
            
            all_rss_articles = []
            if config.get("rss_feed_urls"):
                for name, url in config["rss_feed_urls"].items():
                    logger.debug(f"Fetching RSS from {name}...")
                    rss_arts = await asyncio.to_thread(news_api_module.fetch_news_from_rss, url, name)
                    if rss_arts:
                        all_rss_articles.extend(rss_arts)
            
            combined_articles = (news_api_articles or []) + (all_rss_articles or [])
            
            if combined_articles:
                news_df = pd.DataFrame(combined_articles)
                if not news_df.empty:
                    logger.debug("Adding sentiment scores to news...")
                    news_df_processed = await asyncio.to_thread(sa_module.add_sentiment_scores_to_news, news_df.copy())
                    logger.info(f"Processed {len(news_df_processed)} news articles with sentiment.")
                else:
                    logger.info("No combined news articles to process.")
            else:
                logger.info("No news articles fetched from any source.")

        except Exception as e:
            logger.error(f"Error fetching or processing news data: {e}", exc_info=True)
            # Continue with empty news_df_processed

        # 3. Process each coin in parallel
        tasks = []
        for coin_sym_lower in top_coins_symbols: # CoinGecko returns lowercase symbols like 'btc'
            coin_sym_upper = coin_sym_lower.upper() # Convert to uppercase like 'BTC' for OKX
            tasks.append(process_single_coin(coin_sym_upper, "USDT", config, model, news_df_processed))
        
        logger.info(f"Gathering results for {len(tasks)} coin processing tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log the exception from process_single_coin if it wasn't caught and logged inside
                # Assuming coin_symbols list aligns with tasks for logging
                # This might not be perfectly aligned if top_coins_symbols had an issue
                coin_for_error = top_coins_symbols[i] if i < len(top_coins_symbols) else "Unknown coin"
                logger.error(f"Unhandled exception during processing for {coin_for_error}: {result}", exc_info=result)
        
        logger.info(f"Processing cycle finished. Waiting for {config['loop_interval_seconds']} seconds...")
        await asyncio.sleep(config["loop_interval_seconds"])

if __name__ == '__main__':
    logger.info("Initializing Main Controller...")
    
    # Check if dummy model exists, if not, we might need to run model_trainer's main
    model_file_path = DEFAULT_CONFIG["model_path_template"]
    if not os.path.exists(model_file_path):
        logger.warning(f"Model file not found at {model_file_path}.")
        logger.warning("Please ensure the dummy model is created by running the model_trainer.py script's main block,")
        logger.warning("or that the path in DEFAULT_CONFIG correctly points to an existing model.")
        # Decide if to exit or continue (will fail at model load)
        # exit("Exiting due to missing model file.") 
        # For now, let it try to load and fail, as per run_main_loop logic.

    try:
        asyncio.run(run_main_loop(DEFAULT_CONFIG))
    except KeyboardInterrupt:
        logger.info("Main controller interrupted by user. Shutting down...")
    except Exception as e:
        logger.critical(f"Critical unhandled exception in main_controller: {e}", exc_info=True)
    finally:
        logger.info("Main controller shutdown complete.")
