import os
from dotenv import load_dotenv
from binance_ema_bot import BinanceEMABot
from logger import logger

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to run the trading bot
    """
    # Load API credentials from environment variables
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    if not API_KEY or not API_SECRET:
        logger.error("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Initialize bot (testnet=True for safety - change to False for live trading)
    bot = BinanceEMABot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        testnet=False  # IMPORTANT: Set to False only when you're ready for live trading
    )
    
    # Start the bot
    bot.start_bot()

if __name__ == "__main__":
    main()