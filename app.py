import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import schedule
import os
import requests
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message to send
            parse_mode: Message format (HTML, Markdown, or None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_trade_notification(self, signal: str, price: float, amount: float, 
                              asset: str, success: bool = True) -> bool:
        """
        Send trade notification
        
        Args:
            signal: BUY or SELL
            price: Execution price
            amount: Amount traded
            asset: Asset symbol (BTC or USDT)
            success: Whether trade was successful
            
        Returns:
            True if notification sent successfully
        """
        emoji = "✅" if success else "❌"
        status = "EXECUTED" if success else "FAILED"
        
        if signal == "BUY":
            message = f"""
{emoji} <b>{signal} ORDER {status}</b>

💰 Spent: ${amount:.2f} USDT
₿ Price: ${price:.2f}
📊 Strategy: EMA9 crossed above EMA20
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        else:  # SELL
            message = f"""
{emoji} <b>{signal} ORDER {status}</b>

₿ Sold: {amount:.6f} BTC
💰 Price: ${price:.2f}
📊 Strategy: EMA9 crossed below EMA20
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message.strip())
    
    def send_error_notification(self, error_type: str, error_message: str) -> bool:
        """
        Send error notification
        
        Args:
            error_type: Type of error
            error_message: Error details
            
        Returns:
            True if notification sent successfully
        """
        message = f"""
🚨 <b>TRADING BOT ERROR</b>

⚠️ Type: {error_type}
📝 Details: {error_message}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the bot logs for more details.
        """
        
        return self.send_message(message.strip())
    
    def send_status_notification(self, message: str) -> bool:
        """
        Send general status notification
        
        Args:
            message: Status message
            
        Returns:
            True if notification sent successfully
        """
        formatted_message = f"""
ℹ️ <b>BOT STATUS</b>

{message}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(formatted_message.strip())

class BinanceEMABot:
    def __init__(self, api_key: str, api_secret: str, telegram_token: str = None, 
                 telegram_chat_id: str = None, testnet: bool = True):
        """
        Initialize the Binance EMA Trading Bot
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            telegram_token: Telegram bot token (optional)
            telegram_chat_id: Telegram chat ID (optional)
            testnet: Whether to use testnet (default: True for safety)
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.symbol = 'BTCUSDT'
        self.interval = Client.KLINE_INTERVAL_1HOUR
        self.ema_short = 9
        self.ema_long = 20
        self.last_signal = None
        self.position = 'NONE'  # 'BTC', 'USDT', or 'NONE'
        
        # Initialize Telegram notifier
        self.telegram = None
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram notifications disabled - missing token or chat_id")
        
        # Verify API connection
        try:
            self.client.ping()
            logger.info("Successfully connected to Binance API")
            if self.telegram:
                self.telegram.send_status_notification("🚀 Trading bot started successfully!\n📊 Strategy: EMA9/EMA20 Crossover")
        except Exception as e:
            error_msg = f"Failed to connect to Binance API: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("API Connection", str(e))
            raise

    def get_historical_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Get historical kline data from Binance
        
        Args:
            limit: Number of data points to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to appropriate data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except BinanceAPIException as e:
            error_msg = f"Error fetching historical data: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Data Fetch Error", str(e))
            raise

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price data series
            period: EMA period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Generate trading signals based on EMA crossover
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (signal, signal_data)
        """
        # Calculate EMAs
        df['ema_9'] = self.calculate_ema(df['close'], self.ema_short)
        df['ema_20'] = self.calculate_ema(df['close'], self.ema_long)
        
        # Get the latest values
        current_ema_9 = df['ema_9'].iloc[-1]
        current_ema_20 = df['ema_20'].iloc[-1]
        prev_ema_9 = df['ema_9'].iloc[-2]
        prev_ema_20 = df['ema_20'].iloc[-2]
        
        current_price = df['close'].iloc[-1]
        
        signal_data = {
            'timestamp': df['timestamp'].iloc[-1],
            'price': current_price,
            'ema_9': current_ema_9,
            'ema_20': current_ema_20,
            'prev_ema_9': prev_ema_9,
            'prev_ema_20': prev_ema_20
        }
        
        # Determine signal
        if prev_ema_9 <= prev_ema_20 and current_ema_9 > current_ema_20:
            signal = 'BUY'
        elif prev_ema_9 >= prev_ema_20 and current_ema_9 < current_ema_20:
            signal = 'SELL'
        else:
            signal = 'HOLD'
            
        return signal, signal_data

    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances for BTC and USDT
        
        Returns:
            Dictionary with BTC and USDT balances
        """
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                if balance['asset'] in ['BTC', 'USDT']:
                    balances[balance['asset']] = float(balance['free'])
                    
            return balances
            
        except BinanceAPIException as e:
            error_msg = f"Error getting account balance: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Balance Fetch Error", str(e))
            raise

    def place_buy_order(self, usdt_amount: float) -> Optional[Dict]:
        """
        Place a market buy order using all available USDT
        
        Args:
            usdt_amount: Amount of USDT to spend
            
        Returns:
            Order result or None if failed
        """
        try:
            # Get current price to estimate quantity
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calculate BTC quantity (with small buffer for fees)
            btc_quantity = (usdt_amount * 0.999) / current_price
            
            # Round to appropriate precision (Binance BTC precision is usually 6 decimal places)
            btc_quantity = round(btc_quantity, 6)
            
            if btc_quantity * current_price < 10:  # Minimum order value is usually $10
                logger.warning(f"Order value too small: ${btc_quantity * current_price:.2f}")
                return None
                
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=btc_quantity
            )
            
            logger.info(f"Buy order placed: {btc_quantity} BTC at ~${current_price:.2f}")
            
            # Send Telegram notification
            if self.telegram:
                self.telegram.send_trade_notification(
                    signal="BUY",
                    price=current_price,
                    amount=usdt_amount,
                    asset="USDT",
                    success=True
                )
            
            return order
            
        except BinanceAPIException as e:
            error_msg = f"Error placing buy order: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Buy Order Error", str(e))
            return None

    def place_sell_order(self, btc_amount: float) -> Optional[Dict]:
        """
        Place a market sell order for all available BTC
        
        Args:
            btc_amount: Amount of BTC to sell
            
        Returns:
            Order result or None if failed
        """
        try:
            # Round to appropriate precision
            btc_amount = round(btc_amount, 6)
            
            # Get current price for logging
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            if btc_amount * current_price < 10:  # Minimum order value check
                logger.warning(f"Order value too small: ${btc_amount * current_price:.2f}")
                return None
                
            order = self.client.order_market_sell(
                symbol=self.symbol,
                quantity=btc_amount
            )
            
            logger.info(f"Sell order placed: {btc_amount} BTC at ~${current_price:.2f}")
            
            # Send Telegram notification
            if self.telegram:
                self.telegram.send_trade_notification(
                    signal="SELL",
                    price=current_price,
                    amount=btc_amount,
                    asset="BTC",
                    success=True
                )
            
            return order
            
        except BinanceAPIException as e:
            error_msg = f"Error placing sell order: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Sell Order Error", str(e))
            return None

    def execute_trade(self, signal: str, signal_data: Dict) -> None:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            signal_data: Signal metadata
        """
        logger.info(f"Signal: {signal} | Price: ${signal_data['price']:.2f} | "
                   f"EMA9: ${signal_data['ema_9']:.2f} | EMA20: ${signal_data['ema_20']:.2f}")
        
        if signal == 'HOLD' or signal == self.last_signal:
            logger.info("No action needed")
            return
            
        # Get current balances
        balances = self.get_account_balance()
        btc_balance = balances.get('BTC', 0)
        usdt_balance = balances.get('USDT', 0)
        
        logger.info(f"Current balances - BTC: {btc_balance:.6f}, USDT: ${usdt_balance:.2f}")
        
        if signal == 'BUY' and usdt_balance > 10:  # Minimum $10 to trade
            logger.info(f"Executing BUY signal with ${usdt_balance:.2f} USDT")
            order = self.place_buy_order(usdt_balance)
            if order:
                self.position = 'BTC'
                self.last_signal = signal
                
        elif signal == 'SELL' and btc_balance > 0:
            logger.info(f"Executing SELL signal with {btc_balance:.6f} BTC")
            order = self.place_sell_order(btc_balance)
            if order:
                self.position = 'USDT'
                self.last_signal = signal
        else:
            logger.info("Insufficient balance to execute trade")

    def run_strategy(self) -> None:
        """
        Run the complete trading strategy
        """
        try:
            logger.info("=== Running EMA Trading Strategy ===")
            
            # Get historical data
            df = self.get_historical_data()
            
            # Generate signals
            signal, signal_data = self.generate_signals(df)
            
            # Execute trade
            self.execute_trade(signal, signal_data)
            
            logger.info("=== Strategy execution completed ===\n")
            
        except Exception as e:
            error_msg = f"Error in strategy execution: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Strategy Execution Error", str(e))

    def start_bot(self) -> None:
        """
        Start the trading bot with hourly execution
        """
        logger.info("Starting Binance EMA Trading Bot...")
        
        # Run immediately on start
        self.run_strategy()
        
        # Schedule to run every hour
        schedule.every().hour.at(":01").do(self.run_strategy)  # Run at 1 minute past each hour
        
        logger.info("Bot scheduled to run every hour. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            if self.telegram:
                self.telegram.send_status_notification("🛑 Trading bot stopped by user")
        except Exception as e:
            error_msg = f"Unexpected error in bot main loop: {e}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_notification("Bot Critical Error", str(e))

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
        testnet=False  # Set to False for live trading
    )
    
    # Start the bot
    bot.start_bot()

if __name__ == "__main__":
    main()