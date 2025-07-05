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
import ta

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
        self.last_crossover_time = None
        self.min_trade_amount = 11.0  # Minimum trade amount in USDT
        
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

    def get_historical_data(self, limit: int = 500) -> pd.DataFrame:
        """
        Get historical kline data from Binance
        
        Args:
            limit: Number of data points to retrieve (increased for better EMA calculation)
            
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

    def calculate_ema_ta(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate EMA using ta library
        
        Args:
            data: Price data series
            period: EMA period
            
        Returns:
            EMA series
        """
        try:
            # Use ta library for EMA calculation
            ema = ta.trend.EMAIndicator(close=data, window=period).ema_indicator()
            return ema
        except Exception as e:
            logger.error(f"Error calculating EMA with ta library: {e}")
            # Fallback to pandas ewm calculation
            alpha = 2 / (period + 1)
            ema = data.ewm(alpha=alpha, adjust=False).mean()
            return ema

    def generate_signals(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Generate trading signals based on EMA crossover with improved logic
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (signal, signal_data)
        """
        # Calculate EMAs using ta library
        df['ema_9'] = self.calculate_ema_ta(df['close'], self.ema_short)
        df['ema_20'] = self.calculate_ema_ta(df['close'], self.ema_long)
        
        # Get the latest values (need at least 3 points for crossover detection)
        current_ema_9 = df['ema_9'].iloc[-1]
        current_ema_20 = df['ema_20'].iloc[-1]
        prev_ema_9 = df['ema_9'].iloc[-2]
        prev_ema_20 = df['ema_20'].iloc[-2]
        prev2_ema_9 = df['ema_9'].iloc[-3]
        prev2_ema_20 = df['ema_20'].iloc[-3]
        
        current_price = df['close'].iloc[-1]
        current_timestamp = df['timestamp'].iloc[-1]
        
        signal_data = {
            'timestamp': current_timestamp,
            'price': current_price,
            'ema_9': current_ema_9,
            'ema_20': current_ema_20,
            'prev_ema_9': prev_ema_9,
            'prev_ema_20': prev_ema_20,
            'ema_diff': current_ema_9 - current_ema_20,
            'prev_ema_diff': prev_ema_9 - prev_ema_20
        }
        
        # Improved crossover detection with multiple confirmation points
        # Bullish crossover: EMA9 crosses above EMA20
        bullish_cross = (
            prev2_ema_9 <= prev2_ema_20 and
            prev_ema_9 <= prev_ema_20 and
            current_ema_9 > current_ema_20
        ) or (
            prev_ema_9 <= prev_ema_20 and
            current_ema_9 > current_ema_20 and
            abs(current_ema_9 - current_ema_20) > abs(prev_ema_9 - prev_ema_20)
        )
        
        # Bearish crossover: EMA9 crosses below EMA20
        bearish_cross = (
            prev2_ema_9 >= prev2_ema_20 and
            prev_ema_9 >= prev_ema_20 and
            current_ema_9 < current_ema_20
        ) or (
            prev_ema_9 >= prev_ema_20 and
            current_ema_9 < current_ema_20 and
            abs(current_ema_9 - current_ema_20) > abs(prev_ema_9 - prev_ema_20)
        )
        
        # Avoid duplicate signals by checking time since last crossover
        if self.last_crossover_time:
            time_diff = current_timestamp - self.last_crossover_time
            if time_diff < timedelta(hours=2):  # Minimum 2 hours between signals
                return 'HOLD', signal_data
        
        # Determine signal
        if bullish_cross:
            signal = 'BUY'
            self.last_crossover_time = current_timestamp
        elif bearish_cross:
            signal = 'SELL'
            self.last_crossover_time = current_timestamp
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

    def get_symbol_info(self) -> Dict:
        """
        Get symbol trading info for proper quantity formatting
        
        Returns:
            Dictionary with symbol info
        """
        try:
            info = self.client.get_symbol_info(self.symbol)
            
            # Find quantity and price filters
            quantity_filter = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
            price_filter = next(f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER')
            notional_filter = next(f for f in info['filters'] if f['filterType'] == 'NOTIONAL')
            
            return {
                'quantity_precision': len(quantity_filter['stepSize'].rstrip('0').split('.')[-1]),
                'price_precision': len(price_filter['tickSize'].rstrip('0').split('.')[-1]),
                'min_quantity': float(quantity_filter['minQty']),
                'min_notional': float(notional_filter['minNotional'])
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            # Default values for BTCUSDT
            return {
                'quantity_precision': 6,
                'price_precision': 2,
                'min_quantity': 0.00001,
                'min_notional': 10.0
            }

    def place_buy_order(self, usdt_amount: float) -> Optional[Dict]:
        """
        Place a market buy order using available USDT
        
        Args:
            usdt_amount: Amount of USDT to spend
            
        Returns:
            Order result or None if failed
        """
        try:
            symbol_info = self.get_symbol_info()
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calculate BTC quantity (reserve some USDT for fees)
            fee_buffer = 0.001  # 0.1% fee buffer
            usable_amount = usdt_amount * (1 - fee_buffer)
            btc_quantity = usable_amount / current_price
            
            # Round to proper precision
            btc_quantity = round(btc_quantity, symbol_info['quantity_precision'])
            
            # Check minimum requirements
            if btc_quantity < symbol_info['min_quantity']:
                logger.warning(f"Quantity too small: {btc_quantity} < {symbol_info['min_quantity']}")
                return None
                
            if btc_quantity * current_price < symbol_info['min_notional']:
                logger.warning(f"Notional value too small: ${btc_quantity * current_price:.2f}")
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
                self.telegram.send_trade_notification(
                    signal="BUY",
                    price=0,
                    amount=usdt_amount,
                    asset="USDT",
                    success=False
                )
            return None

    def place_sell_order(self, btc_amount: float) -> Optional[Dict]:
        """
        Place a market sell order for available BTC
        
        Args:
            btc_amount: Amount of BTC to sell
            
        Returns:
            Order result or None if failed
        """
        try:
            symbol_info = self.get_symbol_info()
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Round to proper precision
            btc_amount = round(btc_amount, symbol_info['quantity_precision'])
            
            # Check minimum requirements
            if btc_amount < symbol_info['min_quantity']:
                logger.warning(f"Quantity too small: {btc_amount} < {symbol_info['min_quantity']}")
                return None
                
            if btc_amount * current_price < symbol_info['min_notional']:
                logger.warning(f"Notional value too small: ${btc_amount * current_price:.2f}")
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
                self.telegram.send_trade_notification(
                    signal="SELL",
                    price=0,
                    amount=btc_amount,
                    asset="BTC",
                    success=False
                )
            return None

    def execute_trade(self, signal: str, signal_data: Dict) -> None:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            signal_data: Signal metadata
        """
        logger.info(f"=== SIGNAL ANALYSIS ===")
        logger.info(f"Signal: {signal}")
        logger.info(f"Price: ${signal_data['price']:.2f}")
        logger.info(f"EMA9: ${signal_data['ema_9']:.2f}")
        logger.info(f"EMA20: ${signal_data['ema_20']:.2f}")
        logger.info(f"EMA Diff: {signal_data['ema_diff']:.2f}")
        logger.info(f"Prev EMA Diff: {signal_data['prev_ema_diff']:.2f}")
        logger.info(f"Last Signal: {self.last_signal}")
        logger.info(f"=======================")
        
        if signal == 'HOLD':
            logger.info("No action needed - HOLD signal")
            return
            
        if signal == self.last_signal:
            logger.info(f"Signal unchanged ({signal}) - no action needed")
            return
            
        # Get current balances
        balances = self.get_account_balance()
        btc_balance = balances.get('BTC', 0)
        usdt_balance = balances.get('USDT', 0)
        
        logger.info(f"Current balances - BTC: {btc_balance:.6f}, USDT: ${usdt_balance:.2f}")
        
        if signal == 'BUY':
            if usdt_balance >= self.min_trade_amount:
                logger.info(f"Executing BUY signal with ${usdt_balance:.2f} USDT")
                order = self.place_buy_order(usdt_balance)
                if order:
                    self.position = 'BTC'
                    self.last_signal = signal
                    logger.info("BUY order executed successfully")
                else:
                    logger.error("BUY order failed")
            else:
                logger.warning(f"Insufficient USDT balance: ${usdt_balance:.2f} < ${self.min_trade_amount:.2f}")
                
        elif signal == 'SELL':
            if btc_balance > 0:
                logger.info(f"Executing SELL signal with {btc_balance:.6f} BTC")
                order = self.place_sell_order(btc_balance)
                if order:
                    self.position = 'USDT'
                    self.last_signal = signal
                    logger.info("SELL order executed successfully")
                else:
                    logger.error("SELL order failed")
            else:
                logger.warning(f"No BTC balance to sell: {btc_balance:.6f}")

    def run_strategy(self) -> None:
        """
        Run the complete trading strategy
        """
        try:
            logger.info("=== Running EMA Trading Strategy ===")
            
            # Get historical data
            df = self.get_historical_data()
            
            if len(df) < 50:  # Ensure we have enough data
                logger.warning("Insufficient historical data for EMA calculation")
                return
            
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

    def run_backtest(self, days: int = 30) -> None:
        """
        Run a simple backtest to verify strategy logic
        
        Args:
            days: Number of days to backtest
        """
        try:
            logger.info(f"=== Running {days}-day backtest ===")
            
            # Get more historical data for backtest
            df = self.get_historical_data(limit=days * 24)  # 24 hours per day
            
            if len(df) < 50:
                logger.warning("Insufficient data for backtest")
                return
            
            # Calculate EMAs using ta library
            df['ema_9'] = self.calculate_ema_ta(df['close'], self.ema_short)
            df['ema_20'] = self.calculate_ema_ta(df['close'], self.ema_long)
            
            # Identify crossovers
            df['signal'] = 'HOLD'
            df['prev_ema_9'] = df['ema_9'].shift(1)
            df['prev_ema_20'] = df['ema_20'].shift(1)
            
            # Mark crossovers
            bullish_mask = (df['prev_ema_9'] <= df['prev_ema_20']) & (df['ema_9'] > df['ema_20'])
            bearish_mask = (df['prev_ema_9'] >= df['prev_ema_20']) & (df['ema_9'] < df['ema_20'])
            
            df.loc[bullish_mask, 'signal'] = 'BUY'
            df.loc[bearish_mask, 'signal'] = 'SELL'
            
            # Count signals
            buy_signals = len(df[df['signal'] == 'BUY'])
            sell_signals = len(df[df['signal'] == 'SELL'])
            
            logger.info(f"Backtest Results:")
            logger.info(f"- Buy signals: {buy_signals}")
            logger.info(f"- Sell signals: {sell_signals}")
            logger.info(f"- Data points: {len(df)}")
            logger.info(f"- Latest EMA9: ${df['ema_9'].iloc[-1]:.2f}")
            logger.info(f"- Latest EMA20: ${df['ema_20'].iloc[-1]:.2f}")
            logger.info(f"- Latest Price: ${df['close'].iloc[-1]:.2f}")
            
            # Show recent signals
            recent_signals = df[df['signal'] != 'HOLD'].tail(5)
            if not recent_signals.empty:
                logger.info("Recent signals:")
                for _, row in recent_signals.iterrows():
                    logger.info(f"  {row['timestamp']}: {row['signal']} at ${row['close']:.2f}")
            
            logger.info("=== Backtest completed ===\n")
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")

    def start_bot(self) -> None:
        """
        Start the trading bot with hourly execution
        """
        logger.info("Starting Binance EMA Trading Bot...")
        
        # Run backtest first
        self.run_backtest()
        
        # Run strategy immediately on start
        self.run_strategy()
        
        # Schedule to run every hour at minute 5 (to avoid exact hour timing issues)
        schedule.every().hour.at(":05").do(self.run_strategy)
        
        logger.info("Bot scheduled to run every hour at minute 5. Press Ctrl+C to stop.")
        
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
        testnet=False  # IMPORTANT: Set to False only when you're ready for live trading
    )
    
    # Start the bot
    bot.start_bot()

if __name__ == "__main__":
    main()