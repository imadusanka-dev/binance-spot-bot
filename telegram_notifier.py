import os
import time
import requests
from datetime import datetime
from logger import logger

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
        self.last_update_id = 0
        
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
    
    def send_latest_logs(self, lines: int = 20) -> bool:
        """
        Send the latest log entries to Telegram
        
        Args:
            lines: Number of latest log lines to send (default: 20)
            
        Returns:
            True if logs sent successfully, False otherwise
        """
        try:
            log_file = 'trading_bot.log'
            
            # Check if log file exists
            if not os.path.exists(log_file):
                return self.send_message("❌ Log file not found")
            
            # Read the latest log lines
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                latest_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            if not latest_lines:
                return self.send_message("📝 Log file is empty")
            
            # Format the log message
            log_content = ''.join(latest_lines)
            
            # Split into chunks if too long (Telegram has a 4096 character limit)
            max_length = 4000  # Leave some room for formatting
            
            if len(log_content) <= max_length:
                message = f"📋 <b>Latest {len(latest_lines)} Log Entries</b>\n\n<code>{log_content}</code>"
                return self.send_message(message)
            else:
                # Split into multiple messages
                chunks = []
                current_chunk = ""
                
                for line in latest_lines:
                    if len(current_chunk + line) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        current_chunk += line
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Send chunks
                success = True
                for i, chunk in enumerate(chunks):
                    header = f"📋 <b>Log Part {i+1}/{len(chunks)}</b>\n\n" if len(chunks) > 1 else "📋 <b>Latest Log Entries</b>\n\n"
                    message = f"{header}<code>{chunk}</code>"
                    if not self.send_message(message):
                        success = False
                    time.sleep(1)  # Small delay between messages
                
                return success
        
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return self.send_message(f"❌ Error reading logs: {str(e)}")
        
    def handle_telegram_commands(self) -> None:
        """
        Check for and handle incoming Telegram commands
        This should be called periodically or you can use webhooks
        """
        try:
            # Get updates from Telegram with offset to avoid reprocessing
            url = f"{self.base_url}/getUpdates"
            params = {
                'timeout': 10, 
                'limit': 10,
                'offset': self.last_update_id + 1 if self.last_update_id > 0 else None
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('ok') and data.get('result'):
                for update in data['result']:
                    # Update the last processed message ID
                    self.last_update_id = update['update_id']
                    
                    if 'message' in update:
                        message = update['message']
                        chat_id = str(message['chat']['id'])
                        text = message.get('text', '').lower().strip()
                        
                        # Only respond to messages from the configured chat
                        if chat_id == self.chat_id:
                            if text == 'log':
                                self.send_latest_logs()
                            elif text == 'status':
                                self.send_status_notification("🤖 Bot is running normally")
                            elif text.startswith('log '):
                                # Handle "log 50" to get specific number of lines
                                try:
                                    lines = int(text.split()[1])
                                    self.send_latest_logs(lines)
                                except (ValueError, IndexError):
                                    self.send_message("❌ Invalid format. Use: log [number]")
                    
        except Exception as e:
            logger.error(f"Error handling Telegram commands: {e}")

    def check_telegram_messages(self) -> None:
        """
        Wrapper method to safely check for Telegram messages
        """
        try:
            self.handle_telegram_commands()
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}")
            pass  # Don't let telegram errors stop the bot