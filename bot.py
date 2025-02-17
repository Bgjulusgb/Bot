import os
import logging
import asyncio
import numpy as np
import pandas as pd
from binance.spot import Spot
from binance.error import ClientError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from decimal import Decimal, getcontext

# Konfiguration
getcontext().prec = 8
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class Config:
    API_KEY = os.getenv("BINANCE_API_KEY", "HujbXuSFr2D28MZ9yWYJGTB0DAnKgtrR9qhb2s7Gt")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "AZATnLV6qbAdHNhmZaM8DWUqC23pvgT9rJKBhb")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7562186384:AAEvykmSQHkdkFMQhq8FJss")
    CHAT_ID = os.getenv("CHAT_ID", "6695197012")
    SYMBOL = "BTCUSDT"
    TESTNET = True

class BinanceClient:
    def __init__(self):
        self.client = Spot(
            api_key=Config.API_KEY,
            api_secret=Config.API_SECRET,
            base_url="https://testnet.binance.vision" if Config.TESTNET else "https://api.binance.com"
        )
    
    async def get_klines(self, interval='1h', limit=100):
        return self.client.klines(Config.SYMBOL, interval=interval, limit=limit)
    
    async def execute_order(self, side, quantity, stop_loss=None, take_profit=None):
        try:
            order = self.client.new_order(
                symbol=Config.SYMBOL,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            
            if stop_loss or take_profit:
                self.client.new_oco_order(
                    symbol=Config.SYMBOL,
                    side='SELL' if side == 'BUY' else 'BUY',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    price=take_profit
                )
            return order
        except ClientError as e:
            logging.error(f"Binance API Error: {e}")
            return None

class TradingStrategies:
    @staticmethod
    async def rsi_strategy(data, oversold=30, overbought=70, window=14):
        closes = [float(k[4]) for k in data]
        deltas = np.diff(closes)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gain).rolling(window).mean()
        avg_loss = pd.Series(loss).rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return 'BUY' if rsi.iloc[-1] < oversold else 'SELL' if rsi.iloc[-1] > overbought else None

    @staticmethod
    async def macd_strategy(data, fast=12, slow=26, signal=9):
        closes = [float(k[4]) for k in data]
        ema_fast = pd.Series(closes).ewm(span=fast).mean()
        ema_slow = pd.Series(closes).ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return 'BUY' if macd.iloc[-1] > signal_line.iloc[-1] else 'SELL'

    @staticmethod
    async def bollinger_strategy(data, window=20, num_std=2):
        closes = [float(k[4]) for k in data]
        series = pd.Series(closes)
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        last_price = closes[-1]
        return 'SELL' if last_price > upper.iloc[-1] else 'BUY' if last_price < lower.iloc[-1] else None

class TradingBot:
    def __init__(self):
        self.client = BinanceClient()
        self.strategies = {
            'RSI': TradingStrategies.rsi_strategy,
            'MACD': TradingStrategies.macd_strategy,
            'BOLL': TradingStrategies.bollinger_strategy
        }
        self.active_strategies = {}
        self.risk_per_trade = 0.02  # 2% pro Trade

    async def calculate_position_size(self):
        balance = await self.get_usdt_balance()
        return balance * Decimal(self.risk_per_trade)

    async def get_usdt_balance(self):
        try:
            balance = self.client.client.account()['balances']
            return Decimal(next(item['free'] for item in balance if item['asset'] == 'USDT'))
        except Exception as e:
            logging.error(f"Balance error: {e}")
            return Decimal(0)

    async def run_strategy(self, strategy_name, interval='1h'):
        while self.active_strategies.get(strategy_name, False):
            try:
                data = await self.client.get_klines(interval=interval)
                decision = await self.strategies[strategy_name](data)
                
                if decision:
                    quantity = await self.calculate_position_size()
                    price = float(data[-1][4])
                    btc_quantity = round(quantity / Decimal(price), 6)
                    
                    order = await self.client.execute_order(
                        side=decision,
                        quantity=btc_quantity
                    )
                    await self.send_alert(f"🚀 {strategy_name}-Signal: {decision}\n"
                                         f"Menge: {btc_quantity} BTC\n"
                                         f"Preis: {price} USDT")
                await asyncio.sleep(3600)  # 1h Intervall
            except Exception as e:
                logging.error(f"Strategy error: {e}")
                await asyncio.sleep(60)

    async def send_alert(self, message: str):
        try:
            app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
            await app.bot.send_message(chat_id=Config.CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

class TelegramBot:
    def __init__(self, trading_bot: TradingBot):
        self.trading_bot = trading_bot
        self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self.register_handlers()

    def register_handlers(self):
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("stop", self.stop),
            CommandHandler("balance", self.balance),
            CommandHandler("strategies", self.list_strategies),
            CommandHandler("activate", self.activate_strategy),
            CommandHandler("deactivate", self.deactivate_strategy),
        ]
        for handler in handlers:
            self.app.add_handler(handler)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = """🤖 *AI Trading Bot Pro* 🤖
        
        Verfügbare Befehle:
        /balance - Kontostand anzeigen
        /strategies - Verfügbare Strategien
        /activate [strategie] - Strategie aktivieren
        /deactivate [strategie] - Strategie deaktivieren
        /stop - Alle Strategien stoppen"""
        await context.bot.send_message(chat_id=Config.CHAT_ID, text=text, parse_mode='Markdown')

    async def activate_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        strategy = context.args[0].upper()
        if strategy in self.trading_bot.strategies:
            self.trading_bot.active_strategies[strategy] = True
            asyncio.create_task(self.trading_bot.run_strategy(strategy))
            await update.message.reply_text(f"✅ {strategy}-Strategie aktiviert")
        else:
            await update.message.reply_text("❌ Unbekannte Strategie")

    async def deactivate_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        strategy = context.args[0].upper()
        self.trading_bot.active_strategies[strategy] = False
        await update.message.reply_text(f"⛔ {strategy}-Strategie deaktiviert")

async def main():
    bot = TradingBot()
    telegram_bot = TelegramBot(bot)
    await telegram_bot.app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
