#!/usr/bin/env python3
"""
Telegram bot to control NIFTY bot via commands:
/startbot <otp>  - Start bot
/stopbot <otp>   - Stop bot
/status          - Check if running
"""
import os
import subprocess
import sys
from pathlib import Path

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
except ImportError:
    print("Install python-telegram-bot: pip install python-telegram-bot==20.7")
    sys.exit(1)

# Load config from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_IDS = {7311158582}  # Your chat ID from .env
OTP_SECRET = os.getenv("BOT_CONTROL_OTP", "changeme123")
BOT_DIR = Path("/home/samkumarg/Nifty intra")
PYTHON = sys.executable
MAIN_SCRIPT = BOT_DIR / "main.py"

# Track bot process
bot_process = None


def verify_otp(provided: str) -> bool:
    return provided == OTP_SECRET


def is_running() -> bool:
    global bot_process
    if bot_process is None:
        return False
    return bot_process.poll() is None


async def startbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USER_IDS:
        await update.message.reply_text("❌ Unauthorized")
        return

    args = context.args
    if not args or not verify_otp(args[0]):
        await update.message.reply_text("❌ Invalid OTP. Usage: /startbot <otp>")
        return

    global bot_process
    if is_running():
        await update.message.reply_text("⚠️ Bot already running")
        return

    try:
        env = os.environ.copy()
        env["NIFTY_BOT_MODE"] = "PAPER"
        bot_process = subprocess.Popen(
            [PYTHON, str(MAIN_SCRIPT)],
            cwd=BOT_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        await update.message.reply_text(f"✅ Bot started (PID: {bot_process.pid})")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")


async def stopbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USER_IDS:
        await update.message.reply_text("❌ Unauthorized")
        return

    args = context.args
    if not args or not verify_otp(args[0]):
        await update.message.reply_text("❌ Invalid OTP. Usage: /stopbot <otp>")
        return

    global bot_process
    if not is_running():
        await update.message.reply_text("⚠️ Bot not running")
        return

    bot_process.terminate()
    try:
        bot_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        bot_process.kill()
    bot_process = None
    await update.message.reply_text("🛑 Bot stopped")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USER_IDS:
        await update.message.reply_text("❌ Unauthorized")
        return

    state = "🟢 RUNNING" if is_running() else "🔴 STOPPED"
    await update.message.reply_text(f"Bot status: {state}")


def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in environment")
        sys.exit(1)

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("startbot", startbot))
    app.add_handler(CommandHandler("stopbot", stopbot))
    app.add_handler(CommandHandler("status", status))
    print("Telegram control bot started...")
    app.run_polling()


if __name__ == "__main__":
    main()