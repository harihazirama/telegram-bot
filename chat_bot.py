import asyncio
import logging
import os
import openai

from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters

from speech_to_text import audio_to_text

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VOSK_MODEL_PATH = "./vosk_model"
MAX_HISTORY = 25

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# # Store conversation history
# conversation_history = [{"role": "system", "content": "You are a useful assistant."}]

# Store conversation history separately for each user
user_conversations = {}  # Key: user_id, Value: List of messages

def get_user_history(user_id):
    """Retrieve conversation history for a user, or initialize it."""
    if user_id not in user_conversations:
        user_conversations[user_id] = [{"role": "system", "content": "You are a helpful AI assistant. Answer only the latest question and do not include previous questions and answers in your response. Refer to a past question only if the user explicitly asks about it."}]
    return user_conversations[user_id]

def trim_user_history(user_id):
    """Keep only the last MAX_HISTORY messages for a user."""
    user_conversations[user_id] = [user_conversations[user_id][0]] + user_conversations[user_id][-MAX_HISTORY:]

# def trim_conversation_history(history):
#     """Keeps only the last MAX_HISTORY messages + system message."""
#     return [history[0]] + history[-MAX_HISTORY:]

async def send_long_message(update, text):
    max_length = 4000
    messages = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    for message in messages:
        await update.message.reply_text(message)

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your AI chatbot. How can I help you today?")

async def chat_with_gpt(update: Update, context: CallbackContext, user_text=None) -> None:
    user_id = update.message.from_user.id
    user_history = get_user_history(user_id)

    try:
        prompt = user_text or update.message.text.lower()
        user_history.append({"role": "user", "content": prompt})
        trim_user_history(user_id)

        async def send_typing():
            while True:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                await asyncio.sleep(5)

        typing_task = asyncio.create_task(send_typing())

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=user_history,
            timeout=90
        )

        if not response or not hasattr(response, "choices") or not response.choices:
            typing_task.cancel()
            raise ValueError("Invalid response from OpenRouter API (empty choices)")

        logger.info("Chat response: %s", response.choices[0].message)

        reply_from_bot = response.choices[0].message.content
        typing_task.cancel()

        # Store bot's response in the user's history
        user_history.append({"role": "assistant", "content": reply_from_bot})
        trim_user_history(user_id)

    except asyncio.TimeoutError:
        reply_from_bot = "The AI is taking too long to respond. Please try again later."
        logger.error("Timeout error: OpenRouter API took too long to respond", exc_info=True)
    except Exception as e:
        reply_from_bot = f"I encountered an error: {str(e)}"
        logger.error("Error encountered while chatting with AI: %s", e, exc_info=True)

    await send_long_message(update, reply_from_bot)

async def voice_chat(update: Update, context: CallbackContext) -> None:
    """Handles voice messages: converts them to text and sends to GPT."""
    user_audio_path = None
    try:
        # Download voice message
        voice = await update.message.voice.get_file()
        user_audio_path = f"/tmp/{voice.file_id}.ogg"
        await voice.download_to_drive(user_audio_path)
        logger.info(f"Voice message downloaded: {user_audio_path}")

        # Notify user
        processing_message = await update.message.reply_text("Processing voice message, please wait...")

        # Convert and transcribe
        user_text = await asyncio.to_thread(audio_to_text, user_audio_path, VOSK_MODEL_PATH)

        # Remove processing message
        await processing_message.delete()

        if user_text:
            await chat_with_gpt(update, context, user_text)
        else:
            await update.message.reply_text("Sorry, I couldn't understand the voice message.")

    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your voice message.")

    finally:
        # Cleanup: Remove temp files
        if user_audio_path and os.path.exists(user_audio_path):
            os.remove(user_audio_path)
            logger.info(f"Deleted temporary audio file: {user_audio_path}")

async def error_handler(update: Update, context: CallbackContext) -> None:
    logger.error(f"Exception occurred: {context.error}", exc_info=True)
    await update.message.reply_text("Oops! Something went wrong. Please try again later.")

def bot_run():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_gpt))
    app.add_handler(MessageHandler(filters.VOICE, voice_chat))
    app.add_error_handler(error_handler)
    app.run_polling()

if __name__ == "__main__":
    bot_run()
