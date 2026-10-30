import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes,
)
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
#import json
import logging
from datetime import datetime, timezone
import csv


#LOG_FILE = Path(__file__).parent / "dialog_log.jsonl"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LOG_FILE = Path(__file__).parent / "dialog_log.csv"

def save_dialog(user_id: int, username: str, question: str, answer: str, sources: list[str]):
    """Дописывает одну запись в CSV-файл (разделитель ';')."""
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "username", "question", "answer", "sources"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            user_id,
            username,
            question.replace("\n", " ").replace("\r", " "),   # ← убираем переносы
            answer.replace("\n", " ").replace("\r", " "),      # ← убираем переносы
            " | ".join(sources),
        ])
# def save_dialog(user_id: int, username: str, question: str, answer: str, sources: list[str]):
#     """Дописывает одну запись в JSONL-файл."""
#     record = {
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#         "user_id": user_id,
#         "username": username,
#         "question": question,
#         "answer": answer,
#         "sources": sources,
#     }
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=60.0,
)

# --- Подключение к Chroma ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)
collection = chroma_client.get_collection(
    name="obitel_texts",
    embedding_function=embedding_fn,
)

SYSTEM_PROMPT = """Ты — помощник «Иисусовой Обители».
Тебе даны тексты из материалов Обители. Найди подходящий текст и выдай его пользователю.

Как отвечать:
- Выдавай текст из материалов дословно, без пересказа и без изменения формулировок.
- НЕ пиши «там же приводятся слова», «в источнике сказано», «фрагмент гласит», «материал сообщает», «в тексте говорится» и подобное. Просто сам текст.
- Не добавляй от себя вступлений вроде «Вот что говорится по этому вопросу». Сразу текст.
- В конце ответа укажи источник — URL из метаданных, откуда взят текст. Формат: «Источник: <URL>».
- Если ответ основан на нескольких материалах — перечисли все URL, каждый с новой строки в формате «Источник: <URL>».
- Если подходящего текста нет — напиши: «Извините, я не нашёл такой информации в источниках "Иисусовой Обители"» и предложи обратиться напрямую: в Telegram @iereiR или по контактам на сайте iisusova-obitel.ru.
- Не используй свои общие знания. Не придумывай факты и ссылки.
- Отвечай на русском языке."""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте! Я — помощник «Иисусовой Обители». "
        "Могу найти ответ в материалах сайта iisusova-obitel.ru, "
        "Telegram-канала @isusovam и канала в MAX «ИИСУСОВА ОБИТЕЛЬ»: "
        "об Иисусовой молитве, исихазме, толковании Евангелия, проповедях, фильмах "
        "и других публикациях Обители. Спросите — и я поищу."
    )

def find_relevant(query: str, top_k: int = 5):
    """Ищет в Chroma фрагменты, близкие по смыслу к вопросу."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(documents, metadatas, distances))

async def handle_ai_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user = update.effective_user
    logger.info(f"Сообщение от {user.id} ({user.username}): {user_text!r}")
    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        # 1. Поиск релевантных фрагментов в Chroma
        fragments = await asyncio.to_thread(find_relevant, user_text)

        # 2. Формируем контекст с URL
        source_urls = []
        if fragments:
            # context_parts = []
            # for doc, meta, dist in fragments:
            #     url = meta.get("url", "источник не указан")
            #     title = meta.get("title", "")
            #     context_parts.append(
            #         f"(источник: {url}, заголовок: {title}) ---\n{doc}"
            #     )
            # context_text = "\n\n".join(context_parts)
            context_parts = []
            for doc, meta, dist in fragments:
                url = meta.get("url", "")
                if url and url not in source_urls:
                    source_urls.append(url)
                context_parts.append(f"{doc}\n[URL: {url}]")
            context_text = "\n\n".join(context_parts)
        else:
            context_text = ""

        # if fragments:
        #     context_parts = []
        #     for doc, meta, dist in fragments:
        #         url = meta.get("url", "")
        #         context_parts.append(f"{doc}\n[URL: {url}]")
        #     context_text = "\n\n".join(context_parts)
        # else:
        #     context_text = ""

        user_message = (
            f"Материалы по теме:\n\n{context_text}\n\n"
            f"Вопрос: {user_text}\n\n"
            f"Выдай подходящие тексты дословно и укажи источники."
        )
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer = response.choices[0].message.content
        if not answer:
            answer = "Извините, не удалось сформировать ответ. Попробуйте переформулировать вопрос."
        logger.info(f"Ответ для {user.id}: {answer[:100]!r}...")
        # Сохраняем диалог
        save_dialog(
            user_id=user.id,
            username=user.username or user.first_name or "",
            question=user_text,
            answer=answer,
            sources=source_urls,
        )


        # 4. Telegram ограничивает сообщение 4096 символами — разбиваем при необходимости
        MAX_LEN = 4000
        for i in range(0, len(answer), MAX_LEN):
            await update.message.reply_text(answer[i:i+MAX_LEN])

    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения от {user.id}")
        await update.message.reply_text(f"Ошибка: {e}")

def main():
    app = ApplicationBuilder().token(os.getenv("BOT_TOKEN")).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()