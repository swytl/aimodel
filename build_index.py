import json
import re
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

# --- 1. Загрузка сырых данных ---
with open("posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

print(f"Загружено статей: {len(posts)}")

# --- 2. Очистка HTML и нарезка на фрагменты ---
def clean_html(html: str) -> str:
    """Убирает теги и служебные кнопки, оставляя чистый текст."""
    soup = BeautifulSoup(html, "html.parser")

    # Удаляем элементы, содержащие «Показать больше» / «Показать меньше»
    for element in soup.find_all(["a", "span", "button", "div"]):
        if element.string and (
            "Показать больше" in element.string
            or "Показать меньше" in element.string
        ):
            element.decompose()

    # Теперь извлекаем чистый текст
    return soup.get_text(separator="\n", strip=True)

def chunk_text(text: str, max_chars: int = 900, overlap: int = 150):
    """
    Режет текст на фрагменты, стараясь не разрывать абзацы.
    max_chars: примерный размер фрагмента (800-1000)
    overlap: перекрытие между фрагментами для сохранения контекста
    """
    # Разбиваем на абзацы
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # Если добавление абзаца превысит лимит — сохраняем текущий фрагмент
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Начинаем новый фрагмент с перекрытием (берём конец предыдущего)
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + para
        else:
            current_chunk += " " + para

    # Не забываем последний фрагмент
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# --- 3. Подготовка Chroma ---
client = chromadb.PersistentClient(path="./chroma_db")

# Используем модель, которая хорошо понимает русский язык
# multilingual-e5-small — лёгкая и подходит для начала
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)

collection = client.get_or_create_collection(
    name="obitel_texts",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # косинусное расстояние для семантического поиска [citation:11]
)

# --- 4. Обработка и сохранение ---
all_chunks = []
all_metadatas = []
all_ids = []

for i, post in enumerate(posts):
    title = post.get("title", "")
    html_content = post.get("content", "")
    url = post.get("url", "")

    # Очищаем HTML
    clean_text = clean_html(html_content)

    # Добавляем заголовок в начало текста (важно для контекста)
    full_text = f"{title}\n\n{clean_text}"

    # Нарезаем
    chunks = chunk_text(full_text)

    for j, chunk in enumerate(chunks):
        chunk_id = f"post_{i}_chunk_{j}"
        all_ids.append(chunk_id)
        all_chunks.append(chunk)
        # Сохраняем URL и заголовок как метаданные — именно отсюда бот будет брать ссылку!
        all_metadatas.append({
            "url": url,
            "title": title,
            "source": "iisusova-obitel.ru"
        })

print(f"Всего фрагментов для индексации: {len(all_chunks)}")

# Добавляем в Chroma пакетами (чтобы не перегрузить память)
batch_size = 100
for i in range(0, len(all_chunks), batch_size):
    collection.add(
        ids=all_ids[i:i+batch_size],
        documents=all_chunks[i:i+batch_size],
        metadatas=all_metadatas[i:i+batch_size]
    )
    print(f"Добавлено фрагментов: {min(i+batch_size, len(all_chunks))}")

print("Готово! Индекс сохранён в ./chroma_db")