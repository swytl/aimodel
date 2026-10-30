import requests
import time
import json

base_url = "https://iisusova-obitel.ru/wp-json/wp/v2/posts"
all_posts = []
page = 1

while True:
    response = requests.get(f"{base_url}?per_page=100&page={page}")
    if response.status_code != 200:
        break
    posts = response.json()
    if not posts:
        break
    for post in posts:
        all_posts.append({
            "title": post['title']['rendered'],
            "content": post['content']['rendered'],
            "url": post['link']
        })
    print(f"Скачана страница {page}, всего статей: {len(all_posts)}")
    page += 1
    time.sleep(1)

# Сохраняем результат в файл, чтобы бот мог его использовать
with open("posts.json", "w", encoding="utf-8") as f:
    json.dump(all_posts, f, ensure_ascii=False, indent=2)

print(f"Готово. Всего собрано статей: {len(all_posts)}")
print("Сохранено в posts.json")