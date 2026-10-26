import asyncio
import json
import logging
import os
import re
import html
from datetime import datetime
from typing import List, Dict, Optional

import httpx
import aiosqlite
import feedparser
import trafilatura
from langdetect import detect, LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic_settings import BaseSettings
from telegram import Bot

# ============ SETTINGS (через pydantic) ============

class Settings(BaseSettings):
    """Загружает настройки из переменных окружения."""
    TELEGRAM_BOT_TOKEN: str
    CHANNEL_USERNAME: str
    GROQ_API_KEY: str
    DEEPL_API_KEY: str
    ADMINS_CHAT_ID: str = ""
    DB_CACHE_DIR: str = "."
    
    rss_feeds: List[str] = [
        "https://ria.ru/export/rss2/archive/index.xml",
        "https://www.goodnewsnetwork.org/feed/",
        "https://positive.news/feed/", 
        "https://www.goodgoodgood.co/articles/rss.xml",
        "https://www.cnews.ru/inc/rss/news.xml",
        "https://www.techinsider.ru/out/public-feed-all-news.xml",
        "https://lenta.ru/rss",
    ]
    
    target_language: str = "RU"
    news_per_feed: int = 4
    min_score: int = 7
    summary_max_words: int = 120
    article_max_chars: int = 12000
    history_keep_days: int = 30
    
    allowed_hashtags: List[str] = [
        "люди", "животные", "питомцы", "растения",
        "культура", "наука", "экология", "технологии",
    ]
    
    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()
NEWS_HISTORY_DB = os.path.join(settings.DB_CACHE_DIR, "published_news.db")

# ============ LOGGING ============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
#                   DATABASE (async через aiosqlite)
# ============================================================

class Database:
    """Async wrapper для SQLite."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def init(self):
        """Создаёт таблицу опубликованных ссылок."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS published (
                    link         TEXT PRIMARY KEY,
                    title        TEXT,
                    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()
    
    async def is_already_published(self, link: str) -> bool:
        """Проверяет, публиковалась ли новость ранее."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT 1 FROM published WHERE link = ?', (link,))
            result = await cursor.fetchone()
            return result is not None
    
    async def mark_as_published(self, link: str, title: str):
        """Сохраняет новость в историю после успешной публикации."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    'INSERT OR IGNORE INTO published (link, title) VALUES (?, ?)',
                    (link, title)
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Не удалось сохранить в БД: {e}")
    
    async def cleanup_old_records(self, days: int = None):
        """Удаляет записи старше N дней."""
        days = days or settings.history_keep_days
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'DELETE FROM published WHERE published_at < datetime("now", ?)',
                (f'-{days} days',)
            )
            await db.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info(f"🗑  Удалено {deleted} старых записей из истории")


# ============================================================
#                     HTTP CLIENT (httpx)
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
)
async def fetch_url(client: httpx.AsyncClient, url: str, timeout: float = 30.0) -> str:
    """Загружает URL с ретраями."""
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
)
async def post_json(client: httpx.AsyncClient, url: str, headers: dict, data: dict, timeout: float = 60.0) -> dict:
    """POST запрос с JSON телом и ретраями."""
    resp = await client.post(url, headers=headers, json=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================================================
#                     RSS PARSING (async)
# ============================================================

async def fetch_single_feed(client: httpx.AsyncClient, feed_url: str) -> List[Dict]:
    """Парсит один RSS-фид и возвращает NEWS_PER_FEED последних записей."""
    try:
        text = await fetch_url(client, feed_url)
        feed = feedparser.parse(text)
        entries = []
        for entry in feed.entries[:settings.news_per_feed]:
            entries.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'source': feed.feed.get('title', feed_url),
            })
        logger.info(f"✓ {len(entries)} новостей из {feed_url}")
        return entries
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки {feed_url}: {e}")
        return []


async def fetch_rss_news() -> List[Dict]:
    """Параллельно загружает все RSS-ленты через httpx."""
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (GoodNewsBot/1.0)"}
    ) as client:
        tasks = [fetch_single_feed(client, url) for url in settings.rss_feeds]
        results = await asyncio.gather(*tasks)
    
    all_entries = [e for sublist in results for e in sublist]
    logger.info(f"📰 Всего загружено: {len(all_entries)} новостей")
    return all_entries


# ============================================================
#              FULL ARTICLE FETCHING (trafilatura)
# ============================================================

async def fetch_full_article(client: httpx.AsyncClient, news: Dict) -> Dict:
    """Скачивает страницу новости и извлекает полный текст через trafilatura."""
    result = news.copy()
    full_text = ""
    
    try:
        raw_html = await fetch_url(
            client, 
            result['link'],
            timeout=30.0
        )
        # trafilatura автоматически находит основной текст
        full_text = trafilatura.extract(
            raw_html,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        ) or ""
    except Exception as e:
        logger.warning(f"Не удалось скачать полную статью ({result['link'][:60]}): {e}")

    rss_clean = clean_html(result.get('summary', ''))
    if len(full_text) > max(len(rss_clean), 400):
        result['full_text'] = full_text
        logger.info(f"📄 Полная статья скопирована: {len(full_text)} симв.")
    else:
        result['full_text'] = rss_clean
        logger.info("📄 Полная статья недоступна — использую RSS-описание")
    
    return result


# ============================================================
#                  Groq AI (оценка / саммари / хэштеги)
# ============================================================

async def groq_chat(
    client: httpx.AsyncClient,
    prompt: str,
    model: str = "qwen/qwen3.8-27b",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False
) -> str:
    """Вызов Groq chat completion."""
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if json_mode:
        data["response_format"] = {"type": "json_object"}
    
    resp = await post_json(
        client,
        "https://api.groq.com/openai/v1/chat/completions",
        headers,
        data,
        timeout=60.0
    )
    
    content = resp['choices'][0]['message']['content'].strip()
    if not content:
        msg = resp['choices'][0]['message']
        content = (msg.get('reasoning_content') or '').strip()
    
    return content


async def score_news_batch(client: httpx.AsyncClient, news_list: List[Dict]) -> List[Dict]:
    """Оценивает массив новостей одним запросом (батчинг)."""
    if not news_list:
        return []
    
    # Формируем список заголовков с индексами
    titles_with_idx = [f"{i+1}. {n['title']}" for i, n in enumerate(news_list)]
    titles_text = "\n".join(titles_with_idx)
    
    prompt = (
        "Оцени каждую новость по шкале 1–10 (1 — очень негативная, 10 — очень позитивная).\n"
        "Верни СТРОГО валидный JSON массив с оценками.\n"
        'Формат: [{"index": 1, "score": 8}, {"index": 2, "score": 5}, ...]\n\n'
        f"Новости:\n{titles_text}"
    )
    
    try:
        raw = await groq_chat(
            client,
            prompt,
            model="llama-3.1-8b-instant",  # Легкая модель для скоринга
            temperature=0.1,
            max_tokens=512,
            json_mode=True
        )
        
        # Парсим JSON ответ
        scores = json.loads(raw)
        
        # Применяем оценки к новостям
        result = []
        for news in news_list:
            item = news.copy()
            score_data = next((s for s in scores if s.get('index') == len(result) + 1), None)
            item['score'] = score_data['score'] if score_data else 5
            result.append(item)
            
            emoji = "✓" if item['score'] >= settings.min_score else "✗"
            logger.info(f"{emoji} [{item['score']:2}/10] {news['title'][:55]}")
        
        return result
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Ошибка парсинга JSON от Groq: {e}")
        # Fallback: все новости с оценкой 5
        return [{'score': 5, **n} for n in news_list]
    except Exception as e:
        logger.error(f"Groq batch scoring error: {e}")
        return [{'score': 0, **n} for n in news_list]


async def filter_positive_news(client: httpx.AsyncClient, news_list: List[Dict]) -> List[Dict]:
    """Оценивает ВСЕ новости батчем, возвращает позитивные."""
    scored = await score_news_batch(client, news_list)
    positive = [n for n in scored if n.get('score', 0) >= settings.min_score]
    positive.sort(key=lambda x: x.get('score', 0), reverse=True)
    logger.info(f"🏆 Отобрано {len(positive)} позитивных из {len(news_list)}")
    return positive


async def summarize_and_tag_via_groq(client: httpx.AsyncClient, news: Dict) -> Dict:
    """Через Groq создаёт саммари и подбирает хэштеги (JSON ответ)."""
    result = news.copy()
    article = (news.get('full_text') or clean_html(news.get('summary', '')))[:settings.article_max_chars]
    tag_list = ", ".join(settings.allowed_hashtags)
    
    prompt = (
        "Ты — редактор канала добрых новостей. Прочитай новость и верни СТРОГО валидный JSON.\n\n"
        "Требования к саммари:\n"
        f"- Объём до {settings.summary_max_words} слов\n"
        "- Сохрани эмоциональную окраску (радость, тепло, воодушевление)\n"
        "- Включи ключевые факты: имена, числа, места, даты\n"
        "- Если есть яркие цитаты героев, включи 1-2 в кавычках\n"
        "- Используй подходящие эмоджи и абзацы\n\n"
        f"Хэштеги: выбери от 1 до 4 СТРОГО из списка [{tag_list}]. Не придумывай других.\n\n"
        'Формат ответа (JSON): {"summary": "текст саммари", "hashtags": ["#тег1", "#тег2"]}\n\n'
        f"ЗАГОЛОВОК: {clean_html(news['title'])}\n\n"
        f"ТЕКСТ НОВОСТИ:\n{article}"
    )
    
    try:
        raw = await groq_chat(
            client,
            prompt,
            model="qwen/qwen3.8-27b",  # Тяжелая модель для качественной генерации
            temperature=0.4,
            max_tokens=1024,
            json_mode=True
        )
        
        # Парсим JSON
        data = json.loads(raw)
        summary_part = data.get('summary', '').strip()
        tags_list = data.get('hashtags', [])
        
        if not summary_part:
            summary_part = clean_html(news.get('summary', ''))
        
        # Валидируем хэштеги
        allowed_lower = {t.lower(): t for t in settings.allowed_hashtags}
        hashtags = []
        for t in tags_list:
            if not t.startswith('#'):
                continue
            key = t[1:].lower()
            if key in allowed_lower and allowed_lower[key] not in [h[1:] for h in hashtags]:
                hashtags.append("#" + allowed_lower[key])
        
        if not hashtags:
            hashtags = ["#ДоброNews"]
        
        # Добавляем канальные теги
        for t in ["#хорошиеновости"]:
            if t not in hashtags:
                hashtags.append(t)
        
        result['ai_summary'] = summary_part
        result['hashtags'] = " ".join(hashtags)
        logger.info(f"📝 Саммари готово ({len(summary_part)} симв.), теги: {result['hashtags']}")
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Ошибка парсинга JSON от Groq: {e}")
        result['ai_summary'] = clean_html(news.get('summary', ''))
        result['hashtags'] = "#хорошиеновости"
    except Exception as e:
        logger.error(f"Ошибка генерации саммари через Groq: {e}")
        result['ai_summary'] = clean_html(news.get('summary', ''))
        result['hashtags'] = "#хорошиеновости"
    
    return result


# ============================================================
#                     LANGUAGE DETECTION (langdetect)
# ============================================================

def is_russian(text: str) -> bool:
    """Определяет, русский ли текст (локально, бесплатно)."""
    if not text or not text.strip():
        return False
    try:
        return detect(text) == 'ru'
    except LangDetectException:
        return False


# ============================================================
#                     DeepL TRANSLATION
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
)
async def translate_text(client: httpx.AsyncClient, text: str) -> str:
    """Переводит через DeepL."""
    if not text or not text.strip():
        return text
    
    headers = {"Authorization": f"DeepL-Auth-Key {settings.DEEPL_API_KEY}"}
    data = {"text": text, "target_lang": settings.target_language}
    
    try:
        resp = await client.post(
            "https://api-free.deepl.com/v2/translate",
            headers=headers,
            data=data,
            timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]
    except Exception as e:
        logger.warning(f"DeepL error, returning original: {e}")
        return text


async def maybe_translate(client: httpx.AsyncClient, text: str) -> str:
    """Если текст не на русском — переводит через DeepL."""
    if not text or not text.strip():
        return text
    
    if is_russian(text):
        logger.info("🇷🇺 Текст на русском — перевод не требуется")
        return text
    
    logger.info("🌍 Текст не на русском — перевожу через DeepL...")
    return await translate_text(client, text)


# ============================================================
#                     HTML HELPERS
# ============================================================

def clean_html(text: str) -> str:
    """Убирает теги и декодирует HTML-сущности."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def escape_html(text: str) -> str:
    """Экранирует символы для Telegram parse_mode=HTML."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))


def format_paragraphs(text: str) -> str:
    """Экранирует и нормализует абзацы."""
    escaped = escape_html(text)
    escaped = re.sub(r'\n{3,}', '\n\n', escaped)
    return escaped.strip()


async def format_news_for_telegram(client: httpx.AsyncClient, news: Dict) -> str:
    """Формирует финальное сообщение для Telegram."""
    title_raw = clean_html(news['title'])
    title = escape_html(await maybe_translate(client, title_raw))

    summary_raw = news.get('ai_summary') or clean_html(news.get('summary', ''))
    # Саммари тоже экранируем на случай спецсимволов
    summary = format_paragraphs(summary_raw)

    hashtags = escape_html(news.get('hashtags', '#хорошиеновости'))

    return (
        f"✨ <b>{title}</b>\n\n"
        f"{summary}\n\n"
        f"🔗 <a href=\"{news['link']}\">Читать новость полностью</a>\n\n"
        f"{hashtags}"
    )


# ============================================================
#                      TELEGRAM PUBLISHING
# ============================================================

async def send_to_admins(bot: Bot, message: str, link: str, title: str, db: Database) -> bool:
    """Рассылает готовый пост каждому админу в личку."""
    admin_ids_raw = settings.ADMINS_CHAT_ID
    admin_ids = [int(x.strip()) for x in admin_ids_raw.split(",") if x.strip()]

    if not admin_ids:
        logger.error("❌ ADMINS_CHAT_ID не задан — некому отправлять")
        return False

    ok_count = 0
    for uid in admin_ids:
        try:
            await bot.send_message(
                chat_id=uid,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=False,
            )
            ok_count += 1
            logger.info(f"📨 Отправлено админу {uid}")
        except Exception as e:
            logger.error(f"✗ Не удалось отправить админу {uid}: {e}")

    if ok_count:
        await db.mark_as_published(link, title)
        logger.info(f"✅ Разослано {ok_count}/{len(admin_ids)} админам: {title[:50]}")
        return True
    return False


# ============================================================
#                           MAIN
# ============================================================

async def main():
    logger.info("=" * 55)
    logger.info(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    # Инициализация БД
    db = Database(NEWS_HISTORY_DB)
    await db.init()
    await db.cleanup_old_records()

    # 1. RSS (параллельная загрузка через httpx)
    logger.info("📡 Загружаю RSS-ленты...")
    all_news = await fetch_rss_news()

    # 2. Отсекаем уже опубликованные (дедупликация)
    fresh_news = []
    for n in all_news:
        if not await db.is_already_published(n['link']):
            fresh_news.append(n)
    
    logger.info(f"🆕 Новых (не опубликованных): {len(fresh_news)}")

    if not fresh_news:
        logger.warning("❌ Все новости уже опубликованы — выход")
        return

    # 3. Оценка через Groq (батчинг) + сортировка
    logger.info("🤖 Оцениваю новости через Groq (batch)...")
    async with httpx.AsyncClient() as client:
        positive_news = await filter_positive_news(client, fresh_news)

    if not positive_news:
        logger.warning("❌ Нет позитивных новостей с оценкой ≥ 7")
        return

    # 4. Берём лучшую
    best = positive_news[0]
    logger.info(f"🥇 Лучшая новость (балл {best['score']}): {best['title']}")

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    ) as client:
        # 5. Копируем новость ЦЕЛИКОМ (trafilatura)
        logger.info("📄 Копирую полный текст новости...")
        best = await fetch_full_article(client, best)

        # 6. Groq: саммари + хэштеги (JSON)
        logger.info("🤖 Генерирую саммари и хэштеги через Groq...")
        best = await summarize_and_tag_via_groq(client, best)

        # 7. Форматирование для Telegram
        logger.info("🌍 Форматирую сообщение...")
        message = await format_news_for_telegram(client, best)

    # 8. Публикация
    logger.info("📤 Публикую в Telegram...")
    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    success = await send_to_admins(bot, message, best['link'], best['title'], db)
    
    logger.info("✅ Готово!" if success else "❌ Публикация не удалась")


if __name__ == "__main__":
    asyncio.run(main())