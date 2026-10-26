import asyncio
import json
import logging
import os
import re
import html
from datetime import datetime
from typing import List, Dict

import httpx
import aiosqlite
import feedparser
import trafilatura
from langdetect import detect, LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, reraise
from pydantic_settings import BaseSettings
from telegram import Bot

# ============ SETTINGS ============

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    CHANNEL_USERNAME: str
    GROQ_API_KEY: str
    DEEPL_API_KEY: str
    ADMINS_CHAT_ID: str = ""
    DB_CACHE_DIR: str = "."

    # Модели можно переопределить через Secrets, не трогая код
    groq_scoring_model: str = "openai/gpt-oss-20b"     # дешёвая для скоринга
    groq_summary_model: str = "openai/gpt-oss-120b"    # качественная для саммари

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
        case_sensitive = False


settings = Settings()
NEWS_HISTORY_DB = os.path.join(settings.db_cache_dir, "published_news.db")

# Цепочки фолбэка: если модель отключат (404) — автоматически уйдём на следующую.
def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

GROQ_SCORING_CHAIN = _dedupe([settings.groq_scoring_model, "openai/gpt-oss-20b", "openai/gpt-oss-120b"])
GROQ_SUMMARY_CHAIN = _dedupe([settings.groq_summary_model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
#                   DATABASE (aiosqlite)
# ============================================================

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init(self):
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
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT 1 FROM published WHERE link = ?', (link,))
            return (await cursor.fetchone()) is not None

    async def mark_as_published(self, link: str, title: str):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('INSERT OR IGNORE INTO published (link, title) VALUES (?, ?)', (link, title))
                await db.commit()
        except Exception as e:
            logger.error(f"Не удалось сохранить в БД: {e}")

    async def cleanup_old_records(self, days: int = None):
        days = days or settings.history_keep_days
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'DELETE FROM published WHERE published_at < datetime("now", ?)',
                (f'-{days} days',)
            )
            await db.commit()
            if cursor.rowcount:
                logger.info(f"🗑  Удалено {cursor.rowcount} старых записей из истории")


# ============================================================
#                     HTTP (httpx + умные ретраи)
# ============================================================

def _is_transient_error(exc: BaseException) -> bool:
    """Ретраим ТОЛЬКО временные ошибки. 400/401/404 падают сразу."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
async def fetch_url(client: httpx.AsyncClient, url: str, timeout: float = 30.0) -> str:
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
async def post_json(client: httpx.AsyncClient, url: str, headers: dict, data: dict, timeout: float = 60.0) -> dict:
    resp = await client.post(url, headers=headers, json=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================================================
#                     RSS PARSING
# ============================================================

async def fetch_single_feed(client: httpx.AsyncClient, feed_url: str) -> List[Dict]:
    try:
        text = await fetch_url(client, feed_url)
        feed = feedparser.parse(text)
        entries = [{
            'title': e.get('title', ''),
            'summary': e.get('summary', ''),
            'link': e.get('link', ''),
            'published': e.get('published', ''),
            'source': feed.feed.get('title', feed_url),
        } for e in feed.entries[:settings.news_per_feed]]
        logger.info(f"✓ {len(entries)} новостей из {feed_url}")
        return entries
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки {feed_url}: {e}")
        return []


async def fetch_rss_news() -> List[Dict]:
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (GoodNewsBot/1.0)"}) as client:
        results = await asyncio.gather(*[fetch_single_feed(client, u) for u in settings.rss_feeds])
    all_entries = [e for sub in results for e in sub]
    logger.info(f"📰 Всего загружено: {len(all_entries)} новостей")
    return all_entries


# ============================================================
#              FULL ARTICLE FETCHING (trafilatura)
# ============================================================

async def fetch_full_article(client: httpx.AsyncClient, news: Dict) -> Dict:
    result = news.copy()
    full_text = ""
    try:
        raw_html = await fetch_url(client, result['link'], timeout=30.0)
        full_text = trafilatura.extract(
            raw_html, include_comments=False, include_tables=False, no_fallback=False
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
#                  Groq AI (chain-fallback + JSON)
# ============================================================

async def groq_chat(
    client: httpx.AsyncClient,
    prompt: str,
    models: List[str],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: str = None,
) -> str:
    """Вызов Groq с цепочкой фолбэка моделей и логом тела ошибки."""
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = None
    for model in models:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            data["response_format"] = {"type": "json_object"}
        # reasoning_effort поддерживается только GPT-OSS — экономит токены на "размышления"
        if reasoning_effort and model.startswith("openai/gpt-oss"):
            data["reasoning_effort"] = reasoning_effort

        try:
            resp = await post_json(client, "https://api.groq.com/openai/v1/chat/completions", headers, data)
            content = resp['choices'][0]['message']['content'].strip()
            if not content:
                content = (resp['choices'][0]['message'].get('reasoning_content') or '').strip()
            return content
        except httpx.HTTPStatusError as e:
            # Логируем тело ответа — сразу видно реальную причину от Groq
            logger.error(f"Groq HTTP {e.response.status_code} | model={model} | {e.response.text[:300]}")
            last_error = e
            if e.response.status_code in (400, 404):
                continue  # модель отключена / параметр не поддержан → следующая в цепочке
            raise
    raise last_error or RuntimeError("Groq: все модели из цепочки недоступны")


def _strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _parse_scores(raw: str, count: int) -> List[int]:
    """Надёжно достаёт оценки: JSON-массив объектов, массив чисел или числа из текста."""
    try:
        data = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError:
        nums = [int(x) for x in re.findall(r'\b(10|[1-9])\b', raw)]
        return [nums[i] if i < len(nums) else 5 for i in range(count)]

    if isinstance(data, dict):
        data = data.get("scores") or data.get("results") or data.get("ratings") or []

    scores = [5] * count
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                idx, val = item.get("index", i + 1), item.get("score", item.get("rating"))
            else:
                idx, val = i + 1, item
            try:
                idx, val = int(idx), int(val)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= count:
                scores[idx - 1] = max(1, min(10, val))
    return scores


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


async def _score_chunk(client: httpx.AsyncClient, chunk: List[Dict]) -> List[int]:
    titles = "\n".join(f"{i+1}. {n['title']}" for i, n in enumerate(chunk))
    prompt = (
        "Оцени каждую новость по шкале 1–10 (1 — очень негативная, 10 — очень позитивная).\n"
        "Верни СТРОГО JSON-массив без пояснений.\n"
        'Формат: [{"index": 1, "score": 8}, {"index": 2, "score": 5}]\n\n'
        f"Новости:\n{titles}"
    )
    raw = await groq_chat(
        client, prompt, GROQ_SCORING_CHAIN,
        temperature=0.1, max_tokens=1024,
        json_mode=True, reasoning_effort="low",
    )
    return _parse_scores(raw, len(chunk))


async def score_news_batch(client: httpx.AsyncClient, news_list: List[Dict]) -> List[Dict]:
    """Батчинг: чанки по 10 новостей, параллельно, с фолбэком."""
    if not news_list:
        return []
    chunks = list(_chunks(news_list, 10))
    results = await asyncio.gather(
        *[_score_chunk(client, c) for c in chunks],
        return_exceptions=True,
    )
    scored = []
    for chunk, res in zip(chunks, results):
        if isinstance(res, BaseException):
            logger.error(f"Groq scoring failed for chunk: {res}")
            scores = [0] * len(chunk)
        else:
            scores = res
        for news, sc in zip(chunk, scores):
            item = news.copy()
            item['score'] = sc
            scored.append(item)
            emoji = "✓" if sc >= settings.min_score else "✗"
            logger.info(f"{emoji} [{sc:2}/10] {news['title'][:55]}")
    return scored


async def filter_positive_news(client: httpx.AsyncClient, news_list: List[Dict]) -> List[Dict]:
    scored = await score_news_batch(client, news_list)
    positive = [n for n in scored if n.get('score', 0) >= settings.min_score]
    positive.sort(key=lambda x: x.get('score', 0), reverse=True)
    logger.info(f"🏆 Отобрано {len(positive)} позитивных из {len(news_list)}")
    return positive


async def summarize_and_tag_via_groq(client: httpx.AsyncClient, news: Dict) -> Dict:
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
        'Формат ответа: {"summary": "текст саммари", "hashtags": ["#тег1", "#тег2"]}\n\n'
        f"ЗАГОЛОВОК: {clean_html(news['title'])}\n\n"
        f"ТЕКСТ НОВОСТИ:\n{article}"
    )

    try:
        raw = await groq_chat(
            client, prompt, GROQ_SUMMARY_CHAIN,
            temperature=0.4, max_tokens=1024, json_mode=True,
        )
        data = json.loads(_strip_code_fences(raw))
        summary_part = data.get('summary', '').strip()
        tags_list = data.get('hashtags', [])

        if not summary_part:
            summary_part = clean_html(news.get('summary', ''))

        allowed_lower = {t.lower(): t for t in settings.allowed_hashtags}
        hashtags = []
        for t in tags_list:
            if not isinstance(t, str) or not t.startswith('#'):
                continue
            key = t[1:].lower()
            if key in allowed_lower and allowed_lower[key] not in [h[1:] for h in hashtags]:
                hashtags.append("#" + allowed_lower[key])
        if not hashtags:
            hashtags = ["#ДоброNews"]
        if "#хорошиеновости" not in hashtags:
            hashtags.append("#хорошиеновости")

        result['ai_summary'] = summary_part
        result['hashtags'] = " ".join(hashtags)
        logger.info(f"📝 Саммари готово ({len(summary_part)} симв.), теги: {result['hashtags']}")

    except Exception as e:
        logger.error(f"Ошибка генерации саммари через Groq: {e}")
        result['ai_summary'] = clean_html(news.get('summary', ''))
        result['hashtags'] = "#хорошиеновости"
    return result


# ============================================================
#                     LANGUAGE + DeepL
# ============================================================

def is_russian(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        return detect(text) == 'ru'
    except LangDetectException:
        return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
async def translate_text(client: httpx.AsyncClient, text: str) -> str:
    if not text or not text.strip():
        return text
    headers = {"Authorization": f"DeepL-Auth-Key {settings.DEEPL_API_KEY}"}
    data = {"text": text, "target_lang": settings.target_language}
    try:
        resp = await client.post("https://api-free.deepl.com/v2/translate", headers=headers, data=data, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]
    except Exception as e:
        logger.warning(f"DeepL error, returning original: {e}")
        return text


async def maybe_translate(client: httpx.AsyncClient, text: str) -> str:
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
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def escape_html(text: str) -> str:
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def format_paragraphs(text: str) -> str:
    escaped = escape_html(text)
    return re.sub(r'\n{3,}', '\n\n', escaped).strip()


async def format_news_for_telegram(client: httpx.AsyncClient, news: Dict) -> str:
    title_raw = clean_html(news['title'])
    title = escape_html(await maybe_translate(client, title_raw))
    summary_raw = news.get('ai_summary') or clean_html(news.get('summary', ''))
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
    admin_ids = [int(x.strip()) for x in settings.admins_chat_id.split(",") if x.strip()]
    if not admin_ids:
        logger.error("❌ ADMINS_CHAT_ID не задан — некому отправлять")
        return False

    ok_count = 0
    for uid in admin_ids:
        try:
            await bot.send_message(chat_id=uid, text=message, parse_mode='HTML', disable_web_page_preview=False)
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

    db = Database(NEWS_HISTORY_DB)
    await db.init()
    await db.cleanup_old_records()

    logger.info("📡 Загружаю RSS-ленты...")
    all_news = await fetch_rss_news()

    fresh_news = [n for n in all_news if not await db.is_already_published(n['link'])]
    logger.info(f"🆕 Новых (не опубликованных): {len(fresh_news)}")
    if not fresh_news:
        logger.warning("❌ Все новости уже опубликованы — выход")
        return

    logger.info("🤖 Оцениваю новости через Groq (batch)...")
    async with httpx.AsyncClient() as client:
        positive_news = await filter_positive_news(client, fresh_news)
    if not positive_news:
        logger.warning(f"❌ Нет позитивных новостей с оценкой ≥ {settings.min_score}")
        return

    best = positive_news[0]
    logger.info(f"🥇 Лучшая новость (балл {best['score']}): {best['title']}")

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    ) as client:
        logger.info("📄 Копирую полный текст новости...")
        best = await fetch_full_article(client, best)

        logger.info("🤖 Генерирую саммари и хэштеги через Groq...")
        best = await summarize_and_tag_via_groq(client, best)

        logger.info("🌍 Форматирую сообщение...")
        message = await format_news_for_telegram(client, best)

    logger.info("📤 Публикую в Telegram...")
    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    success = await send_to_admins(bot, message, best['link'], best['title'], db)
    logger.info("✅ Готово!" if success else "❌ Публикация не удалась")


if __name__ == "__main__":
    asyncio.run(main())