import feedparser
import requests
import asyncio
from telegram import Bot
import re
import html
import sqlite3
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]   # Secret: сгенерирован @BotFather
CHANNEL_USERNAME = os.environ.get("CHANNEL_USERNAME")  # Variable: username канала с @

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
DEEPL_API_KEY = os.environ["DEEPL_API_KEY"]

RSS_FEEDS = [
    "https://rss.app/feeds/TiEx0NJFPa0Wc5Y2.xml",
    "https://www.goodnewsnetwork.org/feed/",
    "https://positive.news/feed/",
]

TARGET_LANGUAGE = "RU"
NEWS_PER_FEED = 6         # Берём по 6 новости из каждого фида
MIN_SCORE = 7              # Минимальный балл для публикации

# В GitHub Actions временная файловая система не сохраняется между запусками,
# поэтому БД хранится в кэше actions/cache (см. workflow) и копируется локально.
DB_CACHE_DIR = os.environ.get("DB_CACHE_DIR", ".")
NEWS_HISTORY_DB = os.path.join(DB_CACHE_DIR, "published_news.db")
DB_CACHE_KEY = "published-news-db"
HISTORY_KEEP_DAYS = 30     # Сколько дней хранить историю

SUMMARY_MAX_WORDS = 120    # Ориентировочная длина саммари (в словах)
ARTICLE_MAX_CHARS = 12000  # Максимум текста статьи, отправляемого в Groq

# ============================================================
#           СТОП-СЛОВА И ПАТТЕРНЫ ФИЛЬТРАЦИИ
# ============================================================
# Полные слова — проверяются по границам слова (регистронезависимо).
# Паттерны со звёздочкой (*) — проверяются как часть слова:
#   "бомб*"   — блокирует слова, начинающиеся на "бомб" (бомба, бомбардировка...)
#   "*офобия" — блокирует слова, заканчивающиеся на "офобия" (русофобия, ксенофобия...)
#   "под*ем"  — блокирует слова с "под" ... "ем" в середине
# Проверка идёт по заголовку И тексту (summary + full_text).
BLOCKED_WORDS = [
    # --- Полные слова (точное совпадение по границам слова) ---
    "русофоб*",
    "войн*",
    "военн*",          # префикс — блокирует "военный", "военных", "военная"...
    "подполь*",
    "террор*",
    "стрельб*",
    "ЛГБТ",
    "квир*",
    "геи",
    "гей",
    "гея*", 
    "лесбиян*",
    "вторжен*",
    "транс*",
    "теракт*",
    "погиб*",          # блокирует "погиб", "погибли", "погибших"...
    "бомб*",          # бомба, бомбардировка, бомбоубежище...
    "ракет*",         # ракета, ракетный, ракету...
    "убий*",          # убийца, убийство, убийственный...
    "расстрел*",      # расстрел, расстрелян...
    "взрыв*",         # взрыв, взрывной, взрывчатка...
    "обстрел*",       # обстрел, обстрелян...
    "разруш*",        # разрушение, разрушен, разрушительный...
    "погиб*",         # погиб, погибли, погибших...
    "оккупа*",        # оккупация, оккупант, оккупирован...
    "геноцид*",       # геноцид, геноцида...
    "катастроф*",     # катастрофа, катастрофы, катастрофический...
    "трагеди*",       # трагедия, трагедии, трагический...
    "жертв*",         # жертва, жертвы, жертвами...
    "атак*",          # атака, атаки, атаковать...
    "смерт*",         # смерть, смерти, смертельный...
    "ранен*",         # ранен, ранены, ранение...
    "труп*",          # труп, трупы...
    "murd*",
    "attack*", 
    "victim*", 
    "weapon*",
    "shelling*",
    "LGBT",
    "queer*",
    "gay*",
    "lesbi*",
    "trans-*", 
    "transgender*", 
    "femboy*",
    "furr*",
    "genocid*",
    "corps",        # (англ.)
    "kill*",          # (англ.) kill, killed, killing...
    "bomb*",          # (англ.) bomb, bombing...
    "war",
    "warfare*", 
    "wartime*",
    "wars",         # (англ.) war, wars...
    "massacr*",      # (англ.)
    "terror*",        # (англ.) terror, terrorism...
    "shoot*",         # (англ.) shoot, shooting...
    "dead",          # (англ.) dead, deadly...
    "deadly", 
    "died", 
    "crash*",         # (англ.) crash, crashed...
    "explos*",     # (англ.)
    "invasion*",
]
# ===================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
#          ФИЛЬТРАЦИЯ ПО СТОП-СЛОВАМ / ПАТТЕРНАМ
# ============================================================

def _compile_blocked_patterns(words: list) -> list:
    """Компилирует список стоп-слов/паттернов в список regex-объектов.

    Правила:
    - Если паттерн содержит '*', звёздочка заменяется на '\\w*' (любые буквы/цифры).
    - Если паттерн НЕ содержит '*', он оборачивается в '\\b...\\b' (границы слова).
    - Всё регистронезависимо (re.IGNORECASE).
    """
    compiled = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if '*' in word:
            # Паттерн с подстановкой: заменяем * на \w* и ищем как подстроку в слове
            regex_str = re.escape(word).replace(r'\*', r'\w*')
            # Оборачиваем в границы слова, чтобы "бомб*" не ловило "обомбеть"
            # Но если * в начале — левая граница не нужна (ищем окончание слова)
            if word.startswith('*'):
                regex_str = regex_str + r'\b'
            elif word.endswith('*'):
                regex_str = r'\b' + regex_str
            else:
                regex_str = r'\b' + regex_str + r'\b'
        else:
            # Полное слово — ищем по границам слова
            regex_str = r'\b' + re.escape(word) + r'\b'

        try:
            compiled.append(re.compile(regex_str, re.IGNORECASE | re.UNICODE))
        except re.error as e:
            logger.warning(f"Невалидный паттерн '{word}': {e}")
    return compiled


# Компилируем паттерны один раз при старте
_BLOCKED_PATTERNS = _compile_blocked_patterns(BLOCKED_WORDS)


def is_blocked_by_words(news: dict) -> tuple:
    """Проверяет заголовок и текст новости на наличие стоп-слов/паттернов.

    Возвращает (is_blocked: bool, matched_word: str or None).
    Если найдено совпадение — возвращается первое найденное слово/паттерн.
    """
    # Собираем весь проверяемый текст: заголовок + summary + full_text
    title = clean_html(news.get('title', ''))
    summary = clean_html(news.get('summary', ''))
    full_text = news.get('full_text', '')
    if full_text:
        full_text = clean_html(full_text)

    # Объединяем для проверки (ограничиваем длину, чтобы regex не тормозил)
    text_to_check = f"{title} {summary} {full_text}"

    for pattern in _BLOCKED_PATTERNS:
        match = pattern.search(text_to_check)
        if match:
            return True, match.group(0)

    return False, None


def filter_blocked_news(news_list: list) -> list:
    """Фильтрует список новостей, удаляя те, что содержат стоп-слова.
    Логирует причину блокировки для каждой отклонённой новости."""
    result = []
    blocked_count = 0
    for news in news_list:
        is_blocked, matched = is_blocked_by_words(news)
        if is_blocked:
            blocked_count += 1
            logger.info(f"🚫 Заблокировано (найдено: «{matched}»): {news['title'][:55]}")
        else:
            result.append(news)

    logger.info(f"🔍 Фильтр стоп-слов: пройдено {len(result)}, "
                f"заблокировано {blocked_count} из {len(news_list)}")
    return result


# ============================================================
#                   DATABASE (дедупликация)
# ============================================================

def init_db():
    """Создаёт таблицу опубликованных ссылок, если её нет."""
    os.makedirs(os.path.dirname(NEWS_HISTORY_DB) or ".", exist_ok=True)
    conn = sqlite3.connect(NEWS_HISTORY_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS published (
            link         TEXT PRIMARY KEY,
            title        TEXT,
            published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def is_already_published(link: str) -> bool:
    """Проверяет, публиковалась ли новость ранее."""
    conn = sqlite3.connect(NEWS_HISTORY_DB)
    cur = conn.execute('SELECT 1 FROM published WHERE link = ?', (link,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def mark_as_published(link: str, title: str):
    """Сохраняет новость в историю после успешной публикации."""
    try:
        conn = sqlite3.connect(NEWS_HISTORY_DB)
        conn.execute('INSERT OR IGNORE INTO published (link, title) VALUES (?, ?)',
                     (link, title))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Не удалось сохранить в БД: {e}")


def cleanup_old_records(days: int = HISTORY_KEEP_DAYS):
    """Удаляет записи старше N дней, чтобы БД не разрасталась."""
    conn = sqlite3.connect(NEWS_HISTORY_DB)
    cur = conn.execute(
        'DELETE FROM published WHERE published_at < datetime("now", ?)',
        (f'-{days} days',)
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted:
        logger.info(f"🗑  Удалено {deleted} старых записей из истории")


# ============================================================
#                     RSS PARSING
# ============================================================

def fetch_single_feed(feed_url: str) -> list:
    """Парсит один RSS-фид и возвращает NEWS_PER_FEED последних записей."""
    try:
        feed = feedparser.parse(feed_url, request_headers={
            "User-Agent": "Mozilla/5.0 (GoodNewsBot/1.0)"
        })
        entries = []
        for entry in feed.entries[:NEWS_PER_FEED]:
            entries.append({
                'title':     entry.get('title', ''),
                'summary':   entry.get('summary', ''),
                'link':      entry.get('link', ''),
                'published': entry.get('published', ''),
                'source':    feed.feed.get('title', feed_url),
            })
        logger.info(f"✓ {len(entries)} новостей из {feed_url}")
        return entries
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки {feed_url}: {e}")
        return []


def fetch_rss_news() -> list:
    """Параллельно загружает все RSS-ленты."""
    with ThreadPoolExecutor(max_workers=len(RSS_FEEDS)) as pool:
        results = pool.map(fetch_single_feed, RSS_FEEDS)
    all_entries = [e for sublist in results for e in sublist]
    logger.info(f"📰 Всего загружено: {len(all_entries)} новостей")
    return all_entries


# ============================================================
#              FULL ARTICLE FETCHING (копируем новость целиком)
# ============================================================

def extract_article_from_html(html_text: str) -> str:
    """Эвристически извлекает основной текст статьи из HTML страницы."""
    # Пробуем распространённые контейнеры полных статей
    container_patterns = [
        r'<div[^>]+class="[^"]*(?:entry-content|article-content|post-content|story-body|article-body|text-content)[^"]*"[^>]*>(.*?)</div>',
        r'<article[^>]*>(.*?)</article>',
    ]
    best = ""
    for pattern in container_patterns:
        for m in re.finditer(pattern, html_text, re.IGNORECASE | re.DOTALL):
            candidate = m.group(1)
            if len(candidate) > len(best):
                best = candidate
    if not best:
        # Fallback: берём все <p> со страницы
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_text,
                                re.IGNORECASE | re.DOTALL)
        best = '\n'.join(paragraphs)

    # Удаляем скрипты/стили/ссылки-теги внутри извлечённого куска
    best = re.sub(r'<(?:script|style|noscript|form|iframe|svg)[^>]*>.*?</(?:script|style|noscript|form|iframe|svg)>',
                  '', best, flags=re.IGNORECASE | re.DOTALL)
    best = re.sub(r'<br\s*/?>', '\n', best, flags=re.IGNORECASE)
    best = re.sub(r'</p>', '\n\n', best, flags=re.IGNORECASE)
    best = re.sub(r'<[^>]+>', ' ', best)
    best = html.unescape(best)
    # Схлопываем лишние пробелы, но сохраняем абзацы
    best = re.sub(r'[ \t]+', ' ', best)
    best = re.sub(r'\n\s*\n+', '\n\n', best)
    return best.strip()


def fetch_full_article(news: dict) -> dict:
    """Скачивает страницу новости и копирует её полный текст.
    Если полная страница недоступна — оставляет RSS-описание."""
    result = news.copy()
    full_text = ""
    try:
        resp = requests.get(result['link'], headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }, timeout=30)
        resp.raise_for_status()
        raw_html = resp.text
        # JSON-LD (часто содержит full text у AMP/WordPress)
        full_text = extract_article_from_html(raw_html)
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

def groq_chat(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Низкоуровневый вызов Groq chat completion, возвращает текст ответа."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen/qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content'].strip()
    # Некоторые модели (например qwen с reasoning) могут вернуть пустой content —
    # пробуем достать reasoning_content как запасной вариант.
    if not content:
        msg = resp.json()['choices'][0]['message']
        content = (msg.get('reasoning_content') or '').strip()
    return content


def extract_score(text: str) -> int:
    """Надёжно извлекает балл 1..10 из любого ответа модели."""
    if not text:
        return 0
    # Сначала ищем изолированное число 1-10
    m = re.search(r'\b(10|[1-9])\b', text)
    if m:
        return int(m.group(1))
    # Fallback: любое число, ограничиваем диапазоном
    m = re.search(r'\d+', text)
    if m:
        return max(1, min(10, int(m.group(0))))
    return 5


def score_news_via_groq(news: dict) -> dict:
    """Отправляет новость в Groq, возвращает её копию с полем 'score'."""
    result = news.copy()
    try:
        prompt = (
            "Оцени новость по шкале 1–10 (1 — очень негативная, 10 — очень позитивная).\n"
            "Ответь ОДНИМ числом от 1 до 10, без пояснений.\n\n"
            f"Заголовок: {news['title']}\n"
            f"Текст: {news['summary'][:500]}"
        )
        raw = groq_chat(prompt, temperature=0.1, max_tokens=8)
        result['score'] = extract_score(raw)
        emoji = "✓" if result['score'] >= MIN_SCORE else "✗"
        logger.info(f"{emoji} [{result['score']:2}/10] {news['title'][:55]}")
    except Exception as e:
        logger.error(f"Groq error: {e} | {news['title'][:40]}")
        result['score'] = 0
    return result


def filter_positive_news(news_list: list) -> list:
    """Оценивает ВСЕ новости, возвращает позитивные, отсортированные по баллу (убывание)."""
    # Параллельная оценка через Groq — сильно ускоряет работу
    with ThreadPoolExecutor(max_workers=5) as pool:
        scored = list(pool.map(score_news_via_groq, news_list))

    positive = [n for n in scored if n.get('score', 0) >= MIN_SCORE]

    # Сортируем по баллу (убывание). sort() в Python — стабильная,
    # поэтому при равных баллах сохранится исходный порядок (первая по порядку и будет выбрана).
    positive.sort(key=lambda x: x.get('score', 0), reverse=True)

    logger.info(f"🏆 Отобрано {len(positive)} позитивных из {len(news_list)}")
    return positive


ALLOWED_HASHTAGS = [
    "люди", "животные", "питомцы", "растения",
    "культура", "наука", "экология", "технологии",
]


def summarize_and_tag_via_groq(news: dict) -> dict:
    """Через Groq создаёт саммари полностью скопированной новости
    (с сохранением эмоциональной окраски, цитат и фактов) и подбирает хэштеги."""
    result = news.copy()
    article = (news.get('full_text') or clean_html(news.get('summary', '')))[:ARTICLE_MAX_CHARS]
    tag_list = ", ".join(ALLOWED_HASHTAGS)
    prompt = (
        "Ты — редактор канала добрых новостей. Прочитай новость целиком и сделай на русском языке:\n"
        f"1. Саммари объёмом до {SUMMARY_MAX_WORDS} слов. Обязательно сохрани:\n"
        "   - эмоциональную окраску оригинала (радость, тепло, воодушевление и т.д.);\n"
        "   - ключевые факты: имена, числа, места, даты;\n"
        "   - одну-две самые яркие цитаты героев новости, оформи их в кавычках. Оформи текст подходящими эмоджи и используй абзацы в тексте.\n"
        "2. Подбери от 1 до 4 хэштегов СТРОГО из этого списка: "
        f"[{tag_list}]. Не придумывай других хэштегов.\n\n"
        "Формат ответа (ровно два блока):\n"
        "САММАРИ:\n<текст саммари>\n"
        "ХЭШТЕГИ:\n#хэштег1 #хэштег2 ...\n\n"
        f"ЗАГОЛОВОК: {clean_html(news['title'])}\n\n"
        f"ТЕКСТ НОВОСТИ:\n{article}"
    )
    try:
        raw = groq_chat(prompt, temperature=0.4, max_tokens=1024)

        # Разбираем ответ: саммари и хэштеги
        summary_part = ""
        tags_part = ""
        m = re.search(r'САММАРИ\s*:?\s*(.*?)(?:ХЭШТЕГИ?\s*:?|$)', raw,
                      re.IGNORECASE | re.DOTALL)
        if m:
            summary_part = m.group(1).strip()
        m2 = re.search(r'ХЭШТЕГИ?\s*:?\s*(.*)$', raw, re.IGNORECASE | re.DOTALL)
        if m2:
            tags_part = m2.group(1).strip()

        if not summary_part:  # fallback: весь ответ считаем саммари
            summary_part = raw.strip()

        # Извлекаем хэштеги и оставляем только разрешённые
        found_tags = re.findall(r'#\w+', tags_part)
        allowed_lower = {t.lower(): t for t in ALLOWED_HASHTAGS}
        hashtags = []
        for t in found_tags:
            key = t[1:].lower()
            if key in allowed_lower and allowed_lower[key] not in hashtags:
                hashtags.append("#" + allowed_lower[key])
        # Если модель не выдала ни одного валидного тега — дефолтный
        if not hashtags:
            hashtags = ["#ДоброNews"]
        # Канальные теги добавляем без дублей
        for t in ["#хорошиеновости"]:
            if t not in hashtags:
                hashtags.append(t)

        result['ai_summary'] = summary_part
        result['hashtags'] = " ".join(hashtags)
        logger.info(f"📝 Саммари готово ({len(summary_part)} симв.), "
                    f"теги: {result['hashtags']}")
    except Exception as e:
        logger.error(f"Ошибка генерации саммари через Groq: {e}")
        # Fallback — обычное RSS-описание
        result['ai_summary'] = clean_html(news.get('summary', ''))
        result['hashtags'] = "#хорошиеновости"
    return result


# ============================================================
#                     DeepL TRANSLATION
# ============================================================

def detect_language(text: str) -> str:
    """Определяет язык текста через DeepL (без перевода). Возвращает 'ru', 'en' или None."""
    if not text or not text.strip():
        return None # type: ignore
    try:
        resp = requests.post(
            "https://api-free.deepl.com/v2/translate",
             headers={
                "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
            },
            data={"text": text, "target_lang": TARGET_LANGUAGE},
            timeout=30,
        )
        resp.raise_for_status()
        detected = resp.json()["translations"][0].get("detected_source_language", "").lower()
        return detected or None
    except Exception as e:
        logger.warning(f"DeepL language detection failed: {e}")
        return None


def translate_text(text: str, source_lang: str = None) -> str:
    """Переводит через DeepL. При ошибке возвращает оригинал (не падает)."""
    if not text or not text.strip():
        return text
    try:
        resp = requests.post(
            "https://api-free.deepl.com/v2/translate",
             headers={
                "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
            },
            data={"text": text, "target_lang": TARGET_LANGUAGE},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]
    except Exception as e:
        logger.warning(f"DeepL fallback (оригинал): {e}")
        return text


def maybe_translate(text: str) -> str:
    """Если текст на английском — переводит в DeepL, если на русском — оставляет как есть."""
    if not text or not text.strip():
        return text
    lang = detect_language(text)
    if lang == "EN":
        logger.info("🌍 Текст на английском — перевожу через DeepL...")
        return translate_text(text, source_lang="EN")
    if lang == "RU":
        logger.info("🇷🇺 Текст на русском — перевод не требуется")
        return text
    # Язык неизвестен или другой — пробуем перевести автоматически
    logger.info(f"Язык текста: {lang or 'не определён'} — пробую автоперевод")
    return translate_text(text)


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
    """Экранирует и сохраняет абзацы (Telegram HTML игнорирует одиночные переносы только
    внутри блоков, поэтому просто нормализуем пустые строки)."""
    escaped = escape_html(text)
    escaped = re.sub(r'\n{3,}', '\n\n', escaped)
    return escaped.strip()


def format_news_for_telegram(news: dict) -> str:
    """Формирует финальное сообщение для Telegram:
    заголовок (при необходимости переведённый) + саммари от Groq + ссылка + хэштеги."""
    title_raw = clean_html(news['title'])
    title = escape_html(maybe_translate(title_raw))

    summary_raw = news.get('ai_summary') or clean_html(news.get('summary', ''))
    # Саммари генерируется Groq на русском, но проверяем язык на всякий случай
    summary = format_paragraphs(summary_raw)

    hashtags = news.get('hashtags', '#хорошиеновости')

    return (
        f"✨ <b>{title}</b>\n\n"
        f"{summary}\n\n"
        f"🔗 <a href=\"{news['link']}\">Читать новость полностью</a>\n\n"
        f"{hashtags}"
    )


# ============================================================
#                      TELEGRAM PUBLISHING
# ============================================================

async def publish_to_telegram(message: str, link: str, title: str) -> bool:
    """Отправляет сообщение в канал. Возвращает True при успехе."""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=CHANNEL_USERNAME,
            text=message,
            parse_mode='HTML',
            disable_web_page_preview=False,
        )
        mark_as_published(link, title)
        logger.info(f"📤 Опубликовано в {CHANNEL_USERNAME}")
        return True
    except Exception as e:
        logger.error(f"Ошибка публикации Telegram: {e}")
        return False


# ============================================================
#                           MAIN
# ============================================================

async def main():
    logger.info("=" * 55)
    logger.info(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    init_db()
    cleanup_old_records()

    # 1. RSS
    logger.info("📡 Загружаю RSS-ленты...")
    all_news = await asyncio.to_thread(fetch_rss_news)

    # 2. Отсекаем уже опубликованные (дедупликация)
    fresh_news = [n for n in all_news if not is_already_published(n['link'])]
    logger.info(f"🆕 Новых (не опубликованных): {len(fresh_news)}")

    if not fresh_news:
        logger.warning("❌ Все новости уже опубликованы — выход")
        return

    # 2.5 ФИЛЬТРАЦИЯ ПО СТОП-СЛОВАМ (до Groq, чтобы не тратить API-вызовы)
    logger.info("🔍 Фильтрую по стоп-словам...")
    filtered_news = await asyncio.to_thread(filter_blocked_news, fresh_news)

    if not filtered_news:
        logger.warning("❌ Все новости заблокированы стоп-словами — выход")
        return

    # 3. Оценка через Groq + сортировка
    logger.info("🤖 Оцениваю новости через Groq...")
    positive_news = await asyncio.to_thread(filter_positive_news, filtered_news)

    if not positive_news:
        logger.warning("❌ Нет позитивных новостей с оценкой ≥ 7")
        return

    # 4. Берём лучшую: максимальный балл, при равенстве — первая по порядку
    best = positive_news[0]
    logger.info(f"🥇 Лучшая новость (балл {best['score']}): {best['title']}")

    # 5. Копируем новость ЦЕЛИКОМ (полный текст статьи со страницы)
    logger.info("📄 Копирую полный текст новости...")
    best = await asyncio.to_thread(fetch_full_article, best)

    # 5.5 Повторная проверка стоп-слов по полному тексту статьи
    # (RSS-описание могло не содержать запрещённое слово, а полный текст — содержит)
    is_blocked, matched = is_blocked_by_words(best)
    if is_blocked:
        logger.warning(f"🚫 Полная статья заблокирована (найдено: «{matched}»): {best['title'][:55]}")
        # Берём следующую лучшую новость
        for candidate in positive_news[1:]:
            candidate = await asyncio.to_thread(fetch_full_article, candidate)
            is_blocked2, matched2 = is_blocked_by_words(candidate)
            if not is_blocked2:
                best = candidate
                logger.info(f"🔄 Следующая новость: {best['title'][:55]}")
                break
            else:
                logger.warning(f"🚫 Тоже заблокирована (найдено: «{matched2}»): {candidate['title'][:55]}")
        else:
            logger.warning("❌ Все кандидаты заблокированы стоп-словами — выход")
            return

    # 6. Groq: саммари полного текста (эмоции, цитаты, факты) + хэштеги
    logger.info("🤖 Генерирую саммари и хэштеги через Groq...")
    best = await asyncio.to_thread(summarize_and_tag_via_groq, best)

    # 7. Проверка языка (EN → перевод в DeepL, RU → без перевода) и публикация
    logger.info("🌍 Проверяю язык текста...")
    message = await asyncio.to_thread(format_news_for_telegram, best)

    logger.info("📤 Публикую в Telegram...")
    success = await publish_to_telegram(message, best['link'], best['title'])

    logger.info("✅ Готово!" if success else "❌ Публикация не удалась")


if __name__ == "__main__":
    asyncio.run(main())
