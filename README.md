# TrendWatching Phase 2

TrendWatching Phase 2 - это продвинутая система анализа трендов, использующая различные модели искусственного интеллекта для обработки и анализа текстовых данных. Система поддерживает работу с несколькими LLM-провайдерами (DeepSeek, OpenAI, Gemini) и предоставляет гибкую конфигурацию для каждого из них.

## Основные возможности

- Анализ трендов с использованием различных LLM-моделей
- Извлечение ключевых слов и фраз из текста
- Векторное хранение и поиск данных
- Обработка и анализ текстовых данных
- Интеграция с Telegram ботом
- Поддержка различных источников данных (CSV, веб-страницы)
- Логирование всех операций
- Гибкая конфигурация моделей и параметров

## Структура проекта

```
future2/
├── bot.py                 # Telegram бот и обработчики команд
├── config.py             # Конфигурация LLM-провайдеров
├── csv_reader.py         # Чтение и обработка CSV файлов
├── data_extractor.py     # Извлечение данных из различных источников
├── database.py           # Работа с базой данных MongoDB
├── llm_client.py         # Клиенты для работы с LLM-провайдерами
├── logger_config.py      # Конфигурация логирования
├── main.py              # Точка входа в приложение
├── prompt_manager.py     # Управление промптами для LLM
├── text_processor.py     # Обработка и анализ текста
├── trend_analyzer.py     # Анализ трендов
├── vector_store.py       # Векторное хранение данных
└── requirements.txt      # Зависимости проекта
```

## Установка и настройка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd future2
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` в корневой директории проекта со следующими параметрами:
```env
# API ключи для LLM-провайдеров
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# Выбор провайдера по умолчанию (deepseek, openai, gemini)
LLM_PROVIDER=deepseek

# Настройки базы данных MongoDB
MONGODB_URI=your_mongodb_uri
MONGODB_DB=your_database_name

# Настройки Telegram бота
TELEGRAM_BOT_TOKEN=your_bot_token
```

## Конфигурация LLM-провайдеров

В файле `config.py` вы можете настроить параметры для каждого LLM-провайдера:

```python
LLM_CONFIG = {
    "deepseek": {
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 2
    },
    "openai": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 2
    },
    "gemini": {
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_output_tokens": None,
        "max_retries": 2
    }
}
```

## Использование

### Запуск бота

```bash
python main.py
```

### Примеры использования API

1. Анализ текста с помощью LLM:
```python
from llm_client import get_llm_client

# Получение клиента для выбранного провайдера
client = get_llm_client()  # использует провайдер по умолчанию
# или
client = get_llm_client("openai")  # явное указание провайдера

# Анализ текста
result = client.analyze_text(
    prompt="Проанализируй следующий текст",
    query="Ваш текст для анализа"
)
```

2. Извлечение ключевых слов:
```python
keywords = client.extract_keywords("Ваш текст для анализа")
```

3. Извлечение ключевых фраз:
```python
phrases = client.extract_key_phrases("Ваш текст для анализа")
```

4. Анализ трендов:
```python
from trend_analyzer import TrendAnalyzer

analyzer = TrendAnalyzer()
trends = analyzer.analyze_trends("Ваш текст для анализа")
```

## Компоненты системы

### LLM Client (llm_client.py)
- Поддержка различных LLM-провайдеров
- Единый интерфейс для работы с разными моделями
- Гибкая конфигурация параметров

### Trend Analyzer (trend_analyzer.py)
- Анализ трендов в текстовых данных
- Извлечение ключевых паттернов
- Классификация трендов

### Vector Store (vector_store.py)
- Векторное хранение данных
- Семантический поиск
- Индексация и поиск похожих документов

### Text Processor (text_processor.py)
- Предобработка текста
- Извлечение ключевых слов
- Нормализация текста

### Data Extractor (data_extractor.py)
- Извлечение данных из различных источников
- Парсинг веб-страниц
- Обработка структурированных данных

### Prompt Manager (prompt_manager.py)
- Управление промптами для LLM
- Шаблоны для различных типов анализа
- Динамическая генерация промптов

## Логирование

Система использует подробное логирование всех операций. Логи сохраняются в директории `logs/` и содержат информацию о:
- Запросах к LLM
- Ошибках и исключениях
- Результатах анализа
- Операциях с базой данных

## Зависимости

Основные зависимости проекта:
- aiogram >= 3.0.0 - для работы с Telegram API
- langchain и его расширения - для работы с LLM
- sentence-transformers - для векторных операций
- qdrant-client - для векторного хранения
- pymongo - для работы с MongoDB
- pandas и numpy - для обработки данных

## Безопасность

- API ключи хранятся в файле `.env`
- Все чувствительные данные логируются с маскированием
- Реализована система обработки ошибок и исключений

## Расширение функциональности

Для добавления нового LLM-провайдера:
1. Добавьте конфигурацию в `config.py`
2. Создайте новый класс клиента в `llm_client.py`
3. Добавьте поддержку в функцию `get_llm_client`

## Анализ трендов

Система предоставляет два типа анализа трендов: быстрый и глубокий. Каждый тип анализа имеет свои особенности и оптимальные сценарии использования.

### Быстрый анализ (Quick Analysis)

Быстрый анализ предназначен для получения быстрых результатов при работе с ограниченным набором данных. Он использует оптимизированный подход для быстрого получения результатов.

Особенности:
- Ограниченный набор данных (до 200 материалов)
- Фильтрация по релевантности (порог по умолчанию 0.7)
- Использование ключевых фраз для фокусировки анализа
- Оптимизированная обработка для быстрого ответа

Пример использования:
```python
from trend_analyzer import TrendAnalyzer
from datetime import datetime

analyzer = TrendAnalyzer()

# Быстрый анализ с фильтрацией по категории и датам
result = analyzer.analyze_trends_quick(
    query="Какие тренды в игровой индустрии в 2025 году?",
    category="Видеоигры",
    relevance_threshold=0.7,
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)
```

### Глубокий анализ (Deep Analysis)

Глубокий анализ предназначен для комплексного исследования трендов с использованием всего доступного набора данных. Он обеспечивает более детальный и всесторонний анализ.

Особенности:
- Анализ всего доступного набора данных
- Детальное исследование взаимосвязей
- Разбиение данных на контекстные чанки
- Генерация подробного отчета с детальными выводами

Пример использования:
```python
from trend_analyzer import TrendAnalyzer
from datetime import datetime

analyzer = TrendAnalyzer()

# Глубокий анализ с фильтрацией по категории и датам
result = analyzer.analyze_trends_deep(
    query="Как развиваются технологии в игровой индустрии?",
    category="Видеоигры",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)
```

### Сравнение типов анализа

| Характеристика | Быстрый анализ | Глубокий анализ |
|----------------|----------------|-----------------|
| Скорость | Быстрый | Медленный |
| Объем данных | До 200 материалов | Все доступные материалы |
| Детализация | Базовая | Подробная |
| Релевантность | Фильтрация по порогу | Полный анализ |
| Использование | Предварительный анализ | Детальное исследование |
| Ресурсы | Минимальные | Значительные |

## Запуск проекта с Docker

1. Убедитесь, что у вас установлены:
   - Docker Desktop
   - Docker Compose

2. Соберите и запустите контейнеры:
```bash
docker-compose up -d --build
```

3. Проверьте работу контейнеров:
```bash
docker ps
```

4. Для остановки:
```bash
docker-compose down
```

5. Для просмотра логов:
```bash
docker-compose logs -f
```

### Важные команды

- Пересобрать контейнеры после изменений:
```bash
docker-compose up -d --build
```

- Очистить все данные (включая Qdrant):
```bash
docker-compose down -v
```

- Войти в контейнер приложения:
```bash
docker exec -it trendwatching_app bash
```

### Настройки окружения

Создайте файл `.env` в корне проекта с переменными окружения:
```env
# Qdrant
QDRANT_URL=http://qdrant:6333

# LLM провайдеры
DEEPSEEK_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
LLM_PROVIDER=deepseek

# Telegram бот
TELEGRAM_BOT_TOKEN=your_token

# MongoDB
MONGODB_URI=mongodb://mongo:27017
MONGODB_DB=trendwatching
```

