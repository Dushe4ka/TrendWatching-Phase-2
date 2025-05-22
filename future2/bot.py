import os
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from trend_analyzer import TrendAnalyzer
from logger_config import setup_logger
from vector_store import VectorStore
from text_processor import TextProcessor
from llm_client import get_llm_client

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логгер
logger = setup_logger("bot")

# Инициализация бота
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация компонентов аналитики
vector_store = VectorStore()
text_processor = TextProcessor()
llm_client = get_llm_client()
analyzer = TrendAnalyzer()

# Состояния FSM
class CSVUpload(StatesGroup):
    waiting_for_file = State()

class AnalyticsStates(StatesGroup):
    waiting_for_analysis_type = State()
    waiting_for_category = State()
    waiting_for_start_date = State()
    waiting_for_end_date = State()
    waiting_for_query = State()

def clean_source_data(source: Dict[str, Any]) -> Dict[str, Any]:
    """Очистка и валидация данных источника"""
    cleaned = source.copy()
    
    # Преобразуем NaN в пустые строки
    for key in cleaned:
        if pd.isna(cleaned[key]):
            cleaned[key] = ''
        elif isinstance(cleaned[key], (float, np.float64)):
            cleaned[key] = str(cleaned[key])
    
    # Объединяем описание и контент, если контент пустой
    if not cleaned.get('content') and cleaned.get('description'):
        cleaned['content'] = cleaned['description']
    
    return cleaned

def print_source_info(source: Dict[str, Any], index: int):
    """Вывод информации об источнике"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Источник #{index + 1}:")
    logger.info(f"URL: {source.get('url', 'N/A')}")
    logger.info(f"Заголовок: {source.get('title', 'N/A')}")
    logger.info(f"Описание: {source.get('description', 'N/A')}")
    logger.info(f"Категория: {source.get('category', 'N/A')}")
    logger.info(f"Дата: {source.get('date', 'N/A')}")
    logger.info(f"Тип источника: {source.get('source_type', 'N/A')}")
    
    # Проверяем контент
    content = source.get('content', '')
    logger.info(f"Длина контента: {len(content) if content else 0} символов")
    if content:
        logger.info(f"Первые 200 символов контента: {content[:200]}")
    logger.info(f"{'='*50}\n")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Я бот для анализа трендов.\n\n"
        "Доступные команды:\n"
        "/analyze - Начать анализ\n"
        "/upload - Загрузить CSV файл"
    )

@dp.message(Command("analyze"))
async def cmd_analyze(message: types.Message):
    # Создаем инлайн-клавиатуру с типами анализа
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="🔍 Глубокий анализ",
                    callback_data="deep_analysis"
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="🚀 Быстрый анализ",
                    callback_data="quick_analysis"
                )
            ]
        ]
    )
    
    await message.answer(
        "📊 Выберите тип анализа:\n\n"
        "🔍 Глубокий анализ - полный поиск по всему векторному хранилищу\n"
        "🚀 Быстрый анализ - анализ на основе ключевых слов",
        reply_markup=keyboard
    )

@dp.callback_query(F.data.in_(["quick_analysis", "deep_analysis"]))
async def process_analysis_type(callback: types.CallbackQuery, state: FSMContext):
    # Сохраняем тип анализа
    analysis_type = "🔍 Глубокий анализ" if callback.data == "deep_analysis" else "🚀 Быстрый анализ"
    await state.update_data(analysis_type=analysis_type)
    
    # Отвечаем на callback
    await callback.answer()
    
    # Получаем список категорий
    categories = vector_store.get_categories()
    
    if not categories:
        await callback.message.answer("❌ Нет доступных категорий для анализа")
        return
    
    # Создаем клавиатуру с категориями
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=category)] for category in categories],
        resize_keyboard=True
    )
    
    await callback.message.answer(
        "📊 Выберите категорию для анализа:",
        reply_markup=keyboard
    )
    await state.set_state(AnalyticsStates.waiting_for_category)

@dp.message(AnalyticsStates.waiting_for_category)
async def process_category(message: types.Message, state: FSMContext):
    category = message.text
    
    # Сохраняем категорию
    await state.update_data(category=category)
    
    # Создаем клавиатуру с кнопкой "Пропустить"
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="⏩ Пропустить",
                    callback_data="skip_date"
                )
            ]
        ]
    )
    
    await message.answer(
        "📅 Введите начальную дату в формате ДД.ММ.ГГГГ:",
        reply_markup=keyboard
    )
    await state.set_state(AnalyticsStates.waiting_for_start_date)

@dp.message(AnalyticsStates.waiting_for_start_date)
async def process_start_date(message: types.Message, state: FSMContext):
    date_str = message.text.strip()
    
    # Создаем клавиатуру с кнопкой "Пропустить"
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="⏩ Пропустить",
                    callback_data="skip_date"
                )
            ]
        ]
    )
    
    try:
        start_date = datetime.strptime(date_str, "%d.%m.%Y")
        await state.update_data(start_date=start_date)
        
        await message.answer(
            "📅 Введите конечную дату в формате ДД.ММ.ГГГГ:",
            reply_markup=keyboard
        )
        await state.set_state(AnalyticsStates.waiting_for_end_date)
    except ValueError:
        await message.answer(
            "❌ Неверный формат даты. Попробуйте еще раз в формате ДД.ММ.ГГГГ",
            reply_markup=keyboard
        )

@dp.message(AnalyticsStates.waiting_for_end_date)
async def process_end_date(message: types.Message, state: FSMContext):
    date_str = message.text.strip()
    
    # Создаем клавиатуру с кнопкой "Пропустить"
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="⏩ Пропустить",
                    callback_data="skip_date"
                )
            ]
        ]
    )
    
    try:
        end_date = datetime.strptime(date_str, "%d.%m.%Y")
        await state.update_data(end_date=end_date)
        
        await message.answer(
            "❓ Введите ваш запрос для анализа:",
            reply_markup=types.ReplyKeyboardRemove()
        )
        await state.set_state(AnalyticsStates.waiting_for_query)
    except ValueError:
        await message.answer(
            "❌ Неверный формат даты. Попробуйте еще раз в формате ДД.ММ.ГГГГ",
            reply_markup=keyboard
        )

@dp.callback_query(F.data == "skip_date")
async def process_skip_date(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик нажатия на кнопку 'Пропустить'"""
    # Получаем текущее состояние
    current_state = await state.get_state()
    
    if current_state == AnalyticsStates.waiting_for_start_date.state:
        # Если мы в состоянии выбора начальной даты
        await state.update_data(start_date=None)
        await callback.message.answer(
            "📅 Введите конечную дату в формате ДД.ММ.ГГГГ:",
            reply_markup=types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text="⏩ Пропустить",
                            callback_data="skip_date"
                        )
                    ]
                ]
            )
        )
        await state.set_state(AnalyticsStates.waiting_for_end_date)
    
    elif current_state == AnalyticsStates.waiting_for_end_date.state:
        # Если мы в состоянии выбора конечной даты
        await state.update_data(end_date=None)
        await callback.message.answer(
            "❓ Введите ваш запрос для анализа:",
            reply_markup=types.ReplyKeyboardRemove()
        )
        await state.set_state(AnalyticsStates.waiting_for_query)
    
    # Отвечаем на callback
    await callback.answer()

@dp.message(AnalyticsStates.waiting_for_query)
async def process_query(message: types.Message, state: FSMContext):
    query = message.text
    
    # Получаем сохраненные данные
    data = await state.get_data()
    analysis_type = data.get("analysis_type")
    category = data.get("category")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    # Отправляем сообщение о начале анализа
    status_message = await message.answer("🔄 Начинаю анализ...")
    
    try:
        # Выполняем анализ в зависимости от типа
        if analysis_type == "🚀 Быстрый анализ":
            result = analyzer.analyze_trends_quick(
                query=query,
                category=category,
                start_date=start_date,
                end_date=end_date
            )
        else:  # Глубокий анализ
            result = analyzer.analyze_trends_deep(
                query=query,
                category=category,
                start_date=start_date,
                end_date=end_date
            )
        
        if "error" in result:
            await status_message.edit_text(f"❌ Ошибка: {result['error']}")
            return
        
        # Формируем финальный ответ
        response_parts = []
        
        # Заголовок
        response_parts.append(f"📊 Результаты анализа ({analysis_type}):\n")
        
        # Контекст
        response_parts.append("📝 Контекст анализа:")
        response_parts.append(f"• Категория: {result['context']['category']}")
        response_parts.append(f"• Период: {result['context']['period']}")
        response_parts.append(f"• Найдено материалов: {result['context']['materials_count']}")
        response_parts.append(f"• Количество чанков: {result['context']['chunks_count']}")
        if 'keywords' in result['context']:
            response_parts.append(f"• Ключевые слова: {', '.join(result['context']['keywords'])}")
        response_parts.append("")
        
        # Финальный отчет
        if result.get("final_report"):
            response_parts.append("📈 ФИНАЛЬНЫЙ ОТЧЕТ")
            response_parts.append("="*50)
            report = result['final_report'].get('analysis', '')
            if report.startswith('```markdown'):
                report = report[11:]
            if report.endswith('```'):
                report = report[:-3]
            response_parts.append(report.strip())
            response_parts.append("="*50)
        
        # Отправляем ответ частями
        response_text = "\n".join(response_parts)
        response_chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
        
        for chunk in response_chunks:
            await message.answer(chunk)
        
        # Удаляем статусное сообщение
        await status_message.delete()
        
    except Exception as e:
        logger.error(f"Ошибка при анализе: {str(e)}")
        await status_message.edit_text(f"❌ Произошла ошибка при анализе: {str(e)}")
    
    finally:
        # Сбрасываем состояние
        await state.clear()

@dp.message(Command("upload"))
async def cmd_upload(message: types.Message, state: FSMContext):
    await message.answer(
        "📤 Отправьте CSV файл для загрузки в базу данных.\n"
        "Файл должен содержать следующие колонки:\n"
        "- url: ссылка на источник\n"
        "- title: заголовок\n"
        "- description: описание\n"
        "- content: содержание\n"
        "- date: дата (формат: ДД.ММ.ГГГГ)\n"
        "- category: категория\n"
        "- source_type: тип источника"
    )
    await state.set_state(CSVUpload.waiting_for_file)

@dp.message(CSVUpload.waiting_for_file)
async def process_csv(message: types.Message, state: FSMContext):
    if not message.document:
        await message.answer("❌ Пожалуйста, отправьте CSV файл")
        return
    
    if not message.document.file_name.endswith('.csv'):
        await message.answer("❌ Файл должен быть в формате CSV")
        return
    
    # Скачиваем файл
    file = await bot.get_file(message.document.file_id)
    file_path = file.file_path
    
    # Создаем временный файл
    temp_file = f"temp_{message.document.file_id}.csv"
    
    try:
        # Скачиваем файл
        await bot.download_file(file_path, temp_file)
        
        # Читаем CSV
        df = pd.read_csv(temp_file)
        
        # Проверяем наличие необходимых колонок
        required_columns = ['url', 'title', 'description', 'content', 'date', 'category', 'source_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            await message.answer(f"❌ В файле отсутствуют следующие колонки: {', '.join(missing_columns)}")
            return
        
        # Обрабатываем данные
        processed_data = []
        
        for _, row in df.iterrows():
            source = {
                'url': row['url'],
                'title': row['title'],
                'description': row['description'],
                'content': row['content'],
                'date': row['date'],
                'category': row['category'],
                'source_type': row['source_type']
            }
            processed_source = clean_source_data(source)
            processed_data.append(processed_source)
        
        # Загружаем данные в векторное хранилище
        vector_store.add_materials(processed_data)
        
        await message.answer(
            f"✅ Данные успешно загружены!\n"
            f"• Загружено записей: {len(processed_data)}\n"
            f"• Категории: {', '.join(df['category'].unique())}"
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV: {str(e)}")
        await message.answer(f"❌ Произошла ошибка при обработке файла: {str(e)}")
    
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Сбрасываем состояние
        await state.clear()

async def set_commands():
    """Установка команд бота"""
    commands = [
        types.BotCommand(command="start", description="Запустить бота"),
        types.BotCommand(command="analyze", description="Начать анализ"),
        types.BotCommand(command="upload", description="Загрузить CSV файл")
    ]
    await bot.set_my_commands(commands)

async def main():
    """Основная функция"""
    # Устанавливаем команды
    await set_commands()
    
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
