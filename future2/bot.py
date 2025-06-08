import os
import logging
import asyncio
import json
from typing import Dict, Any, List, Set
from datetime import datetime, time
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from logger_config import setup_logger
from vector_store import VectorStore
from text_processor import TextProcessor
from llm_client import get_llm_client
from usecases.daily_news import analyze_trend
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from database import (
    get_user_subscription,
    update_user_subscription,
    toggle_subscription,
    get_subscribed_users,
    create_subscription
)

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

# Состояния FSM
class CSVUpload(StatesGroup):
    waiting_for_file = State()

class AnalysisStates(StatesGroup):
    waiting_for_category = State()
    waiting_for_query = State()

class DailyNewsStates(StatesGroup):
    waiting_for_date = State()
    waiting_for_category = State()

class SubscriptionStates(StatesGroup):
    waiting_for_category = State()

# Инициализация планировщика
scheduler = AsyncIOScheduler()

async def initialize_scheduler():
    """Инициализация и запуск планировщика"""
    try:
        if not scheduler.running:
            # Запускаем планировщик в фоновом режиме
            scheduler.start(paused=False)
            logger.info("Планировщик запущен")
            logger.info(f"Текущее время системы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Проверяем существующие задачи
            jobs = scheduler.get_jobs()
            logger.info(f"Активные задачи планировщика: {len(jobs)}")
            for job in jobs:
                logger.info(f"Задача {job.id}: следующий запуск - {job.next_run_time if hasattr(job, 'next_run_time') else 'не определено'}")
    except Exception as e:
        logger.error(f"Ошибка при инициализации планировщика: {str(e)}")
        raise

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
        "/upload - Загрузить CSV файл\n"
        "/daily_news - Получить сводку новостей за определенную дату\n"
        "/subscribe - Подписаться на ежедневные новости"
    )

@dp.message(Command("analyze"))
async def cmd_analyze(message: types.Message, state: FSMContext):
    # Получаем список категорий
    categories = vector_store.get_categories()
    
    if not categories:
        await message.answer("❌ Нет доступных категорий для анализа")
        return
    
    # Создаем клавиатуру с категориями
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=category)] for category in categories],
        resize_keyboard=True
    )
    
    await message.answer(
        "📊 Выберите категорию для анализа:",
        reply_markup=keyboard
    )
    await state.set_state(AnalysisStates.waiting_for_category)

@dp.message(AnalysisStates.waiting_for_category)
async def process_category(message: types.Message, state: FSMContext):
    category = message.text
    
    # Сохраняем категорию
    await state.update_data(category=category)
    
    await message.answer(
        "❓ Введите ваш запрос для анализа:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(AnalysisStates.waiting_for_query)

@dp.message(AnalysisStates.waiting_for_query)
async def process_query(message: types.Message, state: FSMContext):
    query = message.text
    
    # Получаем сохраненные данные
    data = await state.get_data()
    category = data.get("category")
    
    # Отправляем сообщение о начале анализа
    status_message = await message.answer("🔄 Начинаю анализ...")
    
    try:
        # Выполняем анализ
        from usecases.analysis import analyze_trend
        result = analyze_trend(
            category=category,
            user_query=query,
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        
        if result['status'] == 'error':
            await status_message.edit_text(f"❌ Ошибка: {result['message']}")
            return
        
        # Формируем финальный ответ
        response_parts = []
        
        # Заголовок
        response_parts.append("📊 Результаты анализа:\n")
        
        # Контекст
        response_parts.append("📝 Контекст анализа:")
        response_parts.append(f"• Категория: {category}")
        response_parts.append(f"• Тематика: {result['theme']}")
        response_parts.append(f"• Найдено материалов: {result['materials_count']}")
        response_parts.append("")
        
        # Финальный отчет
        if result.get("analysis"):
            response_parts.append("📈 ФИНАЛЬНЫЙ ОТЧЕТ")
            response_parts.append("="*50)
            response_parts.append(result['analysis'].strip())
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

@dp.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message, state: FSMContext):
    """Начало процесса подписки"""
    # Получаем список категорий
    categories = vector_store.get_categories()
    
    if not categories:
        await message.answer("❌ Нет доступных категорий для подписки")
        return
    
    # Создаем клавиатуру с категориями
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=category)] for category in categories],
        resize_keyboard=True
    )
    
    await message.answer(
        "📊 Выберите категорию для подписки:",
        reply_markup=keyboard
    )
    await state.set_state(SubscriptionStates.waiting_for_category)

@dp.message(SubscriptionStates.waiting_for_category)
async def process_subscription_category(message: types.Message, state: FSMContext):
    """Обработка выбранной категории для подписки"""
    category = message.text
    user_id = str(message.chat.id)
    
    try:
        # Создаем или обновляем подписку
        if update_user_subscription(user_id, category):
            await message.answer(
                f"✅ Подписка активирована!\n"
                f"Вы будете получать ежедневные обновления по категории: {category}",
                reply_markup=types.ReplyKeyboardRemove()
            )
            
            # Добавляем задачу в планировщик
            job_id = f"daily_digest_{message.chat.id}"
            try:
                # Удаляем существующую задачу, если она есть
                if scheduler.get_job(job_id):
                    scheduler.remove_job(job_id)
                    logger.info(f"Удалена существующая задача {job_id}")
                
                # Создаем новую задачу
                job = scheduler.add_job(
                    send_daily_digest,
                    CronTrigger(hour=13, minute=45),
                    args=[message.chat.id],
                    id=job_id,
                    replace_existing=True
                )
                
                logger.info(f"Создана задача ежедневного дайджеста для чата {message.chat.id}")
                
            except Exception as e:
                logger.error(f"Ошибка при создании задачи планировщика: {str(e)}")
        else:
            await message.answer(
                "❌ Произошла ошибка при активации подписки",
                reply_markup=types.ReplyKeyboardRemove()
            )
    except Exception as e:
        logger.error(f"Ошибка при обработке подписки: {str(e)}")
        await message.answer(
            "❌ Произошла ошибка при обработке подписки",
            reply_markup=types.ReplyKeyboardRemove()
        )
    
    finally:
        await state.clear()

@dp.message(Command("daily_news"))
async def cmd_daily_news(message: types.Message, state: FSMContext):
    """Начало процесса получения сводки новостей за определенную дату"""
    await message.answer(
        "📅 Введите дату в формате YYYY-MM-DD\n"
        "Например: 2024-03-20"
    )
    await state.set_state(DailyNewsStates.waiting_for_date)

@dp.message(DailyNewsStates.waiting_for_date)
async def process_daily_news_date(message: types.Message, state: FSMContext):
    """Обработка введенной даты"""
    date_text = message.text.strip()
    
    try:
        # Проверяем формат даты
        datetime.strptime(date_text, "%Y-%m-%d")
        
        # Сохраняем дату
        await state.update_data(analysis_date=date_text)
        
        # Получаем список категорий
        categories = vector_store.get_categories()
        
        if not categories:
            await message.answer("❌ Нет доступных категорий для анализа")
            await state.clear()
            return
        
        # Создаем клавиатуру с категориями
        keyboard = types.ReplyKeyboardMarkup(
            keyboard=[[types.KeyboardButton(text=category)] for category in categories],
            resize_keyboard=True
        )
        
        await message.answer(
            "📊 Выберите категорию для анализа:",
            reply_markup=keyboard
        )
        await state.set_state(DailyNewsStates.waiting_for_category)
        
    except ValueError:
        await message.answer(
            "❌ Неверный формат даты. Пожалуйста, используйте формат YYYY-MM-DD\n"
            "Например: 2024-03-20"
        )

@dp.message(DailyNewsStates.waiting_for_category)
async def process_daily_news_category(message: types.Message, state: FSMContext):
    """Обработка выбранной категории и отправка сводки новостей"""
    category = message.text
    
    # Получаем сохраненную дату
    data = await state.get_data()
    analysis_date = data.get("analysis_date")
    
    # Отправляем сообщение о начале анализа
    status_message = await message.answer("🔄 Начинаю анализ...")
    
    try:
        # Выполняем анализ
        result = analyze_trend(
            category=category,
            analysis_date=analysis_date
        )
        
        if result['status'] == 'error':
            await status_message.edit_text(f"❌ Ошибка: {result['message']}")
            return
        
        # Формируем сообщение
        message_parts = [
            f"📊 Сводка новостей за {analysis_date}\n",
            f"📋 Категория: {category}\n",
            "="*50 + "\n"
        ]
        
        if result.get("analysis"):
            message_parts.append(result['analysis'].strip())
        
        # Отправляем сообщение
        message_text = "\n".join(message_parts)
        message_chunks = [message_text[i:i+4000] for i in range(0, len(message_text), 4000)]
        
        for chunk in message_chunks:
            await message.answer(chunk)
        
        # Удаляем статусное сообщение
        await status_message.delete()
        
    except Exception as e:
        logger.error(f"Ошибка при анализе: {str(e)}")
        await status_message.edit_text(f"❌ Произошла ошибка при анализе: {str(e)}")
    
    finally:
        # Сбрасываем состояние и клавиатуру
        await state.clear()
        await message.answer(
            "Готово! Используйте /daily_news для получения новой сводки.",
            reply_markup=types.ReplyKeyboardRemove()
        )

async def set_commands():
    """Установка команд бота"""
    commands = [
        types.BotCommand(command="start", description="Запустить бота"),
        types.BotCommand(command="analyze", description="Начать анализ"),
        types.BotCommand(command="upload", description="Загрузить CSV файл"),
        types.BotCommand(command="subscribe", description="Подписаться на ежедневные новости"),
        types.BotCommand(command="daily_news", description="Получить сводку новостей за определенную дату")
    ]
    await bot.set_my_commands(commands)

async def send_welcome_message(chat_id: int):
    """Отправка приветственного сообщения"""
    welcome_text = (
        "👋 Привет! Я бот для анализа трендов в различных категориях.\n\n"
        "📊 Что я умею:\n"
        "• Ежедневно в 9:00 отправляю сводку новостей за последние сутки\n"
        "• Анализирую тренды по запросу\n"
        "• Помогаю отслеживать важные изменения в интересующих вас категориях\n\n"
        "🔍 Доступные команды:\n"
        "/analyze - Начать анализ трендов\n"
        "/upload - Загрузить CSV файл с данными\n\n"
        "📈 Категории для анализа:\n"
        "• Видеоигры\n\n"
        "❓ Если у вас есть вопросы, используйте команду /help"
    )
    await bot.send_message(chat_id=chat_id, text=welcome_text)

async def send_daily_digest(chat_id: int):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Запуск ежедневного дайджеста для чата {chat_id} в {current_time}")
        
        # Получаем настройки подписки пользователя
        user_id = str(chat_id)
        subscription = get_user_subscription(user_id)
        logger.info(f"Настройки подписки для пользователя {user_id}: {subscription}")
        
        if not subscription.get('enabled', False):
            logger.info(f"Пользователь {user_id} не подписан на дайджест")
            return
        
        category = subscription.get('category')
        if not category:
            logger.info(f"У пользователя {user_id} не выбрана категория для дайджеста")
            return
        
        # Получаем текущую дату
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Анализ данных за дату: {current_date}")
        
        # Формируем сообщение
        message_parts = [f"📊 Ежедневный дайджест за {current_date}:\n"]
        
        try:
            logger.info(f"Начинаем анализ категории {category}")
            result = analyze_trend(
                category=category,
                analysis_date=current_date
            )
            
            if result['status'] == 'error':
                logger.error(f"Ошибка при анализе категории {category}: {result['message']}")
                return
            
            message_parts.append(f"\n📋 Категория: {category}")
            message_parts.append("="*30)
            if result.get("analysis"):
                message_parts.append(result['analysis'].strip())
            message_parts.append("="*30)
            
            logger.info("Анализ успешно завершен")
            
        except Exception as e:
            logger.error(f"Ошибка при анализе категории {category}: {str(e)}")
            return
        
        if len(message_parts) > 1:  # Если есть результаты анализа
            # Отправляем сообщение частями
            message_text = "\n".join(message_parts)
            message_chunks = [message_text[i:i+4000] for i in range(0, len(message_text), 4000)]
            
            logger.info(f"Отправляем сообщение из {len(message_chunks)} частей")
            for i, chunk in enumerate(message_chunks, 1):
                await bot.send_message(chat_id, chunk)
                logger.info(f"Отправлена часть {i} из {len(message_chunks)}")
        else:
            logger.info("Нет данных для отправки")
            await bot.send_message(chat_id, "📊 На сегодня нет новых материалов для анализа.")
            
    except Exception as e:
        logger.error(f"Ошибка при отправке ежедневного дайджеста: {str(e)}")
        try:
            await bot.send_message(chat_id, "❌ Произошла ошибка при формировании ежедневного дайджеста. Пожалуйста, попробуйте позже.")
        except:
            pass

@dp.message(F.new_chat_members)
async def on_bot_added(message: types.Message):
    """Обработчик добавления бота в группу"""
    # Проверяем, что бот был добавлен
    bot_user = await bot.get_me()
    if any(member.id == bot_user.id for member in message.new_chat_members):
        # Отправляем приветственное сообщение
        await send_welcome_message(message.chat.id)
        
        # Создаем подписку для пользователя
        user_id = str(message.chat.id)
        try:
            create_subscription(user_id)
            logger.info(f"Создана подписка для пользователя {user_id}")
        except Exception as e:
            logger.error(f"Ошибка при создании подписки: {str(e)}")

async def main():
    """Основная функция запуска бота"""
    try:
        # Инициализируем планировщик
        await initialize_scheduler()
        
        # Проверяем, что планировщик запущен
        if not scheduler.running:
            logger.error("Планировщик не запущен!")
            return
            
        # Устанавливаем команды бота
        await set_commands()
        
        # Запускаем бота
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
