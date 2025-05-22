from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Проверяем наличие необходимых переменных
required_env_vars = ['MONGODB_URI', 'MONGODB_DB']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Отсутствуют необходимые переменные окружения: {', '.join(missing_vars)}")

# Подключение к MongoDB
try:
    client = MongoClient(
        os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
        serverSelectionTimeoutMS=5000
    )
    client.admin.command('ping')
    logger.info("✅ MongoDB подключена")
except ConnectionFailure as e:
    logger.error(f"❌ Ошибка подключения: {e}")
    raise

db = client[os.getenv('MONGODB_DB')]

# Создаем индексы
db.parsed_data.create_index("url", unique=True)
db.parsed_data.create_index([("category", 1), ("date", -1)])

def save_source(source: Dict[str, Any]) -> bool:
    """
    Сохраняет данные в базу данных
    
    Args:
        source: словарь с данными для сохранения
            {
                "url": str,
                "title": str,
                "description": str,
                "content": str,
                "date": str,
                "category": str,
                "type": str
            }
    """
    try:
        # Добавляем timestamp создания записи
        source['created_at'] = datetime.utcnow()
        
        # Сохраняем в коллекцию parsed_data
        db.parsed_data.update_one(
            {"url": source['url']},
            {"$set": source},
            upsert=True
        )
        logger.info(f"Данные сохранены: {source['title'][:50]}...")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {str(e)}")
        return False

def is_source_exists(url: str) -> bool:
    """
    Проверяет, существует ли запись с таким URL
    
    Args:
        url: URL для проверки
        
    Returns:
        bool: True если запись существует, False если нет
    """
    return db.parsed_data.count_documents({"url": url}) > 0

def get_all_sources() -> List[Dict[str, Any]]:
    """
    Получает все источники из MongoDB
    
    Returns:
        List[Dict[str, Any]]: Список источников
    """
    try:
        sources = list(db.parsed_data.find({}, {
            "_id": 0,  # Исключаем поле _id
            "url": 1,
            "title": 1,
            "description": 1,
            "content": 1,
            "date": 1,
            "category": 1,
            "source_type": 1
        }))
        return sources
    except Exception as e:
        logger.error(f"Ошибка при получении источников: {str(e)}")
        return []

def get_data_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Получает все записи по категории
    
    Args:
        category: категория для поиска
        
    Returns:
        list: список записей
    """
    try:
        return list(db.parsed_data.find(
            {"category": category},
            {"_id": 0}
        ).sort("date", -1))
    except Exception as e:
        logger.error(f"Ошибка при получении данных по категории {category}: {str(e)}")
        return []

def get_categories() -> List[str]:
    """
    Получает список всех категорий
    
    Returns:
        list: список уникальных категорий
    """
    try:
        return db.parsed_data.distinct("category")
    except Exception as e:
        logger.error(f"Ошибка при получении категорий: {str(e)}")
        return [] 