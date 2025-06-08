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

# Создаем индексы для подписок
db.subscriptions.create_index("user_id", unique=True)

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

def get_user_subscription(user_id: str) -> Dict[str, Any]:
    """
    Получает настройки подписки пользователя
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Dict[str, Any]: Настройки подписки
    """
    try:
        subscription = db.subscriptions.find_one({"user_id": user_id})
        if subscription:
            return {
                'enabled': subscription.get('enabled', False),
                'category': subscription.get('category', None)
            }
        return {
            'enabled': False,
            'category': None
        }
    except Exception as e:
        logger.error(f"Ошибка при получении подписки пользователя {user_id}: {str(e)}")
        return {
            'enabled': False,
            'category': None
        }

def update_user_subscription(user_id: str, category: str) -> bool:
    """
    Обновляет настройки подписки пользователя
    
    Args:
        user_id: ID пользователя
        category: Категория для подписки
        
    Returns:
        bool: True если успешно, False если произошла ошибка
    """
    try:
        db.subscriptions.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id": user_id,
                "enabled": True,
                "category": category,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Ошибка при обновлении подписки пользователя {user_id}: {str(e)}")
        return False

def toggle_subscription(user_id: str) -> bool:
    """
    Переключает состояние подписки пользователя
    
    Args:
        user_id: ID пользователя
        
    Returns:
        bool: Новое состояние подписки
    """
    try:
        subscription = get_user_subscription(user_id)
        subscription['enabled'] = not subscription.get('enabled', False)
        
        db.subscriptions.update_one(
            {"user_id": user_id},
            {"$set": {
                "enabled": subscription['enabled'],
                "updated_at": datetime.utcnow()
            }}
        )
        return subscription['enabled']
    except Exception as e:
        logger.error(f"Ошибка при переключении подписки пользователя {user_id}: {str(e)}")
        return False

def get_subscribed_users() -> List[Dict[str, Any]]:
    """
    Получает список пользователей с активными подписками
    
    Returns:
        List[Dict[str, Any]]: Список словарей с ID пользователя и категорией
    """
    try:
        return list(db.subscriptions.find(
            {"enabled": True},
            {"user_id": 1, "category": 1, "_id": 0}
        ))
    except Exception as e:
        logger.error(f"Ошибка при получении списка подписанных пользователей: {str(e)}")
        return []

def create_subscription(user_id: str) -> bool:
    """
    Создает новую подписку для пользователя
    
    Args:
        user_id (str): ID пользователя
        
    Returns:
        bool: True если подписка успешно создана, False в противном случае
    """
    try:
        # Проверяем, существует ли уже подписка
        existing_subscription = db.subscriptions.find_one({"user_id": user_id})
        if existing_subscription:
            logger.info(f"Подписка для пользователя {user_id} уже существует")
            return True
            
        # Создаем новую подписку
        subscription = {
            "user_id": user_id,
            "enabled": False,
            "category": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = db.subscriptions.insert_one(subscription)
        if result.inserted_id:
            logger.info(f"Создана новая подписка для пользователя {user_id}")
            return True
        else:
            logger.error(f"Не удалось создать подписку для пользователя {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при создании подписки: {str(e)}")
        return False 