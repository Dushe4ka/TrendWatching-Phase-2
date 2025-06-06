import logging
from typing import Dict, Any, List
from vector_store import VectorStore
from text_processor import TextProcessor
from logger_config import setup_logger

# Настраиваем логгер
logger = setup_logger("test_search")

def check_vector_store_content(embedding_type: str = "openai", openai_model: str = "text-embedding-3-small"):
    """
    Проверка содержимого векторного хранилища
    
    Args:
        embedding_type: Тип эмбеддингов ("ollama" или "openai")
        openai_model: Название модели для OpenAI
    """
    try:
        vector_store = VectorStore(
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        
        # Получаем список категорий
        categories = vector_store.get_categories()
        logger.info(f"Доступные категории: {categories}")
        
        # Получаем все точки из коллекции
        points = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            limit=10000
        )[0]
        
        logger.info(f"Всего точек в коллекции: {len(points)}")
        
        # Анализируем содержимое
        if points:
            logger.info("\nПримеры содержимого:")
            for i, point in enumerate(points[:3], 1):
                logger.info(f"\nТочка {i}:")
                logger.info(f"ID: {point.id}")
                logger.info(f"Категория: {point.payload.get('category', 'N/A')}")
                logger.info(f"Заголовок: {point.payload.get('title', 'N/A')}")
                logger.info(f"Текст: {point.payload.get('text', 'N/A')[:200]}...")
                
            # Проверяем распределение по категориям
            category_counts = {}
            for point in points:
                category = point.payload.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info("\nРаспределение по категориям:")
            for category, count in category_counts.items():
                logger.info(f"{category}: {count} материалов")
        else:
            logger.warning("Векторное хранилище пустое!")
            
    except Exception as e:
        logger.error(f"Ошибка при проверке содержимого: {str(e)}")

def test_embedding_dimension(embedding_type: str = "openai", openai_model: str = "text-embedding-3-small") -> int:
    """
    Тестирование размерности эмбеддингов
    
    Args:
        embedding_type: Тип эмбеддингов ("ollama" или "openai")
        openai_model: Название модели для OpenAI
        
    Returns:
        int: Размерность эмбеддингов или None в случае ошибки
    """
    try:
        text_processor = TextProcessor(
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        test_text = "This is a test sentence for embedding dimension check"
        embedding = text_processor.create_embeddings([test_text])[0]
        dimension = len(embedding)
        logger.info(f"Тестовая размерность эмбеддинга: {dimension}")
        logger.info(f"Первые 5 значений: {embedding[:5]}")
        return dimension
    except Exception as e:
        logger.error(f"Ошибка при тестировании эмбеддингов: {str(e)}")
        return None

def test_search(
    query: str,
    category: str = None,
    score_threshold: float = 0.5,
    limit: int = 5,
    embedding_type: str = "openai",
    openai_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Тестирование поиска релевантных материалов
    
    Args:
        query: Поисковый запрос
        category: Категория для поиска (опционально)
        score_threshold: Порог релевантности
        limit: Максимальное количество результатов
        embedding_type: Тип эмбеддингов ("ollama" или "openai")
        openai_model: Название модели для OpenAI
        
    Returns:
        Dict[str, Any]: Результаты поиска
    """
    try:
        # Инициализация компонентов
        text_processor = TextProcessor(
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        vector_store = VectorStore(
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        
        # Создаем эмбеддинг для запроса
        logger.info(f"Создаем эмбеддинг для запроса: {query}")
        query_embedding = text_processor.create_embeddings([query])[0]
        logger.info(f"Размерность эмбеддинга: {len(query_embedding)}")
        
        # Поиск релевантных материалов
        logger.info(f"Начинаем поиск материалов")
        logger.info(f"Категория: {category if category else 'Все категории'}")
        logger.info(f"Порог релевантности: {score_threshold}")
        
        results = vector_store.search_vectors(
            query_vector=query_embedding,
            category=category,
            score_threshold=score_threshold,
            limit=limit
        )
        
        if results:
            logger.info(f"Найдено {len(results)} релевантных материалов")
            # Выводим детали каждого результата
            for i, result in enumerate(results, 1):
                logger.info(f"\nРезультат {i}:")
                logger.info(f"Заголовок: {result.get('title', 'N/A')}")
                logger.info(f"Категория: {result.get('category', 'N/A')}")
                logger.info(f"Релевантность: {result.get('score', 'N/A'):.4f}")
                logger.info(f"Текст: {result.get('text', 'N/A')[:200]}...")
            
            return {
                'status': 'success',
                'results_count': len(results),
                'results': results
            }
        else:
            logger.warning("Не найдено релевантных материалов")
            return {
                'status': 'error',
                'message': 'Не найдено релевантных материалов'
            }
            
    except Exception as e:
        error_message = f"Ошибка при тестировании поиска: {str(e)}"
        logger.error(error_message)
        return {
            'status': 'error',
            'message': error_message
        }

def test_search_with_different_thresholds(
    query: str,
    category: str = None,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    embedding_type: str = "openai",
    openai_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Тестирование поиска с разными порогами релевантности
    
    Args:
        query: Поисковый запрос
        category: Категория для поиска (опционально)
        thresholds: Список порогов релевантности для тестирования
        embedding_type: Тип эмбеддингов ("ollama" или "openai")
        openai_model: Название модели для OpenAI
        
    Returns:
        Dict[str, Any]: Результаты тестирования
    """
    results = {}
    for threshold in thresholds:
        logger.info(f"\nТестирование с порогом {threshold}:")
        result = test_search(
            query, 
            category, 
            threshold,
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        results[threshold] = result
    
    return results

if __name__ == "__main__":
    # Настройки по умолчанию
    embedding_type = "openai"
    openai_model = "text-embedding-3-small"
    
    # Проверяем содержимое векторного хранилища
    logger.info("Проверка содержимого векторного хранилища:")
    check_vector_store_content(embedding_type, openai_model)
    
    # Проверяем размерность эмбеддингов
    dimension = test_embedding_dimension(embedding_type, openai_model)
    if dimension:
        logger.info(f"Размерность эмбеддингов: {dimension}")
        expected_dimension = 1536 if openai_model == "text-embedding-3-small" else 3072
        if dimension != expected_dimension:
            logger.warning(f"Размерность эмбеддингов ({dimension}) отличается от ожидаемой ({expected_dimension})")
            logger.warning("Необходимо обновить размерность в VectorStore и пересоздать коллекцию")
            
            # Пересоздаем коллекцию с новой размерностью
            vector_store = VectorStore(
                embedding_type=embedding_type,
                openai_model=openai_model
            )
            if vector_store.recreate_collection():
                logger.info("Коллекция успешно пересоздана с новой размерностью")
            else:
                logger.error("Не удалось пересоздать коллекцию")
    
    # Тестируем поиск с разными порогами
    test_query = "CS GO игровая механика и баланс"
    category = "Видеоигры"
    
    logger.info("\nТестирование поиска с разными порогами релевантности:")
    results = test_search_with_different_thresholds(
        test_query, 
        category,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        embedding_type=embedding_type,
        openai_model=openai_model
    )
    
    # Выводим статистику
    logger.info("\nСтатистика поиска:")
    for threshold, result in results.items():
        if result['status'] == 'success':
            logger.info(f"Порог {threshold}: найдено {result['results_count']} результатов")
        else:
            logger.info(f"Порог {threshold}: {result['message']}") 