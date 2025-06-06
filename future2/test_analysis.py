import logging
from typing import Dict, Any, List
from llm_client import get_llm_client
from vector_store import VectorStore
from text_processor import TextProcessor
from logger_config import setup_logger

# Настраиваем логгер
logger = setup_logger("test_analysis")

def test_embeddings():
    """
    Тестирование размерности эмбеддингов
    """
    try:
        text_processor = TextProcessor(
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        test_text = "This is a test sentence for embedding dimension check"
        embedding = text_processor.create_embeddings([test_text])[0]
        logger.info(f"Тестовая размерность эмбеддинга: {len(embedding)}")
        logger.info(f"Первые 5 значений: {embedding[:5]}")
        return len(embedding)
    except Exception as e:
        logger.error(f"Ошибка при тестировании эмбеддингов: {str(e)}")
        return None

def analyze_trend(category: str, user_query: str) -> Dict[str, Any]:
    """
    Анализ тренда на основе категории и запроса пользователя
    
    Args:
        category: Категория для анализа
        user_query: Запрос пользователя
        
    Returns:
        Dict[str, Any]: Результаты анализа
    """
    try:
        # Инициализация компонентов
        llm_client = get_llm_client()
        vector_store = VectorStore(
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        text_processor = TextProcessor(
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        
        # 1. Получаем основную тематику из запроса пользователя через LLM
        theme_prompt = f"""
        Проанализируй запрос пользователя и напиши короткое предложение, которое будет использоваться для поиска релевантных материалов.
        Тематика должна быть максимально конкретной и отражать суть запроса.
        
        Запрос: {user_query}
        
        Верни только предложение, без дополнительных пояснений.
        """
        
        theme_response = llm_client.analyze_text(theme_prompt, user_query)
        theme = theme_response.get('analysis', '').strip()
        logger.info(f"Выделенная тематика: {theme}")
        
        # 2. Создаем эмбеддинг только для тематики (theme)
        search_text = theme
        logger.info(f"Текст для векторизации: {search_text}")
        
        try:
            search_embedding = text_processor.create_embeddings([search_text])[0]
            logger.info(f"Размерность созданного эмбеддинга: {len(search_embedding)}")
            logger.info(f"Первые 5 значений эмбеддинга: {search_embedding[:5]}")
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {str(e)}")
            raise
        
        # 3. Ищем релевантные материалы по тематике и категории
        logger.info(f"Начинаем поиск материалов для категории: {category}")
        logger.info(f"Порог релевантности: 0.70")
        
        try:
            relevant_materials = vector_store.search_vectors(
                query_vector=search_embedding,
                category=category,
                score_threshold=0.70,
                limit=1000
            )
            
            if relevant_materials:
                logger.info(f"Найдено {len(relevant_materials)} релевантных материалов")
                # Логируем первые 3 результата для проверки
                for i, material in enumerate(relevant_materials[:3]):
                    logger.info(f"Результат {i+1}:")
                    logger.info(f"Заголовок: {material.get('title', 'N/A')}")
                    logger.info(f"Категория: {material.get('category', 'N/A')}")
                    logger.info(f"Релевантность: {material.get('score', 'N/A')}")
            else:
                logger.warning("Не найдено релевантных материалов")
                return {
                    'status': 'error',
                    'message': 'Не найдено релевантных материалов'
                }
                
        except Exception as e:
            logger.error(f"Ошибка при поиске материалов: {str(e)}")
            raise
        
        # 4. Анализируем найденные материалы с использованием исходного запроса (user_query)
        analysis_prompt = f"""
        Проанализируй следующие материалы по запросу: {user_query}
        
        Основная тематика: {theme}
        
        Материалы для анализа:
        {[material['text'] for material in relevant_materials]}
        {[material['url'] for material in relevant_materials]}

        Важно чтобы результаты были релевантными и соответствовали запросу пользователя.
        Сделай анализ и предоставь ответ в формате:

        Результат
        • Событие: Релиз кейса «Dragonfire Case» 15 октября 2024
        • Ссылка: тут должна быть ссылка на материал если ее нет, то напиши что ссылка отсутствует
        • Влияние:
        ◦ Продано 1.2 млн копий кейса за первые 24 часа (+40% к среднему показателю за
        последний год)
        ◦ Рост онлайна CS2 на 25% (пик — 1.8 млн игроков)
        ◦ Цена ножа «Dragonclaw» достигла $2000 на Steam Market (+300% за неделю)
        ◦ 45% обсуждений в Reddit содержат жалобы на «слишком низкий шанс выпадения
        ножа»
        • Рекомендации:
        b. Для инвесторов/трейдеров:
        ▪ Скупить скины из кейса в первые 2 недели (исторически цены растут через
        месяц после релиза)
        ▪ Мониторить активность стримеров, массовые открытия кейсов на Twitch могут
        вызвать скачки цен.

        таких событий может быть несколько, их нужно анализировать по очереди, и выводить в виде списка.
        

        Используй только текст, без форматирования. Ответ должен касаться только того что было сказано в запросе пользователя.
        """
        
        analysis_response = llm_client.analyze_text(analysis_prompt, user_query)
        
        # Добавляем URL в результаты анализа
        if relevant_materials:
            analysis_with_urls = []
            for material in relevant_materials:
                analysis_with_urls.append({
                    'event': material.get('title', 'N/A'),
                    'url': material.get('url', 'N/A'),
                    'analysis': analysis_response.get('analysis', '')
                })
        else:
            analysis_with_urls = []
        
        return {
            'status': 'success',
            'theme': theme,
            'materials_count': len(relevant_materials),
            'analysis': analysis_response.get('analysis', ''),
            'materials': relevant_materials,
            'analysis_with_urls': analysis_with_urls
        }
        
    except Exception as e:
        error_message = f"Ошибка при анализе тренда: {str(e)}"
        logger.error(error_message)
        return {
            'status': 'error',
            'message': error_message
        }

if __name__ == "__main__":
    # Сначала проверяем размерность эмбеддингов
    embedding_size = test_embeddings()
    if embedding_size:
        logger.info(f"Размерность эмбеддингов: {embedding_size}")
        if embedding_size != 1024:
            logger.warning(f"Размерность эмбеддингов ({embedding_size}) отличается от ожидаемой (1024)")
            logger.warning("Необходимо обновить размерность в VectorStore и пересоздать коллекцию")
    
    # Пример использования
    category = "Видеоигры"
    user_query = "Проанализируй CS GO, выяви ключевые факторы успеха/провала, спрогнозируй долгосрочную вовлеченность игроков, предложи возможные решения по развитию направления"
    
    result = analyze_trend(category, user_query)
    
    if result['status'] == 'success':
        print("\nРезультаты анализа:")
        print(f"Тематика: {result['theme']}")
        print(f"Проанализировано материалов: {result['materials_count']}")
        print("\nАнализ:")
        print(result['analysis'])
    else:
        print(f"Ошибка: {result['message']}") 

