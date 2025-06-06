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
        text_processor = TextProcessor()
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
        vector_store = VectorStore()
        text_processor = TextProcessor()
        
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
                score_threshold=0.70
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

        # 4. Проверяем общее количество токенов и максимальный размер контекста
        total_tokens = sum(len(material['text'].split()) for material in relevant_materials)
        max_context_size = llm_client.get_max_context_size()
        logger.info(f"Общее количество токенов: {total_tokens}")
        logger.info(f"Максимальный размер контекста модели: {max_context_size}")
        
        # 5. Разбиваем на чанки и анализируем
        if total_tokens <= max_context_size:
            # Если общее количество токенов не превышает контекстное окно, анализируем все материалы сразу
            logger.info("Количество токенов в пределах контекстного окна, анализируем все материалы сразу")
            chunks = [relevant_materials]
        else:
            # Если превышает, делим материалы на чанки
            logger.info(f"Количество токенов превышает контекстное окно, разбиваем на чанки")
            chunks = _create_context_aware_chunks(relevant_materials, max_context_size)
            logger.info(f"Материалы разбиты на {len(chunks)} чанков")
        
        # 6. Анализируем каждый чанк
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Анализ чанка {i+1}/{len(chunks)}")
            chunk_analysis = analyze_chunk(chunk, user_query)
            chunk_analyses.append(chunk_analysis.get('analysis', ''))
        
        # 7. Генерируем финальный отчет на основе всех чанков
        final_report = generate_final_report(chunk_analyses, user_query)
        
        return {
            'status': 'success',
            'theme': theme,
            'materials_count': len(relevant_materials),
            'analysis': final_report.get('analysis', ''),
            'materials': relevant_materials
        }
        
    except Exception as e:
        error_message = f"Ошибка при анализе тренда: {str(e)}"
        logger.error(error_message)
        return {
            'status': 'error',
            'message': error_message
        }

# Добавляем функцию _create_context_aware_chunks
def _create_context_aware_chunks(materials, max_context_size=1000):
    chunks = []
    current_chunk = []
    current_size = 0
    for material in materials:
        material_size = len(material['text'].split())  # Примерная оценка размера в токенах
        if current_size + material_size > max_context_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [material]
            current_size = material_size
        else:
            current_chunk.append(material)
            current_size += material_size
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Добавляем функцию analyze_chunk для анализа одного чанка
def analyze_chunk(chunk, query):
    """Анализ отдельного чанка материалов"""
    context = "\n".join([material['text'] for material in chunk])
    prompt = f"""
    Проанализируй следующие материалы по запросу: {query}
    
    Материалы для анализа:
    {context}
    
    Сделай анализ и предоставь ответ в формате:
    
    Результат
    • Событие: [название события]
    • Ссылка: [ссылка на материал]
    • Влияние:
    ◦ [пункт 1]
    ◦ [пункт 2] и тд
    • Рекомендации:
    ◦ [пункт 1]
    ◦ [пункт 2] и тд
    """
    
    return llm_client.analyze_text(prompt, query)

# Добавляем функцию generate_final_report для формирования финального отчета
def generate_final_report(chunk_analyses, query):
    """Генерация финального отчета на основе всех чанков"""
    prompt = f"""
    На основе следующих анализов отдельных частей материалов, сформируй единый отчет по запросу: {query}
    
    Анализы частей:
    {chunk_analyses}
    
    Сформируй структурированный отчет, объединив и обобщив информацию из всех частей.
    """
    
    return llm_client.analyze_text(prompt, query)

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

