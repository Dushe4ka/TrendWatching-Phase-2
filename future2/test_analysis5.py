# test_analysis.py

import logging
from typing import Dict, Any, List
from llm_client import get_llm_client
from vector_store import VectorStore
from text_processor import TextProcessor
from logger_config import setup_logger
import tiktoken
import json

logger = setup_logger("test_analysis")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Ошибка при подсчете токенов: {str(e)}")
        return len(text.split()) * 1.3

def calculate_chunk_size(materials: List[Dict[str, Any]], max_context_size: int) -> int:
    available_tokens = int(max_context_size * 0.8)
    total_tokens = sum(count_tokens(material['text']) for material in materials)
    avg_tokens = total_tokens / len(materials)
    return max(1, int(available_tokens / avg_tokens))

def _create_context_aware_chunks(materials: List[Dict[str, Any]], max_context_size: int) -> List[List[Dict[str, Any]]]:
    chunks, current_chunk, current_size = [], [], 0
    chunk_size = calculate_chunk_size(materials, max_context_size)

    for material in materials:
        tokens = count_tokens(material['text'])
        if current_size + tokens > max_context_size * 0.8:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [material]
            current_size = tokens
        else:
            current_chunk.append(material)
            current_size += tokens

        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk, current_size = [], 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def test_embeddings():
    try:
        text_processor = TextProcessor(
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        embedding = text_processor.create_embeddings(["Test embedding"])[0]
        logger.info(f"Размерность эмбеддинга: {len(embedding)}")
        return len(embedding)
    except Exception as e:
        logger.error(f"Ошибка в тесте эмбеддинга: {str(e)}")
        return None

def analyze_trend(
    category: str,
    user_query: str,
    embedding_type: str = "openai",
    openai_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    try:
        llm_client = get_llm_client()
        vector_store = VectorStore(
            embedding_type=embedding_type,
            openai_model=openai_model
        )
        text_processor = TextProcessor(
            embedding_type=embedding_type,
            openai_model=openai_model
        )

        theme_prompt = f"""
        Ты анализируешь информацию для категорийного менеджера цифровых товаров в сфере {category}.
        На основе запроса пользователя:

        Запрос: {user_query}

        Выдели максимально конкретную тему, игру, продукт, событие или ключевое словосочетание, которое будет использовано для поиска релевантных материалов.
        Верни только эту тему/словосочетание, без лишних слов и пояснений.
        """
        theme = llm_client.analyze_text(theme_prompt, user_query).get('analysis', '').strip()
        logger.info(f"Тема запроса: {theme}")

        search_embedding = text_processor.create_embeddings([theme])[0]
        relevant_materials = vector_store.search_vectors(
            query_vector=search_embedding,
            category=category,
            score_threshold=0.30
        )

        if not relevant_materials:
            logger.warning("Нет подходящих материалов")
            return {'status': 'error', 'message': 'Нет релевантных материалов'}

        max_context_size = llm_client.get_max_context_size()
        total_tokens = sum(count_tokens(m['text']) for m in relevant_materials)

        if total_tokens <= max_context_size * 0.8:
            filter_prompt = f"""
            Ты работаешь для категорийного менеджера цифровых товаров в сфере {category}. На основе материалов по теме "{theme}" и запроса пользователя:

            Запрос пользователя: {user_query}

            Проанализируй список материалов. Оставь только те материалы (text и url), которые очень релевантны запросу и теме (например, относятся к конкретной игре, событию, релизу, патчу, метрикам и т.д.). Исключи обобщенную информацию.
            Ответ верни СТРОГО в виде списка словарей: [{{"text": "...", "url": "..."}}]
            Если релевантных материалов нет, верни пустой список: []
            """
            filtered_materials_str = llm_client.analyze_text(filter_prompt, user_query).get('analysis', '')
            # Попытка парсинга JSON из строки
            try:
                filtered_materials = json.loads(filtered_materials_str)
                if not isinstance(filtered_materials, list):
                    filtered_materials = [] # В случае ошибки парсинга или если не список
            except json.JSONDecodeError:
                logger.error(f"Не удалось распарсить отфильтрованные материалы как JSON: {filtered_materials_str}")
                filtered_materials = [] # В случае ошибки парсинга

            if not filtered_materials:
                logger.warning("После фильтрации не осталось подходящих материалов")
                return {'status': 'error', 'message': 'Нет релевантных материалов после фильтрации'}

            analysis_prompt = f"""
            Ты - эксперт по анализу рынка цифровых товаров для категорийного менеджера. На основе следующего запроса и отфильтрованных релевантных материалов по теме "{theme}" в сфере {category}:

            Запрос пользователя: {user_query}

            Отфильтрованные материалы:
            {filtered_materials}

            Проведи подробный анализ и составь отчет для категорийного менеджера. Сфокусируйся на метриках (GMV, ADV, ETR, AOV, Orders, CR, ADV/GM), причинах изменений спроса/предложения, сравнении с конкурентами (если есть данные), и четких рекомендациях.

            Структура отчета:
            Тренды и События (что происходит):
            - Кратко, что произошло (дата, суть)
            - Ссылка на источник (если указана в материалах)

            Влияние и Метрики:
            - Как событие повлияло на продажи, активность игроков/пользователей, цены, метрики (GMV, ADV, CR и т.д.). Укажи конкретные метрики, если данные есть.
            - Общественная реакция (форумы, соцсети, стримеры - если упомянуто в материалах).

            Анализ и Рекомендации для категорийного менеджера:
            - Анализ ситуации (сравнение с конкурентами, сезонность, ЦА, цены).
            - Четкие рекомендации: что делать, на что обратить внимание (например, запуск акции, изменение цены, добавление новой услуги, реструктуризация категории и т.д.).

            Сделай отчет максимально полезным для принятия бизнес-решений.
            """
            final_analysis = llm_client.analyze_text(analysis_prompt, user_query)
            return {'status': 'ok', 'report': final_analysis.get('analysis', '')}

        else:
            chunks = _create_context_aware_chunks(relevant_materials, max_context_size)
            chunk_analyses_texts = [] # Сохраняем тексты анализов для финального объединения

            for i, chunk in enumerate(chunks):
                chunk_filter_prompt = f"""
                Ты работаешь для категорийного менеджера цифровых товаров в сфере {category}. На основе материалов (часть {i+1}/{len(chunks)}) по теме "{theme}" и запроса пользователя:

                Запрос пользователя: {user_query}

                Проанализируй список материалов. Оставь только те материалы (text и url), которые очень релевантны запросу и теме (например, относятся к конкретной игре, событию, релизу, патчу, метрикам и т.д.). Исключи обобщенную информацию.
                Ответ верни СТРОГО в виде списка словарей: [{{"text": "...", "url": "..."}}]
                Если релевантных материалов нет, верни пустой список: []
                """
                filtered_chunk_materials_str = llm_client.analyze_text(chunk_filter_prompt, user_query).get('analysis', '')
                try:
                    filtered_chunk_materials = json.loads(filtered_chunk_materials_str)
                    if not isinstance(filtered_chunk_materials, list):
                        filtered_chunk_materials = []
                except json.JSONDecodeError:
                    logger.error(f"Не удалось распарсить отфильтрованные материалы чанка {i+1} как JSON: {filtered_chunk_materials_str}")
                    filtered_chunk_materials = []

                if not filtered_chunk_materials:
                    logger.warning(f"После фильтрации в чанке {i+1} не осталось подходящих материалов")
                    continue # Пропускаем этот чанк, если нет релевантных материалов

                chunk_analysis_prompt = f"""
                Ты - эксперт по анализу рынка цифровых товаров для категорийного менеджера. На основе отфильтрованных материалов (часть {i+1}/{len(chunks)}) по теме "{theme}" в сфере {category}:

                Отфильтрованные материалы из чанка:
                {filtered_chunk_materials}

                Проведи промежуточный анализ для категорийного менеджера. Сфокусируйся на метриках (GMV, ADV, ETR, AOV, Orders, CR, ADV/GM), причинах изменений спроса/предложения, и потенциальных рекомендациях, основанных на этом чанке материалов.

                Структура промежуточного анализа (для последующего объединения):
                Тренды и События (что происходит):
                - Кратко, что произошло (дата, суть)
                - Ссылка на источник (если указана в материалах)

                Влияние и Метрики:
                - Как событие повлияло на продажи, активность, цены, метрики. Укажи конкретные метрики, если данные есть.
                - Общественная реакция (если упомянуто).

                Потенциальные Рекомендации:
                - Что можно сделать, на что обратить внимание, исходя из этого чанка.

                Сделай анализ максимально информативным. Если в чанке нет релевантной информации по этим пунктам, напиши это явно.
                """
                analysis = llm_client.analyze_text(chunk_analysis_prompt, user_query).get('analysis', '')
                chunk_analyses_texts.append(analysis)

            if not chunk_analyses_texts:
                 logger.warning("После анализа чанков не получено ни одного промежуточного анализа.")
                 return {'status': 'error', 'message': 'Не удалось получить анализ из материалов'}

            separator = '\n\n---\n\n'
            final_prompt = f"""
            Ты - главный аналитик для категорийного менеджера цифровых товаров в сфере {category}. Объедини следующие промежуточные анализы по теме "{theme}" в единый, связный и подробный отчет. Основывайся на исходном запросе пользователя: {user_query}.

            Промежуточные анализы для объединения:
            {separator.join(chunk_analyses_texts)}

            Составь финальный структурированный отчет для категорийного менеджера. Включи все ключевые данные из промежуточных анализов, устрани дублирование, синтезируй выводы и рекомендации.

            Структура финального отчета:
            Общий Обзор и Ключевые Тренды (по категории/продукту):
            - Краткое резюме основных событий и трендов.

            Детальный Анализ Влияния и Метрики:
            - Синтезированный анализ влияния на продажи, активность, цены, метрики (GMV, ADV, CR, и т.д.) со ссылками на конкретные данные из материалов, если возможно.
            - Общая картина общественной реакции.

            Конкурентный Анализ и Рыночная Ситуация:
            - Общие выводы по конкурентам и рынку на основе материалов.

            Рекомендации для Категорийного Менеджера:
            - Четкий список финальных рекомендаций для действий (запуск новых продуктов/услуг, изменение ценовой политики, маркетинговые активности и т.д.), основанных на всем анализе.

            Сделай отчет максимально практически применимым для принятия решений.
            """
            final_report = llm_client.analyze_text(final_prompt, user_query).get('analysis', '')
            return {'status': 'ok', 'report': final_report}

    except Exception as e:
        logger.error(f"Ошибка при анализе: {str(e)}")
        return {'status': 'error', 'message': str(e)}


if __name__ == "__main__":
    # Проверка размерности эмбеддинга
    embedding_size = test_embeddings()
    if embedding_size:
        logger.info(f"✅ Размерность эмбеддингов: {embedding_size}")

    # Пример использования функции анализа
    category = "Видеоигры"
    user_query = (
        "Определи текущие тренды спроса на внутриигровые предметы для 'Dota 2' (например, сундуки, ключи, сеты). Какие типы предметов сейчас наиболее популярны, и" 
        "есть ли данные о том, какие конкуренты активно продают эти товары? Предложи рекомендации по расширению ассортимента."
        )

    logger.info(f"🚀 Запуск анализа тренда для категории: {category} и запроса: {user_query}")
    result = analyze_trend(category, user_query)

    if result['status'] == 'ok':
        print("\n📊 Результаты анализа:")
        print(result['report'])
    else:
        print(f"\n❌ Ошибка: {result['message']}")
