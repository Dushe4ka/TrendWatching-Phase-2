import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from text_processor import TextProcessor
from vector_store import VectorStore
from llm_client import get_llm_client
from prompt_manager import PromptManager
from logger_config import setup_logger

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логгер
logger = setup_logger("trend_analyzer")

class TrendAnalyzer:
    """Класс для анализа трендов с использованием RAG"""
    
    def __init__(self):
        """Инициализация компонентов"""
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.llm_client = get_llm_client(provider="deepseek")
        self.prompt_manager = PromptManager()
        # Максимальный размер контекста для LLM (в токенах)
        self.max_context_size = 4000  # Примерное значение, можно настроить
        logger.info("TrendAnalyzer инициализирован")
    
    def analyze_trends_deep(
        self,
        query: str,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Глубокий анализ трендов с поиском по всему хранилищу
        
        Args:
            query: Текст запроса
            category: Категория для фильтрации
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        try:
            # 1. Создание эмбеддинга запроса
            query_embedding = self.text_processor.create_embeddings([query])[0]
            
            # 2. Поиск всех релевантных документов с большим лимитом
            materials = self.vector_store.search_vectors(
                query_vector=query_embedding,
                limit=1000000,  # Очень большой лимит для получения всех данных
                category=category,
                start_date=start_date,
                end_date=end_date
            )
            
            if not materials:
                return self._create_empty_response(query, category, start_date, end_date)
            
            # 3. Разбиение на чанки с учетом размера контекста
            chunks = self._create_context_aware_chunks(materials)
            chunk_analyses = [self._analyze_chunk(chunk, query) for chunk in chunks]
            
            # 4. Генерация финального отчета
            final_report = self._generate_final_report(chunk_analyses, query)
            
            return {
                "query": query,
                "context": {
                    "category": category,
                    "materials_count": len(materials),
                    "chunks_count": len(chunks),
                    "period": f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}" if start_date and end_date else "Весь период"
                },
                "materials": materials,
                "chunk_analyses": chunk_analyses,
                "final_report": final_report
            }
            
        except Exception as e:
            logger.error(f"Ошибка при глубоком анализе: {str(e)}")
            return self._create_empty_response(query, category, start_date, end_date, str(e))
    
    def analyze_trends_quick(
        self,
        query: str,
        category: Optional[str] = None,
        relevance_threshold: float = 0.7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Быстрый анализ трендов на основе ключевых предложений
        
        Args:
            query: Текст запроса
            category: Категория для фильтрации
            relevance_threshold: Порог релевантности
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        try:
            # 1. Извлечение ключевых предложений из запроса
            key_phrases = self.llm_client.extract_key_phrases(query)
            logger.info(f"Извлеченные ключевые предложения: {key_phrases}")
            
            # 2. Создание эмбеддинга для запроса
            query_embedding = self.text_processor.create_embeddings([query])[0]
            
            # 3. Поиск релевантных документов с учетом категории и дат
            materials = self.vector_store.search_vectors(
                query_vector=query_embedding,
                limit=200,  # Уменьшенный лимит для оптимизации
                category=category,
                start_date=start_date,
                end_date=end_date
            )
            
            if not materials:
                return self._create_empty_response(query, category, start_date, end_date)
            
            # 4. Фильтрация по релевантности
            relevant_materials = self._filter_by_relevance(materials, query, relevance_threshold)
            
            # 5. Разбиение на чанки с учетом размера контекста
            chunks = self._create_context_aware_chunks(relevant_materials)
            
            # 6. Анализ чанков с учетом ключевых предложений
            chunk_analyses = [
                self._analyze_chunk(chunk, query, key_phrases) 
                for chunk in chunks
            ]
            
            # 7. Генерация финального отчета
            final_report = self._generate_final_report(chunk_analyses, query, key_phrases)
            
            return {
                "query": query,
                "context": {
                    "category": category,
                    "materials_count": len(relevant_materials),
                    "chunks_count": len(chunks),
                    "key_phrases": key_phrases,
                    "period": f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}" if start_date and end_date else "Весь период"
                },
                "materials": relevant_materials,
                "chunk_analyses": chunk_analyses,
                "final_report": final_report
            }
            
        except Exception as e:
            logger.error(f"Ошибка при быстром анализе: {str(e)}")
            return self._create_empty_response(query, category, start_date, end_date, str(e))
    
    def _deduplicate_materials(self, materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Удаление дубликатов из списка материалов"""
        seen = set()
        unique_materials = []
        for material in materials:
            key = f"{material['title']}_{material['date']}"
            if key not in seen:
                seen.add(key)
                unique_materials.append(material)
        return unique_materials
    
    def _filter_by_relevance(
        self,
        materials: List[Dict[str, Any]],
        query: str,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация материалов по релевантности с учетом семантического сходства
        
        Args:
            materials: Список материалов
            query: Исходный запрос
            threshold: Порог релевантности
            
        Returns:
            List[Dict[str, Any]]: Отфильтрованные материалы
        """
        query_embedding = self.text_processor.create_embeddings([query])[0]
        relevant_materials = []
        
        for material in materials:
            # Создание эмбеддинга для материала
            material_embedding = self.text_processor.create_embeddings([material['text']])[0]
            
            # Расчет семантического сходства
            similarity = self.text_processor.calculate_similarity(
                query_embedding,
                material_embedding
            )
            
            if similarity >= threshold:
                material['relevance_score'] = similarity
                relevant_materials.append(material)
        
        # Сортировка по релевантности
        return sorted(relevant_materials, key=lambda x: x['relevance_score'], reverse=True)
    
    def _create_context_aware_chunks(self, materials: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Создание чанков с учетом размера контекста"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for material in materials:
            material_size = len(material['text'].split())  # Примерная оценка размера в токенах
            
            if current_size + material_size > self.max_context_size:
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
    
    def _analyze_chunk(
        self,
        chunk: List[Dict[str, Any]],
        query: str,
        key_phrases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Анализ чанка материалов с учетом ключевых предложений
        
        Args:
            chunk: Список материалов в чанке
            query: Исходный запрос
            key_phrases: Ключевые предложения (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        # Формирование контекста для анализа
        context = {
            'materials': chunk,
            'query': query,
            'key_phrases': key_phrases or []
        }
        
        # Создание промпта для анализа
        prompt = self.prompt_manager.get_chunk_analysis_prompt(context)
        
        # Анализ с помощью LLM
        analysis = self.llm_client.analyze_text(prompt, query)
        
        # Добавление информации о релевантности
        analysis['relevance'] = self._calculate_chunk_relevance(chunk, key_phrases or [])
        
        return analysis
    
    def _calculate_chunk_relevance(
        self,
        chunk: List[Dict[str, Any]],
        key_phrases: List[str]
    ) -> float:
        """
        Расчет релевантности чанка на основе ключевых предложений
        
        Args:
            chunk: Список материалов в чанке
            key_phrases: Ключевые предложения
            
        Returns:
            float: Оценка релевантности
        """
        if not chunk or not key_phrases:
            return 0.0
            
        # Создание эмбеддингов для ключевых предложений
        phrase_embeddings = self.text_processor.create_embeddings(key_phrases)
        
        # Расчет максимального сходства для каждого материала
        max_similarities = []
        for material in chunk:
            material_embedding = self.text_processor.create_embeddings([material['text']])[0]
            similarities = [
                self.text_processor.calculate_similarity(material_embedding, phrase_embedding)
                for phrase_embedding in phrase_embeddings
            ]
            max_similarities.append(max(similarities))
        
        # Возвращаем среднюю релевантность
        return sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
    
    def _generate_final_report(self, chunk_analyses: List[Dict[str, Any]], query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Генерация финального отчета"""
        prompt = self.prompt_manager.get_final_report_prompt(chunk_analyses, keywords, query)
        return self.llm_client.analyze_text(prompt, query)
    
    def _create_empty_response(
        self,
        query: str,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Создание пустого ответа в случае ошибки или отсутствия данных"""
        return {
            "query": query,
            "context": {
                "category": category,
                "materials_count": 0,
                "chunks_count": 0,
                "period": f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}" if start_date and end_date else "Весь период"
            },
            "error": error,
            "materials": [],
            "chunk_analyses": [],
            "final_report": None
        } 