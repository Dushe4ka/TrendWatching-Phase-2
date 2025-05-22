import logging
from typing import Dict, Any, List
from vector_store import VectorStore
from text_processor import TextProcessor
from logger_config import setup_logger

logger = setup_logger("data_extractor")

class DataExtractor:
    """Класс для извлечения данных из векторного хранилища"""
    
    def __init__(self, vector_store: VectorStore, text_processor: TextProcessor):
        self.vector_store = vector_store
        self.text_processor = text_processor
    
    def extract_data(
        self,
        keywords: List[str],
        context: Dict[str, Any],
        batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Поиск и извлечение релевантных данных
        
        Args:
            keywords: Список ключевых слов
            context: Контекст поиска
            batch_size: Размер батча для поиска
            
        Returns:
            List[Dict[str, Any]]: Список найденных материалов
        """
        try:
            # Создаем эмбеддинги для ключевых слов
            keyword_embeddings = self.text_processor.create_embeddings(keywords)
            
            # Получаем все точки из хранилища
            total_points = len(self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0])
            
            if total_points == 0:
                logger.error("В хранилище нет данных для анализа")
                return []
            
            # Обрабатываем материалы батчами
            offset = 0
            all_materials = []
            
            while offset < total_points:
                logger.info(f"Обработка батча {offset + 1}-{min(offset + batch_size, total_points)}")
                
                # Получаем батч материалов
                results = []
                for embedding in keyword_embeddings:
                    batch_results = self.vector_store.search_vectors(
                        query_vector=embedding,
                        limit=batch_size,
                        offset=offset,
                        category=context.get("industry"),
                        start_date=context.get("timeframe"),
                        end_date=None,
                        date_format="%Y-%m-%d"
                    )
                    results.extend(batch_results)
                
                if not results:
                    logger.info("Больше результатов нет")
                    break
                
                # Фильтруем результаты по релевантности
                filtered_results = self._filter_results(results, keywords, context)
                all_materials.extend(filtered_results)
                
                offset += batch_size
            
            logger.info(f"Найдено материалов для анализа: {len(all_materials)}")
            return all_materials
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении данных: {str(e)}")
            raise
    
    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        keywords: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация результатов по релевантности
        
        Args:
            results: Список результатов
            keywords: Ключевые слова
            context: Контекст поиска
            
        Returns:
            List[Dict[str, Any]]: Отфильтрованные результаты
        """
        filtered_results = []
        
        for result in results:
            # Проверяем релевантность по ключевым словам
            relevance_score = self._calculate_relevance(result["text"], keywords)
            
            # Проверяем соответствие контексту
            context_match = self._check_context_match(result, context)
            
            if relevance_score > 0.3 and context_match:  # Снижаем порог релевантности
                result["relevance_score"] = relevance_score
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """
        Расчет релевантности текста по ключевым словам
        
        Args:
            text: Текст для проверки
            keywords: Ключевые слова
            
        Returns:
            float: Оценка релевантности
        """
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return keyword_matches / len(keywords) if keywords else 0
    
    def _check_context_match(self, result: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Проверка соответствия результата контексту
        
        Args:
            result: Результат поиска
            context: Контекст поиска
            
        Returns:
            bool: Соответствует ли результат контексту
        """
        # Проверяем соответствие категории
        if context.get("industry") and result.get("category") != context["industry"]:
            return False
        
        # Проверяем соответствие временному периоду
        if context.get("timeframe"):
            result_date = result.get("date")
            if not result_date or not self._is_date_in_timeframe(result_date, context["timeframe"]):
                return False
        
        return True
    
    def _is_date_in_timeframe(self, date: str, timeframe: str) -> bool:
        """
        Проверка, входит ли дата в указанный временной период
        
        Args:
            date: Дата для проверки
            timeframe: Временной период
            
        Returns:
            bool: Входит ли дата в период
        """
        try:
            from datetime import datetime
            
            # Преобразуем дату материала в datetime
            material_date = datetime.strptime(date, "%Y-%m-%d")
            
            # Если timeframe - это диапазон дат
            if " - " in timeframe:
                start_date, end_date = timeframe.split(" - ")
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                return start <= material_date <= end
            
            # Если timeframe - это одна дата
            else:
                target_date = datetime.strptime(timeframe, "%Y-%m-%d")
                return material_date == target_date
                
        except Exception as e:
            logger.error(f"Ошибка при проверке временного периода: {str(e)}")
            return True  # В случае ошибки пропускаем проверку 