import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, Range, Payload
from logger_config import setup_logger
from text_processor import TextProcessor

# Настраиваем логгер
logger = setup_logger("vector_store")

class VectorStore:
    def __init__(
        self,
        collection_name: str = "trends",
        vector_size: int = 384,  # Размерность для all-MiniLM-L6-v2
        host: str = "localhost",
        port: int = 6333
    ):
        """
        Инициализация хранилища векторов
        
        Args:
            collection_name: Название коллекции в Qdrant
            vector_size: Размерность векторов
            host: Хост Qdrant
            port: Порт Qdrant
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(host=host, port=port)
        self.text_processor = TextProcessor()
        
        # Создаем коллекцию, если она не существует
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Создает коллекцию в Qdrant, если она не существует"""
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        on_disk=True  # Сохраняем векторы на диск
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,  # Индексируем сразу
                        memmap_threshold=10000  # Используем memory mapping для больших коллекций
                    )
                )
                logger.info(f"Создана новая коллекция: {self.collection_name}")
            else:
                logger.info(f"Коллекция {self.collection_name} уже существует")
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {str(e)}")
            raise

    def _parse_date(self, date_str: str) -> str:
        """Преобразует дату из различных форматов в формат '%Y-%m-%d'"""
        try:
            # Пробуем распарсить дату в формате RFC 2822 с полным годом
            date_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Пробуем распарсить дату в формате RFC 2822 с сокращенным годом
                date_obj = datetime.strptime(date_str, "%a, %d %b %y %H:%M:%S %z")
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    # Пробуем распарсить дату в формате 'YYYY-MM-DD HH:MM:SS'
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    try:
                        # Пробуем распарсить дату в формате ISO
                        date_obj = datetime.fromisoformat(date_str)
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        # Если не удалось распарсить, возвращаем оригинальную строку
                        return date_str

    def store_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Сохраняет векторы в коллекцию
        
        Args:
            vectors: Список векторов
            texts: Список текстов
            metadata: Список метаданных
            
        Returns:
            bool: True если сохранение прошло успешно, False в противном случае
        """
        try:
            if not vectors or not texts or not metadata:
                logger.warning("Пустые данные для сохранения")
                return False
            
            points = []
            
            for i, (vector, text, meta) in enumerate(zip(vectors, texts, metadata)):
                # Создаем UUID для точки
                point_id = str(uuid.uuid4())
                
                # Логируем информацию о сохраняемой точке
                logger.info(f"Сохранение точки {i+1}/{len(vectors)}")
                logger.info(f"URL: {meta.get('url', 'N/A')}")
                logger.info(f"Заголовок: {meta.get('title', 'N/A')}")
                logger.info(f"Категория: {meta.get('category', 'N/A')}")
                
                # Преобразуем дату в нужный формат
                date_str = meta.get("date", "")
                formatted_date = self._parse_date(date_str)
                
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": text,
                        "url": meta.get("url", ""),
                        "title": meta.get("title", ""),
                        "category": meta.get("category", ""),
                        "date": formatted_date,  # Используем преобразованную дату
                        "date_timestamp": datetime.strptime(formatted_date, "%Y-%m-%d").timestamp() if formatted_date else None,
                        "source_type": meta.get("source_type", ""),
                        "chunk_index": i,
                        "total_chunks": len(vectors),
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Сохраняем векторы
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Принудительно запускаем индексацию
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0
                )
            )
            
            logger.info(f"Сохранено {len(points)} чанков в коллекцию {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении векторов: {str(e)}")
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        score_threshold: float = 0.7,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[dict]:
        """
        Поиск векторов в Qdrant с фильтрацией по релевантности и метаданным.
        
        Args:
            query_vector: Вектор запроса.
            score_threshold: Минимальный порог схожести (0.0 - 1.0).
            limit: Максимальное количество результатов.
            category: Фильтр по категории.
            start_date: Начальная дата.
            end_date: Конечная дата.
            
        Returns:
            List[dict]: Релевантные материалы.
        """
        try:
            # Подготовка фильтра для Qdrant
            filters = []
            if category:
                filters.append(
                    FieldCondition(
                        key="category",
                        match={"value": category}
                    )
                )
            if start_date and end_date:
                filters.append(
                    FieldCondition(
                        key="date",
                        range={
                            "gte": start_date.isoformat(),
                            "lte": end_date.isoformat()
                        }
                    )
                )
            
            # Поиск в Qdrant с учетом порога релевантности
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=Filter(must=filters) if filters else None,
                score_threshold=score_threshold,
                limit=limit or 100  # Разумный лимит по умолчанию
            )
            
            # Преобразование результатов в нужный формат
            return [
                {
                    "id": hit.id,
                    "text": hit.payload.get("text", ""),
                    "title": hit.payload.get("title", ""),
                    "date": hit.payload.get("date"),
                    "category": hit.payload.get("category"),
                    "score": hit.score  # Для отладки
                }
                for hit in search_result
            ]
        except Exception as e:
            logger.error(f"Ошибка при поиске векторов: {str(e)}")
            return []

    def delete_vectors(self, filter_conditions: Dict[str, Any]) -> bool:
        """
        Удаляет векторы по условиям фильтра
        
        Args:
            filter_conditions: Условия фильтрации
            
        Returns:
            bool: True если успешно, False в случае ошибки
        """
        try:
            logger.info(f"Удаление векторов с условиями: {filter_conditions}")
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                            for key, value in filter_conditions.items()
                        ]
                    )
                )
            )
            
            logger.info("Векторы успешно удалены")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении векторов: {str(e)}")
            return False

    def get_categories(self) -> List[str]:
        """
        Получение списка уникальных категорий из коллекции
        
        Returns:
            List[str]: Список категорий
        """
        try:
            # Получаем все точки из коллекции
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Максимальное количество точек
            )[0]
            
            # Извлекаем уникальные категории
            categories = set()
            for point in points:
                category = point.payload.get("category", "")
                if category:
                    categories.add(category)
            
            logger.info(f"Найдено {len(categories)} уникальных категорий")
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Ошибка при получении категорий: {str(e)}")
            return []

    def add_materials(self, materials: List[Dict[str, Any]]) -> bool:
        """
        Добавляет материалы в векторное хранилище
        
        Args:
            materials: Список материалов для добавления
            
        Returns:
            bool: True если добавление прошло успешно, False в противном случае
        """
        try:
            if not materials:
                logger.warning("Пустой список материалов для добавления")
                return False
            
            vectors = []
            texts = []
            metadata = []
            
            for material in materials:
                # Формируем текст для векторизации
                text = f"{material.get('title', '')} {material.get('description', '')} {material.get('content', '')}"
                texts.append(text)
                
                # Формируем метаданные
                meta = {
                    'url': material.get('url', ''),
                    'title': material.get('title', ''),
                    'description': material.get('description', ''),
                    'content': material.get('content', ''),
                    'date': material.get('date', ''),
                    'category': material.get('category', ''),
                    'source_type': material.get('source_type', '')
                }
                metadata.append(meta)
            
            # Получаем векторы для текстов
            vectors = self.text_processor.get_embeddings(texts)
            
            # Сохраняем векторы
            return self.store_vectors(vectors, texts, metadata)
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении материалов: {str(e)}")
            return False

    def upsert_vector(
        self,
        vector: List[float],
        payload: Dict[str, Any],
        vector_id: Optional[int] = None
    ) -> bool:
        """
        Добавление или обновление вектора в хранилище.
        
        Args:
            vector: Вектор для добавления.
            payload: Метаданные (текст, заголовок, дата, категория и т.д.).
            vector_id: ID вектора (опционально).
            
        Returns:
            bool: Успешно ли выполнена операция.
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": vector_id or payload.get("id"),
                        "vector": vector,
                        "payload": payload
                    }
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Ошибка при добавлении вектора: {str(e)}")
            return False 