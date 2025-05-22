import logging
from typing import List, Dict, Any, Generator
from sentence_transformers import SentenceTransformer
import numpy as np
import re

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 1024,
        min_chunk_size: int = 50
    ):
        """
        Инициализация процессора текста
        
        Args:
            model_name: Название модели для создания эмбеддингов
            max_chunk_size: Максимальный размер чанка в токенах
            min_chunk_size: Минимальный размер чанка в токенах
        """
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"Инициализирован TextProcessor с моделью {model_name}")
        logger.info(f"Параметры чанков: макс.размер={max_chunk_size}, мин.размер={min_chunk_size}")
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и форматирования
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Очищенный текст
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Удаляем HTML-теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаляем множественные пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем специальные символы, оставляя знаки препинания
        text = re.sub(r'[^\w\s.,!?;:()\-–—]', '', text)
        
        # Удаляем множественные знаки препинания
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения
        
        Args:
            text: Исходный текст
            
        Returns:
            List[str]: Список предложений
        """
        if not text or not isinstance(text, str):
            logger.warning("Получен пустой или некорректный текст")
            return []
            
        # Разбиваем по знакам препинания, сохраняя их
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?':
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        
        # Добавляем последнее предложение, если оно есть
        if current.strip():
            sentences.append(current.strip())
            
        # Если предложений нет, разбиваем по переносам строк
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            
        # Если все еще нет предложений, возвращаем весь текст
        if not sentences:
            sentences = [text]
            
        logger.debug(f"Разбито на {len(sentences)} предложений")
        return sentences
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Создает чанки из текста, если он превышает максимальный размер
        
        Args:
            text: Исходный текст
            
        Returns:
            List[str]: Список чанков
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning("Получен пустой или некорректный текст")
                return []
                
            # Очищаем текст
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.warning("Текст пустой после очистки")
                return []
            
            # Если текст достаточно короткий, возвращаем его как один чанк
            if len(cleaned_text) // 4 <= self.max_chunk_size:
                logger.info("Текст не требует разбиения на чанки")
                return [cleaned_text]
                
            # Разбиваем на предложения
            sentences = self.split_into_sentences(cleaned_text)
            if not sentences:
                logger.warning("Не удалось разбить текст на предложения")
                return []
                
            # Создаем чанки
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                # Примерная оценка размера в токенах (4 символа ~ 1 токен)
                sentence_size = len(sentence) // 4
                
                if current_size + sentence_size > self.max_chunk_size and current_chunk:
                    # Сохраняем текущий чанк
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Добавляем последний чанк
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
            
            if not chunks:
                logger.warning(f"Не удалось создать чанки из текста длиной {len(text)} символов")
                logger.debug(f"Текст после очистки: {cleaned_text[:200]}...")
            else:
                logger.info(f"Создано {len(chunks)} чанков")
                logger.debug(f"Пример чанка: {chunks[0][:200]}...")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка при создании чанков: {str(e)}")
            return []
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Создает эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            List[np.ndarray]: Список эмбеддингов
        """
        try:
            if not texts:
                return []
                
            # Создаем эмбеддинги
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Преобразуем в список numpy массивов
            if isinstance(embeddings, np.ndarray):
                embeddings = [embeddings[i] for i in range(len(embeddings))]
            
            logger.info(f"Создано {len(embeddings)} эмбеддингов")
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {str(e)}")
            return []
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Обрабатывает текст: очищает, разбивает на чанки если нужно и создает эмбеддинги
        
        Args:
            text: Исходный текст
            
        Returns:
            Dict[str, Any]: Результаты обработки
        """
        try:
            if not text or not isinstance(text, str):
                return {
                    "success": False,
                    "chunks": [],
                    "embeddings": [],
                    "error": "Пустой или некорректный текст"
                }
            
            # Создаем чанки
            chunks = self.create_chunks(text)
            if not chunks:
                return {
                    "success": False,
                    "chunks": [],
                    "embeddings": [],
                    "error": "Не удалось создать чанки"
                }
            
            # Создаем эмбеддинги
            embeddings = self.create_embeddings(chunks)
            if not embeddings:
                return {
                    "success": False,
                    "chunks": chunks,
                    "embeddings": [],
                    "error": "Не удалось создать эмбеддинги"
                }
            
            return {
                "success": True,
                "chunks": chunks,
                "embeddings": embeddings,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обработке текста: {str(e)}")
            return {
                "success": False,
                "chunks": [],
                "embeddings": [],
                "error": str(e)
            }
    
    def process_batch(self, texts: List[str]) -> Generator[Dict[str, Any], None, None]:
        """
        Обрабатывает список текстов
        
        Args:
            texts: Список текстов
            
        Yields:
            Dict[str, Any]: Результаты обработки для каждого текста
        """
        for text in texts:
            yield self.process_text(text)
            
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            
        Returns:
            float: Значение косинусного сходства от 0 до 1
        """
        try:
            # Нормализуем векторы
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Вычисляем косинусное сходство
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ограничиваем значение от 0 до 1
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении сходства: {str(e)}")
            return 0.0
            
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Алиас для метода create_embeddings для обратной совместимости
        
        Args:
            texts: Список текстов
            
        Returns:
            List[np.ndarray]: Список эмбеддингов
        """
        return self.create_embeddings(texts) 