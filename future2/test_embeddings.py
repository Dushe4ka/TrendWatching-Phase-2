import logging
from text_processor import TextProcessor
from vector_store import VectorStore
from logger_config import setup_logger

# Настраиваем логгер
logger = setup_logger("test_embeddings")

def test_embeddings():
    """
    Тестирование обоих типов эмбеддингов
    """
    try:
        # Тестируем OpenAI эмбеддинги
        logger.info("Тестирование OpenAI эмбеддингов...")
        text_processor_openai = TextProcessor(
            embedding_type="openai",
            openai_model="text-embedding-3-small"
        )
        test_text = "This is a test sentence for embedding dimension check"
        
        # Создаем эмбеддинги через OpenAI
        openai_embedding = text_processor_openai.create_embeddings([test_text])[0]
        logger.info(f"OpenAI размерность эмбеддинга: {len(openai_embedding)}")
        logger.info(f"OpenAI первые 5 значений: {openai_embedding[:5]}")
        
        # Тестируем Ollama эмбеддинги
        logger.info("\nТестирование Ollama эмбеддингов...")
        text_processor_ollama = TextProcessor(embedding_type="ollama")
        
        # Создаем эмбеддинги через Ollama
        ollama_embedding = text_processor_ollama.create_embeddings([test_text])[0]
        logger.info(f"Ollama размерность эмбеддинга: {len(ollama_embedding)}")
        logger.info(f"Ollama первые 5 значений: {ollama_embedding[:5]}")
        
        # Тестируем сохранение в VectorStore
        logger.info("\nТестирование сохранения в VectorStore...")
        
        # Создаем тестовые данные
        test_materials = [{
            'title': 'Test Title',
            'description': 'Test Description',
            'content': test_text,
            'url': 'http://test.com',
            'date': '2024-03-20',
            'category': 'test',
            'source_type': 'test'
        }]
        
        # Тестируем с Ollama
        logger.info("Сохранение с Ollama эмбеддингами...")
        vector_store_ollama = VectorStore(embedding_type="ollama")
        success_ollama = vector_store_ollama.add_materials(test_materials)
        logger.info(f"Ollama сохранение успешно: {success_ollama}")
        
        # Тестируем с OpenAI
        logger.info("Сохранение с OpenAI эмбеддингами...")
        vector_store_openai = VectorStore(embedding_type="openai")
        success_openai = vector_store_openai.add_materials(test_materials)
        logger.info(f"OpenAI сохранение успешно: {success_openai}")
        
        return {
            'ollama_dimension': len(ollama_embedding),
            'openai_dimension': len(openai_embedding),
            'ollama_save_success': success_ollama,
            'openai_save_success': success_openai
        }
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании эмбеддингов: {str(e)}")
        return None

if __name__ == "__main__":
    test_embeddings() 