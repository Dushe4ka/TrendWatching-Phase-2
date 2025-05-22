import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_manager import PromptManager
from logger_config import setup_logger
from config import get_provider_config, get_api_key, CURRENT_PROVIDER

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логгер
logger = setup_logger("llm_client")

class BaseLLMClient:
    """Базовый класс для работы с LLM"""
    
    def __init__(self):
        """Инициализация базового клиента"""
        self.prompt_manager = PromptManager()
        logger.info(f"{self.__class__.__name__} инициализирован")
    
    def analyze_text(self, prompt: str, query: str) -> Dict[str, Any]:
        """
        Анализ текста с помощью LLM
        
        Args:
            prompt: Промпт для анализа
            query: Текст запроса
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Извлечение ключевых слов из запроса
        
        Args:
            query: Текст запроса
            
        Returns:
            List[str]: Список ключевых слов
        """
        prompt = self.prompt_manager.get_keywords_extraction_prompt(query)
        response = self.analyze_text(prompt, query)
        keywords = [k.strip() for k in response.get('analysis', '').split(',') if k.strip()]
        return keywords if keywords else [query]
    
    def extract_trend_data(self, text: str, analysis: Dict[str, Any], query: str = "") -> List[Dict[str, Any]]:
        """
        Извлечение данных о трендах из материала
        
        Args:
            text: Текст материала
            analysis: Результат анализа материала
            query: Вопрос пользователя
            
        Returns:
            List[Dict[str, Any]]: Список данных о трендах
        """
        prompt = self.prompt_manager.get_trend_extraction_prompt(text, analysis, query)
        response = self.analyze_text(prompt, text)
        return response.get('trends', [])

    def extract_key_phrases(self, query: str) -> List[str]:
        """
        Извлечение ключевых предложений из запроса
        
        Args:
            query: Текст запроса
            
        Returns:
            List[str]: Список ключевых предложений
        """
        prompt = f"""
        Проанализируй запрос и выдели ключевые предложения, которые наиболее точно отражают суть вопроса.
        Каждое предложение должно быть самодостаточным и содержать важную информацию.
        
        Запрос: {query}
        
        Выдели 2-3 ключевых предложения, которые:
        1. Содержат основную тему запроса
        2. Включают важные детали или условия
        3. Отражают цель запроса
        
        Верни только список предложений, без дополнительных пояснений.
        """
        
        response = self.analyze_text(prompt, query)
        phrases = response.get('analysis', '').split('\n')
        
        # Очистка и фильтрация предложений
        cleaned_phrases = [
            phrase.strip() 
            for phrase in phrases 
            if phrase.strip() and len(phrase.strip()) > 10
        ]
        
        return cleaned_phrases[:3]  # Ограничиваем количество предложений

class DeepseekClient(BaseLLMClient):
    """Клиент для работы с Deepseek API через LangChain"""
    
    def __init__(self):
        """Инициализация клиента Deepseek"""
        super().__init__()
        config = get_provider_config("deepseek")
        self.llm = ChatDeepSeek(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=config["timeout"],
            max_retries=config["max_retries"]
        )
        logger.info("DeepseekClient инициализирован")
    
    def analyze_text(self, prompt: str, query: str) -> Dict[str, Any]:
        """
        Анализ текста с помощью Deepseek
        
        Args:
            prompt: Промпт для анализа
            query: Текст запроса
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            system_prompt = """Ты - аналитик трендов. 
            Отвечай простым текстом на русском языке.
            НЕ используй markdown разметку, эмодзи или специальные символы.
            Используй только обычный текст с переносами строк.
            Для структурирования используй нумерованные списки (1., 2., 3.) или маркированные списки (-).
            """
            
            messages = [
                ("system", system_prompt),
                ("human", prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                'analysis': response.content,
                'model': 'deepseek-chat'
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе текста: {str(e)}")
            return {
                'analysis': f"Ошибка при анализе: {str(e)}",
                'model': 'deepseek-chat'
            }

class OpenAIClient(BaseLLMClient):
    """Клиент для работы с OpenAI API через LangChain"""
    
    def __init__(self):
        """Инициализация клиента OpenAI"""
        super().__init__()
        config = get_provider_config("openai")
        self.llm = ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=config["timeout"],
            max_retries=config["max_retries"]
        )
        logger.info("OpenAIClient инициализирован")
    
    def analyze_text(self, prompt: str, query: str) -> Dict[str, Any]:
        """
        Анализ текста с помощью OpenAI
        
        Args:
            prompt: Промпт для анализа
            query: Текст запроса
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            system_prompt = """Ты - аналитик трендов. 
            Отвечай простым текстом на русском языке.
            НЕ используй markdown разметку, эмодзи или специальные символы.
            Используй только обычный текст с переносами строк.
            Для структурирования используй нумерованные списки (1., 2., 3.) или маркированные списки (-).
            """
            
            messages = [
                ("system", system_prompt),
                ("human", prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                'analysis': response.content,
                'model': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе текста: {str(e)}")
            return {
                'analysis': f"Ошибка при анализе: {str(e)}",
                'model': 'gpt-3.5-turbo'
            }

class GeminiClient(BaseLLMClient):
    """Клиент для работы с Google Gemini API через LangChain"""
    
    def __init__(self):
        """Инициализация клиента Gemini"""
        super().__init__()
        config = get_provider_config("gemini")
        self.llm = ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            max_retries=config["max_retries"]
        )
        logger.info("GeminiClient инициализирован")
    
    def analyze_text(self, prompt: str, query: str) -> Dict[str, Any]:
        """
        Анализ текста с помощью Gemini
        
        Args:
            prompt: Промпт для анализа
            query: Текст запроса
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            system_prompt = """Ты - аналитик трендов. 
            Отвечай простым текстом на русском языке.
            НЕ используй markdown разметку, эмодзи или специальные символы.
            Используй только обычный текст с переносами строк.
            Для структурирования используй нумерованные списки (1., 2., 3.) или маркированные списки (-).
            """
            
            messages = [
                ("system", system_prompt),
                ("human", prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                'analysis': response.content,
                'model': 'gemini-pro'
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе текста: {str(e)}")
            return {
                'analysis': f"Ошибка при анализе: {str(e)}",
                'model': 'gemini-pro'
            }

def get_llm_client(provider: str = None) -> BaseLLMClient:
    """
    Получение клиента LLM для указанного провайдера
    
    Args:
        provider: Название провайдера (deepseek, openai, gemini)
        
    Returns:
        BaseLLMClient: Клиент LLM
    """
    provider = provider or CURRENT_PROVIDER
    
    if provider == "deepseek":
        return DeepseekClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}") 