from typing import Dict, Any, List

class PromptManager:
    """Класс для управления промптами"""
    
    def __init__(self):
        self.prompts = {
            'chunk_analysis': """
                Проанализируй следующие материалы и ответь на вопрос пользователя: "{query}"
                
                Ключевые предложения для анализа:
                {key_phrases}
                
                Материалы для анализа:
                {materials}
                
                ВАЖНО:
                - Фокусируйся на информации, связанной с ключевыми предложениями
                - Учитывай контекст и временные рамки материалов
                - Выделяй наиболее релевантные тренды и инсайты
                - Объясняй, как каждый тренд связан с вопросом пользователя
                
                Структура ответа:
                1. Основные тренды и инсайты
                2. Связь с вопросом пользователя
                3. Временные паттерны
                4. Рекомендации и выводы
            """,
            
            'final_report': """
                На основе анализа материалов сформируй итоговый отчет по вопросу: "{query}"
                
                Ключевые предложения:
                {key_phrases}
                
                Результаты анализа чанков:
                {chunk_analyses}
                
                ВАЖНО:
                - Объедини и структурируй информацию из всех чанков
                - Убери дубликаты и противоречия
                - Сделай акцент на наиболее релевантных трендах
                - Предоставь четкие выводы и рекомендации
                
                Структура отчета:
                1. Краткое резюме
                2. Основные тренды и инсайты
                3. Анализ по временным периодам
                4. Рекомендации и прогнозы
                5. Заключение
            """,
            
            'keywords_extraction': """
                Извлеки ключевые слова и фразы из запроса пользователя: "{query}"
                Фокусируйся только на словах, относящихся к вопросу пользователя.
                
                Учитывай:
                - Основной вопрос/проблема
                - Конкретные аспекты вопроса
                - Временные рамки вопроса
                - Географию вопроса
                - Специфические характеристики вопроса
                
                Игнорируй общие темы, не связанные с вопросом пользователя.
                
                Верни только список слов через запятую, без дополнительного текста.
                Пример: искусственный интеллект, машинное обучение, 2024, глобальный рынок, инновации
            """,
            
            'trend_extraction': """
                Проанализируй материал и извлеки информацию о трендах, отвечая на вопрос пользователя: "{query}"
                
                ВАЖНО:
                - Каждый тренд должен быть связан с вопросом пользователя
                - Объясни, как каждый тренд влияет на ответ на вопрос
                - Используй только релевантные данные из материала
                - Не включай тренды, не относящиеся к вопросу
                
                Анализ материала:
                {analysis}
                
                Материал:
                {text}
                
                Для каждого тренда, связанного с вопросом, предоставь:
                - trend: название тренда
                - relevance: как это влияет на ответ на вопрос "{query}"
                - description: описание в контексте вопроса
                - impact: влияние на ответ на вопрос
                - timeframe: временной горизонт в контексте вопроса
                - evidence: примеры из материала, относящиеся к вопросу
                - action_items: что делать с учетом вопроса пользователя
                
                ВАЖНО: Включай только тренды, связанные с вопросом "{query}".
                Не давай общую информацию о трендах, не относящихся к вопросу.
            """
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Получение промпта с подстановкой параметров
        
        Args:
            prompt_type: Тип промпта
            **kwargs: Параметры для подстановки
            
        Returns:
            str: Готовый промпт
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Неизвестный тип промпта: {prompt_type}")
            
        prompt = self.prompts[prompt_type]
        return prompt.format(**kwargs)
    
    def get_chunk_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """
        Получение промпта для анализа чанка
        
        Args:
            context: Контекст анализа, содержащий материалы, запрос и ключевые предложения
            
        Returns:
            str: Промпт для анализа
        """
        materials_text = "\n\n".join([
            f"Заголовок: {m['title']}\n"
            f"Дата: {m['date']}\n"
            f"Текст: {m['text']}\n"
            f"URL: {m['url']}"
            for m in context['materials']
        ])
        
        return self.prompts['chunk_analysis'].format(
            query=context['query'],
            key_phrases="\n".join(f"- {phrase}" for phrase in context['key_phrases']),
            materials=materials_text
        )
    
    def get_final_report_prompt(self, chunk_analyses: List[Dict[str, Any]], query: str, key_phrases: List[str]) -> str:
        """
        Получение промпта для финального отчета
        
        Args:
            chunk_analyses: Результаты анализа чанков
            query: Исходный запрос
            key_phrases: Ключевые предложения
            
        Returns:
            str: Промпт для финального отчета
        """
        analyses_text = "\n\n".join([
            f"Анализ чанка {i+1}:\n{analysis['analysis']}"
            for i, analysis in enumerate(chunk_analyses)
        ])
        
        return self.prompts['final_report'].format(
            query=query,
            key_phrases="\n".join(f"- {phrase}" for phrase in key_phrases),
            chunk_analyses=analyses_text
        )
    
    def get_keywords_extraction_prompt(self, query: str) -> str:
        """Получение промпта для извлечения ключевых слов"""
        return self.get_prompt('keywords_extraction', query=query)
    
    def get_trend_extraction_prompt(self, text: str, analysis: Dict[str, Any], query: str = "") -> str:
        """Получение промпта для извлечения данных о трендах"""
        return self.get_prompt('trend_extraction', text=text, analysis=analysis['analysis'], query=query) 