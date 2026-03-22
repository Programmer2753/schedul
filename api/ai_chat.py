from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

MODEL_NAME = 'models/gemma-3-27b-it'

# Выносим правила в константу
SYSTEM_RULES = (
    "CORE IDENTITY: You are a highly intelligent, adaptive AI assistant and a large language model developed by the SelfNote team. "
    "Your mission is to be a helpful expert who understands the user intuitively.\n\n"
    "OPERATIONAL RULES:\n"
    "1. IDENTITY & ORIGIN: ONLY if explicitly asked 'who are you', state: 'I am a large language model developed by the SelfNote team.' Otherwise, do not mention this.\n"
    "2. LANGUAGE ADAPTABILITY: Always respond ONLY in the language the user is currently using.\n"
    "3. PROFESSIONAL PERSONA: Act as a 'Modern Mentor.' Your tone should be insightful, professional, and relaxed.\n"
    "4. DIALOGUE EFFICIENCY: In an ongoing conversation, DO NOT repeat greetings (don't say 'Hi' in every message). "
    "However, remain conversational. If the user just says 'Hi', respond naturally without dumping all data immediately.\n"
    "5. LOGIC & CONTEXT: For complex tasks or problem-solving, apply Chain-of-Thought reasoning.\n"
    "6. INTUITIVE UNDERSTANDING: Be highly tolerant of typos and intent-focused.\n"
    "7. DATA PRIORITY: ALWAYS prioritize the information in the 'CURRENT USER DATA' block over conversation history. "
    "If a task is not in the 'CURRENT USER DATA' block, it means it has been DELETED or completed.\n"
    "8. CONTEXTUAL RELEVANCE: Use the 'CURRENT USER DATA' ONLY when the user asks about their tasks, "
    "needs analysis, or when the data is directly relevant to the question. Don't force data into a simple greeting."
)

app = FastAPI()

# Инициализируем модель сразу с системными инструкциями
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_RULES
)

class AIRequest(BaseModel):
    message: str
    history: list[dict]
    notes: list[str]

@app.post("/api/ai_chat")
async def ai_chat(req: AIRequest):
    try:
        # 1. Форматируем историю в нативный формат Gemini API
        formatted_history = []
        for msg in req.history[:-1]:
            # API ожидает роли 'user' и 'model'
            role = "user" if msg['role'] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg['content']]})

        # 2. Создаем сессию чата с уже загруженной историей
        chat = model.start_chat(history=formatted_history)

        # 3. Формируем контекст с данными пользователя (строго соблюдаем нейминг из правил)
        if req.notes:
            notes_str = "\n".join([f"- {note}" for note in req.notes])
            user_data_block = f"### CURRENT USER DATA (Active Tasks/Notes):\n{notes_str}\n"
        else:
            user_data_block = "### CURRENT USER DATA:\n[No active tasks or notes at the moment]\n"

        # 4. Формируем финальное сообщение пользователя
        final_message = f"{user_data_block}\nUser message: {req.message}"

        # 5. Отправляем сообщение в чат
        response = chat.send_message(final_message)
        
        # Проверка на наличие текста (может быть заблокировано safety фильтрами)
        if not response.parts:
            return {"answer": "Ответ был заблокирован фильтрами безопасности или модель не смогла сгенерировать текст."}
            
        return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"An error has occurred: {str(e)}"}