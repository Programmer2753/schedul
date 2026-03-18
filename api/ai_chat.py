from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List

# Используй свой API ключ от Google AI Studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

class AIRequest(BaseModel):
    message: str
    notes: List[str]

@app.post('/api/ai_chat')
async def ai_chat(req: AIRequest):
    # Указываем модель Gemma 3 (в SDK это обычно 'models/gemma-3-27b-it' или аналогично)
    model = genai.GenerativeModel('gemma-3-27b-it')
    
    # Формируем контекст из заметок для модели
    notes_context = "\n".join([f"- {note}" for note in req.notes])
    
    prompt = (
        f"Ты — помощник по планированию в приложении SelfNote.\n"
        f"Контекст (заметки пользователя):\n{notes_context}\n\n"
        f"Вопрос пользователя: {req.message}\n"
        f"Ответь кратко, профессионально и только по делу."
    )

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        # Если модель упадет по лимитам или цензуре
        return {"answer": "Извини, я временно не могу ответить. Попробуй позже."}