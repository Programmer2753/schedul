from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai

openai.api_key = 'твой-ключ-openai'

app = FastAPI()

class AIRequest(BaseModel):
    message: str
    notes: list  # список заметок

@app.post('/api/ai-chat')
async def ai_chat(req: AIRequest):
    context = "Вот заметки пользователя:\n"
    for note in req.notes:
        context += f"- {note}\n"
    prompt = f"{context}\nВопрос: {req.message}\nОтветь кратко и по теме заметок."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return {"answer": answer}