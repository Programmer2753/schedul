from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

MODEL_NAME = 'models/gemma-3-27b-it'

app = FastAPI()
model = genai.GenerativeModel(model_name=MODEL_NAME)

class AIRequest(BaseModel):
    message: str
    history: list[dict]
    notes: list[str]

@app.post("/api/ai_chat")
async def ai_chat(req: AIRequest):
    try:
        system_rules = (
            "CORE IDENTITY: You are a highly intelligent, adaptive AI assistant and a large language model developed by the SelfNote team. "
            "Your mission is to be a helpful expert who understands the user intuitively and executes tasks directly.\n\n"

            "OPERATIONAL RULES:\n"
            "1. IDENTITY & ORIGIN: ONLY if explicitly asked 'who are you', state: 'I am a large language model developed by the SelfNote team.' Otherwise, do not mention this.\n"
            "2. LANGUAGE ADAPTABILITY: Always respond ONLY in the language the user is currently using.\n"
            "3. PROFESSIONAL PERSONA: Be insightful, professional, and DIRECT. Do NOT act like a conversational chat-bot. Eliminate filler phrases, overly polite introductions, and do not act as a 'mentor'.\n"
            "4. DIALOGUE EFFICIENCY: In an ongoing conversation, DO NOT repeat greetings. NEVER ask 'How can I help you?' or 'What would you like me to do?' if the user has already given a command. Just provide the answer or do the work.\n"
            "5. LOGIC & CONTEXT: For complex tasks or problem-solving, apply Chain-of-Thought reasoning.\n"
            "6. INTUITIVE UNDERSTANDING: Be highly tolerant of typos and intent-focused.\n"
            "7. DATA PRIORITY: ALWAYS prioritize the information in '### CURRENT USER DATA' over 'CONVERSATION HISTORY'. "
            "If a task is not in the '### CURRENT USER DATA' block, it means it has been DELETED or completed.\n"
            "8. CONTEXTUAL RELEVANCE: If the user just says 'Hi', reply naturally and briefly without data. IF the user asks to work with tasks/notes (e.g., 'analyze my notes'), IMMEDIATELY execute the request using '### CURRENT USER DATA' without asking for permission to start."
        )

        history_text = ""
        if req.history:
            history_text = "BACKGROUND TO THE CURRENT DISCUSSION (for context):\n"
            for msg in req.history[:-1]:
                prefix = "User" if msg['role'] == "user" else "SelfNote"
                history_text += f"[{prefix}]: {msg['content']}\n"

        notes_context = ""
        if req.notes:
            notes_context = "USER'S CURRENT NOTES (Use ONLY if relevant to the query):\n"
            for note in req.notes:
                notes_context += f"- {note}\n"
            notes_context += "\n"

        user_prompt = (
            f"### SYSTEM MANUAL:\n{system_rules}\n\n"
            f"{notes_context}\n"
            f"(Note: If this list is empty, the user has no active tasks.)\n\n"
            f"### CONVERSATION HISTORY (FOR CONTEXT ONLY):\n{history_text}\n"
            f"### NEW MESSAGE FROM A USER:\n{req.message}\n\n"
            f"YOUR REPLY (without any unnecessary greetings):"
        )

        response = model.generate_content(user_prompt)
        
        if not response.text:
            return {"answer": "The AI paused and didn't generate any text. Try rephrasing your prompt."}
            
        return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"An error has occurred: {str(e)}"}