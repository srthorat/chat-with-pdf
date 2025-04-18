from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import time

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://srthorat.github.io", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    pdfText: str
    userInput: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    max_retries = 3
    retry_delay = 5
    # No truncation of inputs
    pdf_text = request.pdfText
    user_input = request.userInput
    # Get API key from environment
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Hugging Face API key not configured")

    # API endpoint and headers
    url = "https://router.huggingface.co/nebius/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prompt to enforce PDF-based answers
    prompt = f"""You are an expert assistant specializing in analyzing PDF content. Your task is to provide precise and detailed answers based exclusively on the following PDF text:

{pdf_text}

User question: "{user_input}"

Instructions:
1. If the question can be answered using the PDF text, provide a detailed and accurate response, including specific details, quotes, or references from the PDF to support your answer.
2. If the question is unrelated to the PDF content or cannot be answered based on the provided text, respond exactly with: "This question is outside the PDF content."
3. Do not use external knowledge or assumptions; rely only on the PDF text.
4. Ensure your response is clear, concise, and directly addresses the question."""
    
    payload = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 512,
        "model": "microsoft/Phi-3-mini-4k-instruct-fast",
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            return {"reply": reply}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                error_msg = "Hugging Face API rate limit exceeded. Try again later or reduce input size."
            elif "model" in error_msg.lower():
                error_msg = "Model unavailable or access restricted."
            raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")
