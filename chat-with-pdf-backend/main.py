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
    # Truncate inputs to reduce token usage
    pdf_text = request.pdfText[:500]
    user_input = request.userInput[:200]
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

    # Prompt and payload
    prompt = f"""You are a concise assistant. Below is text from a PDF:

{pdf_text}

Question: "{user_input}"

Answer in 1-2 sentences based on the PDF. If unrelated, note it’s not based on the PDF."""
    payload = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 50,
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
                error_msg = "Hugging Face API rate limit exceeded. Try again later."
            elif "model" in error_msg.lower():
                error_msg = "Model unavailable or access restricted."
            raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")
