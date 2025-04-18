from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
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

    # Try LLaMA 3.3-70B, fallback to LLaMA 3.2-8B
    models = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.2-8B-Instruct"
    ]

    for model in models:
        for attempt in range(max_retries):
            try:
                client = InferenceClient(api_key=api_key)
                prompt = f"""You are a concise assistant. Below is text from a PDF:

{pdf_text}

Question: "{user_input}"

Answer in 1-2 sentences based on the PDF. If unrelated, note it’s not based on the PDF."""
                
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                reply = completion.choices[0].message.content.strip()
                return {"reply": reply}
            except Exception as e:
                print(f"Attempt {attempt + 1} with {model} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                if model == models[-1]:
                    error_msg = str(e)
                    if "rate limit" in error_msg.lower():
                        error_msg = "Hugging Face API rate limit exceeded. Try again later."
                    elif "model" in error_msg.lower():
                        error_msg = "Model unavailable or access restricted."
                    raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")
                break  # Try next model
