from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
from fastapi.middleware.cors import CORSMiddleware
import time
import psutil

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
    # Log memory
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024} MB")
    # Truncate inputs
    pdf_text = request.pdfText[:500]
    user_input = request.userInput[:200]

    for attempt in range(max_retries):
        try:
            client = Client(host="http://localhost:11434")
            prompt = f"""Assistant: You are a concise helper. PDF text:

{pdf_text}

Question: "{user_input}"

Answer in 1-2 sentences based on the PDF. If unrelated, note it’s not based on the PDF."""
            
            response = client.chat(
                model="tinyllama",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                options={"temperature": 0.7, "num_ctx": 512, "num_predict": 50}
            )
            reply = response["message"]["content"].strip()
            return {"reply": reply}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            error_msg = str(e)
            if "model \"tinyllama\" not found" in error_msg:
                error_msg = "TinyLlama model not found. Ensure it’s pulled in the deployment."
            elif "llama runner process has terminated" in error_msg:
                error_msg = "Model failed due to resource limits. Try a smaller PDF or simpler question."
            raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")
