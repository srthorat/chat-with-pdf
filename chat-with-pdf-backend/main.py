from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS to allow requests from GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://srthorat.github.io", "*"],  # Replace with your GitHub Pages URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    pdfText: str
    userInput: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        client = Client(host="http://localhost:11434")
        prompt = f"""You are a helpful assistant. Below is the text extracted from a PDF:

{request.pdfText}

The user has asked: "{request.userInput}"

Provide a concise and accurate response based on the PDF content. If the question is unrelated to the PDF, answer generally but note that the response is not based on the PDF."""
        
        response = client.chat(
            model="phi3",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request.userInput}
            ],
            options={"temperature": 0.7}
        )
        reply = response["message"]["content"].strip()
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")