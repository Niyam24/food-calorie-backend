from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os

app = FastAPI()

# Allow your website to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a nutrition AI.
Identify foods, estimate grams, calories, protein, carbs, and fat.
Return clean Markdown.
"""

def pil_to_b64(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), notes: str = Form("")):
    image = Image.open(BytesIO(await file.read()))
    image_b64 = pil_to_b64(image)

    user_msg = "Analyze the food photo."

    if notes.strip():
        user_msg += " Notes: " + notes

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                },
            ],
        )

        result = response.choices[0].message.content
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}
