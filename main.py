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

# Short, token-efficient system prompt
SYSTEM_PROMPT = """You are a nutrition assistant.
Given one food photo, briefly:
- list foods,
- estimate grams,
- give calories, protein, carbs, fat per item,
- give totals.
Reply in short Markdown. Be concise.
"""


def pil_to_b64(image):
    """Convert PIL image to base64 data URL (JPEG)."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", optimize=True, quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), notes: str = Form("")):
    # Load image from upload
    image = Image.open(BytesIO(await file.read()))
    image_b64 = pil_to_b64(image)

    # Very short user message to save tokens
    user_msg = "Food photo. Give concise nutrition estimate."
    if notes.strip():
        user_msg += " Notes: " + notes.strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",             # cheap model
            max_completion_tokens=320,       # limit reply length
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
        # Return error text so frontend can show it nicely
        return {"error": str(e)}
